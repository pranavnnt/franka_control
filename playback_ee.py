# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Dict
import time

import numpy as np
import torch
import math


import glob
import numpy as np
from franka_env import FrankaEnv

from util import HOMES
from util import R, T

import torchcontrol as toco

parser = argparse.ArgumentParser()
parser.add_argument("file")

class MyPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, ee_pos_current, ee_rot_current, kx, kxd, robot_model, **kwargs):
        """
        Args:
            joint_pos_current (torch.Tensor):   Joint positions at initialization
            kx, kxd (torch.Tensor):             PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=True
        )

        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_rot_desired = torch.nn.Parameter(ee_rot_current)

        # Initialize modules
        self.feedback = toco.modules.CartesianSpacePDFast(kx, kxd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        ee_pos_current, ee_rot_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current

        # Execute PD control
        wrench_feedback = self.feedback(
            ee_pos_current, ee_rot_current, ee_twist_current,
            self.ee_pos_desired, self.ee_rot_desired, torch.zeros(6),
        )

        torque_feedback = jacobian.T @ wrench_feedback

        #torque_feedforward = self.invdyn(
        #   joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        #)  # coriolis
        #print(torque_feedforward)

        torque_feedforward = torch.zeros_like(torque_feedback)

        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}

def _separate_filename(filename):
    split = filename[:-4].split("_")
    name = "_".join(split[:-1:])
    i = int(split[-1])
    return name, i


def _format_out_dict(list_obs, actions, hz, home):
    out_dict = {k: [] for k in list(list_obs[0].keys())}
    for obs in list_obs:
        for k in out_dict.keys():
            out_dict[k].append(obs[k])
    out_dict = {k: np.array(v) for k, v in out_dict.items()}

    out_dict["actions"] = actions
    out_dict["rate"] = hz
    out_dict["home"] = home
    return out_dict

if __name__ == "__main__":
    # Initialize robot interface
    args = parser.parse_args()

    name, i = _separate_filename(args.file)
    num_files = len(glob.glob("data/{}_*.npz".format(name)))
    gain_type = (
        "stiff" if name.endswith("insertion") or name.endswith("zip") else "default"
    )

    data = np.load("data/" + args.file, allow_pickle=True)
    home, rel_pose_hist, hz = data["home"], data["traj_pose"], data["hz"]
    env = FrankaEnv(home=HOMES["cloth"], hz=hz, gain_type=gain_type, camera=False)

    ee_pos_home, ee_rot_home = env.robot.robot_model.forward_kinematics(env.robot.get_joint_positions())
    print("Home pos: ", ee_pos_home)
    print("Home rot: ", ee_rot_home)
    
    #home eef frame
    T_home = T.from_rot_xyz(
                    rotation=R.from_quat(ee_rot_home),
                    translation=ee_pos_home)

    joint_vel_limits = env.robot.robot_model.get_joint_velocity_limits()
    print("----------------------------------------------------")
    print("Joint velocity limits: ", joint_vel_limits)
    print("----------------------------------------------------")

    for i in range(i, num_files):
        data = np.load("data/{}_{}.npz".format(name, i))

        user_in = "r"
        while user_in == "r":
            obs = [env.reset()]
            user_in = input("Ready. Loaded {} ({} hz):".format(name, hz))
        actions = []

        # Create policy instance
        ee_pos_initial, ee_rot_initial = env.robot.robot_model.forward_kinematics(env.robot.get_joint_positions())
        default_kx = torch.Tensor(env.robot.metadata.default_Kx)
        default_kxd = torch.Tensor(env.robot.metadata.default_Kxd)
        policy = MyPDPolicy(
            ee_pos_current=ee_pos_initial,
            ee_rot_current=ee_rot_initial,
            kx=default_kx,
            kxd=default_kxd,
            robot_model=env.robot.robot_model,
        )

        # Send policy
        print("\nRunning PD policy...")
        env.robot.send_torch_policy(policy, blocking=False)

        # Update policy to execute a sine trajectory on joint 6 for 5 seconds
        print("Starting playback updates...")
        # ee_pos_desired = ee_pos_initial.clone()
        # ee_rot_desired = ee_rot_initial.clone()

        time_to_go = 50.0
        m = 0.07  # magnitude of sine wave (rad)
        # T = 0.5  # period of sine wave
        hz = 30  # update frequency
        for i in range(len(rel_pose_hist)):
            T_frame = T_home * rel_pose_hist[i] 
            print("Translation: ", T_frame.translation())
            print("Rotation: ", T_frame.rotation().as_quat())
            env.robot.update_current_policy({"ee_pos_desired": T_frame.translation(), "ee_rot_desired": T_frame.rotation().as_quat()})
            # print(f"Desired position: {ee_pos_desired}")
            # print("Current robot pos : %s", env.robot.get_ee_pos()[0])
            time.sleep(1 / hz)

        print("Terminating PD policy...")
        state_log = env.robot.terminate_current_policy()
