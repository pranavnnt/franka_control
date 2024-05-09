import os
import time

import argparse
import glob
import torch
import numpy as np
import yaml
time.sleep

from franka_env import FrankaEnv
from util import Rate, TIME, HZ, HOMES
from util import quat_conj, quat_mult, quat_rot
from util import T, R


parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--task", type=str, default="cloth")


def _get_filename(dir, input, task):
    index = 0
    for name in glob.glob("{}/{}_{}_*.npz".format(dir, input, task)):
        n = int(name[:-4].split("_")[-1])
        if n >= index:
            index = n + 1
    return "{}/{}_{}_{}.npz".format(dir, input, task, index)


if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    task = args.task

    home = HOMES[task]
    env = FrankaEnv(home=home, hz=HZ, gain_type="record", camera=False)

    ee_pos_home, ee_rot_home = env.robot.robot_model.forward_kinematics(env.robot.get_joint_positions())
    ee_quat_home = R.from_quat(ee_rot_home)
    print("Home pos: ", ee_pos_home)
    print("Home rot: ", ee_rot_home)

    #home eef frame
    T_home = T.from_rot_xyz(
                    rotation=R.from_quat(ee_rot_home),
                    translation=ee_pos_home)

    ee_rot_home_conj = quat_conj(ee_rot_home)

    while True:
        filename = _get_filename("data", name, task)

        user_in = "r"
        while user_in == "r":
            # time.sleep(5)
            print("Current eef pose : ", env.robot.get_ee_pose())
            # env.robot.robot_model.urdf_path
            env.reset()
            user_in = input("Press [ENTER] to record {}".format(filename))

        print("Started recording ...")

        rel_pose_hist = []
        for state in range(int(TIME * HZ) - 1):
            # get end effector pose
            current_pos = env.robot.get_ee_pose()[0]
            current_rot = env.robot.get_ee_pose()[1]
            current_quat = R.from_quat(current_rot)

            # compute relative position in home frame
            print("EE pos : ", current_pos)
            current_pose = T.from_rot_xyz(rotation=current_quat, translation=current_pos)
            # print(T_home.inv())
            rel_pose = T_home.inv() * current_pose
            rel_pose_hist.append(rel_pose)
            print("Current relative pose: ", rel_pose.translation())
            print("Current relative orientation: ", rel_pose.rotation())
            print("----------------------------------------------------")
            time.sleep(1 / HZ)
        env.close()

        print("Recording complete.")

        print()

        if not os.path.exists("./data"):
            os.mkdir("data")
        np.savez(filename, home=home, hz=HZ, traj_pose=rel_pose_hist)
