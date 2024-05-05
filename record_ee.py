import os
import time

import argparse
import glob
import numpy as np
import yaml
time.sleep

from franka_env import FrankaEnv
from util import Rate, TIME, HZ, HOMES
from util import quat_conj, quat_mult, quat_rot


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

    ee_pos_home, ee_quat_home = env.robot.robot_model.forward_kinematics(env.robot.get_joint_positions())
    print("Home pose: ", ee_pos_home)
    print("Home quat: ", ee_quat_home)

    ee_quat_home_conj = quat_conj(ee_quat_home)

    while True:
        filename = _get_filename("data", name, task)

        user_in = "r"
        while user_in == "r":
            env.reset()
            user_in = input("Press [ENTER] to record {}".format(filename))

        print("Started recording ...")

        ee_pos = []
        ee_quat = []
        ee_pose_rel = []
        ee_quat_rel = []
        for state in range(int(TIME * HZ) - 1):
            ee_pos.append(env.robot.get_ee_pose()[0])
            ee_quat.append(env.robot.get_ee_pose()[1])
            ee_pose_rel.append(env.robot.get_ee_pose()[0] - ee_pos_home)
            ee_quat_rel.append(quat_mult(env.robot.get_ee_pose()[1], ee_quat_home_conj))
            print("Current relative pose: ", ee_pose_rel[-1], ee_quat_home)
            print("Current relative orientation: ", ee_quat_rel[-1])
            print("----------------------------------------------------")
            time.sleep(1 / HZ)
        env.close()

        print("Recording complete.")

        print()

        if not os.path.exists("./data"):
            os.mkdir("data")
        np.savez(filename, home=home, hz=HZ, traj_pose=ee_pose_rel, traj_quat=ee_quat_rel)
