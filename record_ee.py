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

    while True:
        filename = _get_filename("data", name, task)

        user_in = "r"
        while user_in == "r":
            print("Going to start recording {}".format(filename))
            env.reset()
            user_in = input("Move to the initial pose and press [ENTER].")

        ee_pos_init, ee_rot_init = env.robot.get_ee_pose()
        ee_quat_init = R.from_quat(ee_rot_init)
        print("Home pos: ", ee_pos_init)
        print("Home rot: ", ee_rot_init)

        #home eef frame
        T_home = T.from_rot_xyz(
                        rotation=R.from_quat(ee_rot_init),
                        translation=ee_pos_init)

        user_in = "r"
        while user_in == "r":
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
        np.savez(filename, hz=HZ, traj_pose=rel_pose_hist)
