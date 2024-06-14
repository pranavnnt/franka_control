import os
import time

import argparse
import glob
import numpy as np
import yaml
time.sleep

from franka_env import FrankaEnv
from util import Rate, TIME, HZ, HOMES

import torch


parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--task", type=str, default="cloth")

if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    task = args.task

    home = HOMES[task]
    env = FrankaEnv(home=home, hz=HZ, gain_type="default", camera=False)

    ee_pos_home, ee_quat_home = env.robot.robot_model.forward_kinematics(env.robot.get_joint_positions())
    print("Home pose: ", ee_pos_home)
    print("Home quat: ", ee_quat_home)

    ee_pos_home = torch.Tensor([0.6993193011745447, -0.29078151793354506, 0.2769424484234622])
    ee_pos_home[2] = ee_pos_home[2] + 0.1

    env.robot.update_current_policy({"ee_pos_desired": torch.Tensor(ee_pos_home)})
