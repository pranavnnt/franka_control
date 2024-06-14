import os
import time

import argparse
import glob
import numpy as np
import yaml
time.sleep

from franka_env import FrankaEnv
from util import Rate, TIME, HZ, HOMES


parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--task", type=str, default="cloth")

if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    task = args.task

    home = HOMES[task]
    env = FrankaEnv(home=home, hz=HZ, gain_type="record", camera=False)

    while True:
        print(env.robot.get_joint_positions())
        time.sleep(1)
