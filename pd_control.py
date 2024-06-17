import argparse
from typing import Dict
import time

import numpy as np
import torch
import math

import glob
import numpy as np
from franka_control.franka_env import FrankaEnv

from franka_control.util import HOMES
from franka_control.util import R, T

import torchcontrol as toco

class PDControl(toco.PolicyModule):
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