import math
import os

import torch as th

from omnigibson.macros import gm
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class xarm6(ManipulationRobot):

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        fixed_base=True,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        **kwargs,
    ):
        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            **kwargs,
        )

    def _post_load(self):
        super()._post_load()

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("xarm6 does not support discrete actions!")

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        controllers[f"arm_{self.default_arm}"] = "JointController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_joint_pos(self):
        return th.tensor([
                    0.0,
                    -0.576,
                    -1.0,
                    0.0,
                    1.5708,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ])
        #return th.tensor([-87.0, -44.0, 26.0, 20.0, 30.0, -80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.04}


    @property
    def disabled_collision_pairs(self):
        return [
            ["base_link", "link1"],
            ["link1", "link2"],
            ["link2", "link3"],
            ["link3", "link4"],
            ["link4", "link5"],
            ["link5", "link6"],
            ["link6", "xarm_gripper_base_link"],
            ["xarm_gripper_base_link", "left_outer_knuckle"],
            ["xarm_gripper_base_link", "left_finger"],
            ["xarm_gripper_base_link", "left_inner_knuckle"],
            ["xarm_gripper_base_link", "right_outer_knuckle"],
            ["xarm_gripper_base_link", "right_finger"],
            ["xarm_gripper_base_link", "right_inner_knuckle"],
            ["left_outer_knuckle", "left_finger"],
            ["left_outer_knuckle", "left_inner_knuckle"],
            ["left_finger", "left_inner_knuckle"],
            ["right_outer_knuckle", "right_finger"],
            ["right_outer_knuckle", "right_inner_knuckle"],
            ["right_finger", "right_inner_knuckle"],
            ["link_tcp","right_finger"],
            ["link_tcp","left_finger"],
        ]

    @property
    def arm_link_names(self):
        return {
            self.default_arm: [
                "base_link",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "xarm_gripper_base_link",
            ]
        }

    @property
    def arm_joint_names(self):
        return {
            self.default_arm: [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
        }

    @property
    def eef_link_names(self):
        return {self.default_arm: "link_tcp"}

    @property
    def finger_link_names(self):
        return {
            self.default_arm: [
                "right_finger",
                "left_finger",
            ]
        }
    

    @property
    def assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="right_finger", position=th.tensor([0.0, -0.022, -0.0256])),
                GraspingPoint(link_name="right_finger", position=th.tensor([0.0, -0.022, -0.05636])),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="left_finger", position=th.tensor([0.0, 0.022, -0.0256])),
                GraspingPoint(link_name="left_finger", position=th.tensor([0.0, 0.022, -0.05636])),
            ]
        }
    

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["drive_joint", "left_inner_knuckle_joint", "right_inner_knuckle_joint", "right_outer_knuckle_joint", "right_finger_joint", "left_finger_joint"]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/xarm6/xarm6.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/xarm6/xarm6.urdf")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/xarm6/xarm6_descriptor.yaml")}
