#!/usr/bin/env python3
'''
========================================================================
Usage Instructions
========================================================================
1. Start the script; OmniGibson window pops up.
2. Press and release  Z  (keyboard) once:
   - The pen is teleported above the BLACK holder and dropped naturally.
3. Press and release  Z  again:
   - The pen is teleported above the WHITE holder and dropped naturally.
4. Press  Z  a third time to quit.
========================================================================
'''

SCENE_FILE = "../xarm6_tasks/xarm6_envs/short_2.json"
OUT_DIR = ""
PEN_USD = "../assets/tennis/tennis.usd"

import os, yaml, numpy as np, torch as th
import omnigibson as og
from omnigibson import lazy

def main():
    # Load environment
    cfg = yaml.safe_load(open("../xarm6_tasks/rekep/configs/config_xarm6.yaml"))["env"]
    cfg["scene"]["scene_file"] = SCENE_FILE
    cfg["objects"] = [{
        "type": "USDObject", "name": "tennis1",
        "usd_path": PEN_USD, "scale": [0.04] * 3,
        "position": [0.2, -0.23, 1.361],
        "orientation": [-0.173, -0.688, 0.207, 0.673],
    }]
    env = og.Environment(dict(scene=cfg["scene"], env=cfg["og_sim"]))
    for _ in range(35):
        og.sim.step()

    holder1_pos = np.array(env.scene.object_registry("name", "holder_1").get_position_orientation()[0])
    holder2_pos = np.array(env.scene.object_registry("name", "holder_2").get_position_orientation()[0])
    pen = env.scene.object_registry("name", "pen")

    # Move 1: above BLACK holder
    pen.set_position_orientation(position=holder1_pos + np.array([0.02, 0, 0.16]), orientation=[0, 1, 0, 0])
    while not lazy.carb.input.is_keyboard_input(lazy.carb.input.KeyboardInput.Z):
        og.sim.step()

    # Move 2: above WHITE holder
    pen.set_position_orientation(position=holder2_pos + np.array([0.02, 0, 0.16]), orientation=[0, 1, 0, 0])
    while not lazy.carb.input.is_keyboard_input(lazy.carb.input.KeyboardInput.Z):
        og.sim.step()

    # Quit on third Z
    while not lazy.carb.input.is_keyboard_input(lazy.carb.input.KeyboardInput.Z):
        og.sim.step()

if __name__ == "__main__":
    main()