import os
import torch as th
import yaml
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler
TEST_OUT_PATH = "/home/dell/workspace/xwj/simworld/env_build"  # Define output directory here.

    
def main():
    config_filename = "/home/dell/workspace/xwj/simworld/xarm6_rekep/rekep/configs/config_xarm6.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)['env']
    scene_file = "/home/dell/workspace/xwj/simworld/xarm6_rekep/xarm6_envs/short_2.json"
    config['scene']['scene_file'] = scene_file
    config["objects"] = [
                {
            "type": "USDObject",
            "name": "tennis1",
            "usd_path": "/home/dell/workspace/xwj/simworld/assets/tennis/tennis.usd",
            "scale": [
                        0.03999999910593033,
                        0.03999999910593033,
                        0.03999999910593033
                    ],
            "position": [0.2,
                        -0.23,
                        1.3611752557754517], 
            "orientation": [-0.17277531325817108,
                        -0.6883426904678345,
                        0.206947922706604,
                        0.6734283566474915],
        },
    ]
    '''{
            "type": "USDObject",
            "name": "pencil_holder_3",
            "usd_path": "/home/dell/workspace/xwj/two_drawer_blender.usd",
            "scale": [
                        0.25,
                        0.25,
                        0.25
                    ],
            "position": [0,
                        0,
                        0.1411752557754517], 
            "orientation": [-1.0058e-07, -1.6804e-07, -7.0709e-01,  7.0713e-01],
        },'''
    '''config['objects'] = [
            {
                "type": "DatasetObject",
                "name": "desk",
                "category": "desk",
                "model": "ccxuey",
                "scale": [
                            1.4,
                            1.3,
                            1.1
                        ],
                "position": [
                        0.62,
                        0,
                        0.810266375541687
                    ],
                "orientation": [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ],
        },
        ]
    
    config['scene']['robot'] = [
    {
        "name": "xarm6",
        "type": "xarm6",
        "obs_modalities": [
            "rgb"
        ],
        "action_modalities": [
            "continuous"
        ],
        "action_normalize": False,
        "position": [
                        0,
                        0,
                        1.0
                    ],
        "orientation": [
                        0,
                        0,
                        0,
                        1.0
                    ],
        "grasping_mode": "physical",
        "controller_config": {
            "arm_0": {
                "name": "OperationalSpaceController",
                "kp": 250,
                "kp_limits": [
                    50,
                    400
                ]
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": [
                    -1.0,
                    1.0
                ],
                "mode": "binary"
            }
        }
    }
    ]'''

    #env = og.Environment(dict(scene=config['scene'], objects=config["objects"], robots=config['scene']['robot'], env=config['og_sim']))
    #env = og.Environment(dict(scene=config['scene'], objects=config["objects"], env=config['og_sim']))
    env = og.Environment(dict(scene=config['scene'], env=config['og_sim']))
    scene = env.scene
    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_state()
    env.scene.reset()

    og.sim.viewer_camera.set_position_orientation(
        th.tensor([ 0.9161, -0.0056,  1.8559]), th.tensor([0.2015, 0.2056, 0.6840, 0.6703])
    )

    modalities_required = ["depth"]
    for modality in modalities_required:
        og.sim.viewer_camera.add_modality(modality)
    # Let the object settle
    for _ in range(30):
        og.sim.step()

    cam = og.sim.enable_viewer_camera_teleoperation()
    def complete_loop():
        nonlocal completed
        completed = True

    KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.Z, complete_loop)

    completed = False
    while not completed:
        og.sim.step()
    print(og.sim.viewer_camera.get_position_orientation())

    print(env.robots[0].get_eef_position(), env.robots[0].get_eef_orientation())
    save_path = os.path.join(TEST_OUT_PATH, "saved_stage.json")
    og.sim.save(json_paths=[save_path])

    print("Re-loading scene...")
    og.clear()
    og.sim.restore(scene_files=[save_path])
    og.sim.step()
    og.sim.play()
    # env is no longer valid after og.clear()
    del env
    completed = False
    while not completed:
        og.sim.step()




if __name__ == "__main__":
    main()