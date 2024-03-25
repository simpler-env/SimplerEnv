import time

import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer


def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 500.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True

    robot: sapien.Articulation = loader.load(
        "ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/wx250s.urdf"
    )
    print(robot.get_links())
    robot.set_root_pose(sapien.Pose([0, 0, 0.2], [1, 0, 0, 0]))
    print([x.name for x in robot.get_active_joints()])
    print(robot.get_qlimits())

    # Set initial joint positions
    qpos = np.array(
        [
            -0.01840777,
            0.0398835,
            0.22242722,
            -0.00460194,
            1.36524296,
            0.00153398,
            0.037,
            0.037,
        ]
    )
    # qpos = np.array([-0.00153398,  0.04448544,  0.21629129, -0.00306796,  1.36524296, 0.,
    #                  0.015, 0.015])
    # qpos = np.array([-0.13192235, -0.76238847,  0.44485444, -0.01994175,  1.7564081,  -0.15953401,
    #                  0.015, 0.015])
    robot.set_qpos(qpos)
    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=1e5, damping=1e3)

    while not viewer.closed:
        print(robot.get_qpos())
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            print("target qpos", qpos)
            print("current qpos", robot.get_qpos())
            robot.set_drive_target(qpos)
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    demo(fix_root_link=True, balance_passive_force=True)


if __name__ == "__main__":
    main()

    """
    [Actor(name="base_link", id="2"), Actor(name="shoulder_link", id="3"), Actor(name="upper_arm_link", id="4"), Actor(name="upper_forearm_link", id="5"),
    Actor(name="lower_forearm_link", id="6"), Actor(name="wrist_link", id="7"), Actor(name="gripper_link", id="8"), Actor(name="ee_arm_link", id="9"),
    Actor(name="gripper_prop_link", id="15"), Actor(name="gripper_bar_link", id="10"), Actor(name="fingers_link", id="11"),
    Actor(name="left_finger_link", id="14"), Actor(name="right_finger_link", id="13"), Actor(name="ee_gripper_link", id="12")]
    active_joints:
        ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'left_finger', 'right_finger']
    joint_limits:
        [[-3.1415927  3.1415927]
        [-1.8849556  1.9896754]
        [-2.146755   1.6057029]
        [-3.1415827  3.1415827]
        [-1.7453293  2.146755 ]
        [-3.1415827  3.1415827]
        [ 0.015      0.037    ]
        [ 0.015      0.037    ]]
    """
