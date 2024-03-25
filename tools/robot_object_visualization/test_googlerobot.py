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
        "ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"
    )
    # robot: sapien.Articulation = loader.load("ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip_recolor_cabinet_visual_matching_1.urdf")
    print(robot.get_links())
    robot.set_root_pose(sapien.Pose([0, 0, 0.06205], [1, 0, 0, 0]))

    # Set initial joint positions
    qpos = [
        -0.2639457174606611,
        0.0831913360274175,
        0.5017611504652179,
        1.156859026208673,
        0.028583671314766423,
        1.592598203487462,
        -1.080652960128774,
        0,
        0,
        -0.00285961,
        0.7851361,
    ]
    robot.set_qpos(qpos)
    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=1e5, damping=1e3)

    camera = scene.add_camera(
        name="camera",
        width=int(848),
        height=int(480),
        fovy=np.deg2rad(78.0),  # D435 fovy
        near=0.1,
        far=10.0,
    )
    camera.set_focal_lengths(605.12, 604.91)
    camera.set_principal_point(424.59, 236.67)
    link_camera = [x for x in robot.get_links() if x.name == "link_camera"][0]
    camera.set_parent(parent=link_camera, keep_pose=False)
    camera.set_local_pose(
        sapien.Pose.from_transformation_matrix(np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
    )  # SAPIEN uses ros camera convention; the rotation matrix of link_camera's pose is in opencv convention, so we need to transform it to ros convention

    tcp_link = [x for x in robot.get_links() if x.name == "link_gripper_tcp"][0]
    while not viewer.closed:
        # print(robot.get_qpos())
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            # print("target qpos", qpos)
            # print("current qpos", robot.get_qpos())
            # print("tcp pose wrt robot base", robot.pose.inv() * tcp_link.pose)
            robot.set_drive_target(qpos)
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    demo(fix_root_link=True, balance_passive_force=True)


if __name__ == "__main__":
    main()
    """
    robot.qpos 13-dim if mobile else 11
    robot qlimits
        array([[     -inf,       inf],
       [     -inf,       inf],
       [-4.49e+00,  1.35e+00],
       [-2.66e+00,  3.18e+00],
       [-2.13e+00,  3.71e+00],
       [-2.05e+00,  3.79e+00],
       [-2.92e+00,  2.92e+00],
       [-1.79e+00,  1.79e+00],
       [-4.49e+00,  1.35e+00],
       [-1.00e-04,  1.30e+00], # gripper plus direction = close
       [-1.00e-04,  1.30e+00],
       [-3.79e+00,  2.22e+00],
       [-1.17e+00,  1.17e+00]], dtype=float32)
    robot.get_active_joints()
        ['joint_wheel_left', 'joint_wheel_right', 'joint_torso', 'joint_shoulder',
        'joint_bicep', 'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper',
        'joint_finger_right', 'joint_finger_left', 'joint_head_pan', 'joint_head_tilt']
    If robot is not mobile, then the first two joints are not active
    """
