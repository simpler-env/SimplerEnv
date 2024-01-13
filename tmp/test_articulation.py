import time
from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer


def demo(filename):
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

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(filename)
    articulation.set_pose(sapien.Pose([0, 0, -0.865], [1, 0, 0, 0]))

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
    camera.set_parent(parent=articulation.get_links()[0], keep_pose=False)

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    demo("./ManiSkill2_real2sim/data/mk_station.urdf")


if __name__ == "__main__":
    main()
