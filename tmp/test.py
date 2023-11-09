import sapien.core as sapien

import numpy as np



def main():

    engine = sapien.Engine()

    # renderer = sapien.SapienRenderer()

    # engine.set_renderer(renderer)

    scene = engine.create_scene()

    dt = 1 / 100.0

    scene.set_timestep(dt)


    # ---------------------------------------------------------------------------- #

    # Add two boxes

    # ---------------------------------------------------------------------------- #

    actor_builder = scene.create_actor_builder()

    actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])

    # actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1, 0, 0])

    box1 = actor_builder.build_kinematic(name='box1')

    box1.set_pose(sapien.Pose(p=[0, 0, 1.0]))

    print('Mass of box1:', box1.mass)


    actor_builder = scene.create_actor_builder()

    actor_builder.add_box_collision(half_size=[0.25, 0.25, 0.25])

    # actor_builder.add_box_visual(half_size=[0.25, 0.25, 0.25], color=[0, 1, 0])

    box2 = actor_builder.build(name='box2')

    box2.set_pose(sapien.Pose(p=[0, 0, 1.75]))

    print('Mass of box2:', box2.mass)


    # ---------------------------------------------------------------------------- #

    # Check contacts

    # ---------------------------------------------------------------------------- #

    scene.step()

    contacts = scene.get_contacts()

    support_force = 0

    for contact in contacts:

        print(contact)

        for point in contact.points:

            print('Impulse (F * dt) on the first actor:', point.impulse)

            print('Normal (same direction as impulse):', point.normal)

            print('Contact position (in the world frame):', point.position)

            print('Minimum distance between two shapes:', point.separation)

            if contact.actor0.name == 'box2':

                support_force += point.impulse[2] / dt

            elif contact.actor0.name == 'box1':

                support_force -= point.impulse[2] / dt

            else:

                raise RuntimeError('Impossible case in this example.')

    # Sanity check: the support force should balance the gravity

    np.testing.assert_allclose(support_force, 9.81 * box2.mass, rtol=1e-3)



if __name__ == '__main__':

    main()

