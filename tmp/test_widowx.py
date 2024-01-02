import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import time


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
    
    robot: sapien.Articulation = loader.load("/home/xuanlin/Real2Sim/ManiSkill2_real2sim/mani_skill2/assets/descriptions/widowx_description/wx250s.urdf")
    print(robot.get_links())
    robot.set_root_pose(sapien.Pose([0, 0, 0.2], [1, 0, 0, 0]))
    print([x.name for x in robot.get_active_joints()])
    print(robot.get_qlimits())

    # Set initial joint positions
    qpos = np.array([-0.13192235, -0.76238847,  0.44485444, -0.01994175,  1.7564081,  -0.15953401,
                     0.015, 0.015])
    qpos = np.array([0, 0, 0, -np.pi, np.pi / 2, 0,
                     0.015, 0.015])
    # qpos = np.array([-0.39407882,  0.05721467, -0.32068512, -3.0768952 ,  1.0481696 ,
    #                  -3.1414535 ,  0.02599986,  0.02600013])
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
    demo(fix_root_link=True,
         balance_passive_force=True)


if __name__ == '__main__':
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
        [-1.6057029  2.146755 ]
        [-3.1415927  3.1415927]
        [-2.146755   1.7453293]
        [-3.1415927  3.1415927]
        [ 0.015      0.037    ]
        [ 0.015      0.037    ]]
    """
    
    
"""
tf.Tensor(
[ 0.29806843 -0.11465699  0.10782038  0.04275148 -0.14888743 -0.31455365
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.29785287 -0.11455903  0.10827518  0.04278725 -0.14889328 -0.3144694
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.29529363 -0.11402838  0.10024154  0.04370506 -0.12329954 -0.30710596
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.27248308 -0.10138711  0.09138915  0.0508096  -0.07173306 -0.09391912
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.24823941 -0.0811879   0.09790358  0.08269297 -0.11706779  0.15128727
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22304264 -0.06334073  0.09159752  0.09572256 -0.14960448  0.36500037
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.20875835 -0.04440302  0.07813202  0.05387358 -0.1205819   0.62855744
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.19996543 -0.02887496  0.073396    0.00277641 -0.11770173  0.88297635
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.1946491  -0.01116679  0.06324816 -0.01838067 -0.08065976  1.0252526
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 1.9829217e-01  9.8673906e-04  5.9709169e-02 -4.1111078e-02
 -7.1967825e-02  1.0576786e+00  1.0001532e+00], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.20370732  0.01017785  0.05182235 -0.05654271 -0.05470593  1.0557181
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.21565837  0.01779467  0.04834643 -0.07132344 -0.03498556  1.0397375
  0.9994886 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22297677  0.02933743  0.04380894 -0.09354865 -0.02507365  1.011129
  0.9994886 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22714865  0.0457576   0.0470006  -0.09128537 -0.06947262  0.93128586
  0.9994886 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22840336  0.06180144  0.04813604 -0.05977548 -0.09688041  0.879188
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22890148  0.07768954  0.048669   -0.02057483 -0.10894583  0.8200851
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.23880063  0.09118433  0.05060545  0.03750009 -0.13101928  0.69535744
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.24353023  0.1076001   0.05164775  0.04086299 -0.13797843  0.65432376
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.24376857  0.11273084  0.0494793   0.04601268 -0.13167101  0.6528622
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.24224146  0.12236504  0.04823136  0.04898258 -0.13174039  0.6600122
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2507077   0.12959582  0.04735126  0.05555115 -0.1355397   0.63680494
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.25908503  0.13650663  0.04458536  0.07772507 -0.13646534  0.60074306
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2684987   0.14177048  0.04270804  0.07637818 -0.13630357  0.5658602
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.27579528  0.14782624  0.03817681  0.07922222 -0.13182658  0.5549625
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2843842   0.15495905  0.04006465  0.08096801 -0.14492877  0.55150914
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.29281217  0.1625134   0.03947584  0.08569371 -0.15496475  0.54383963
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.29850808  0.16434242  0.03721014  0.08071601 -0.15660357  0.538345
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.30172914  0.16794588  0.03813419  0.08978813 -0.18639836  0.5386496
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.29912996  0.16320486  0.03683846  0.09627592 -0.16873643  0.53748536
  0.9994886 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2935191   0.15329115  0.04873599  0.09387705 -0.17896824  0.52981085
  0.9994886 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.28071576  0.13859238  0.05578263  0.08551347 -0.15968022  0.5200476
  0.9994886 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.27311552  0.12127746  0.06554092  0.08567463 -0.15404579  0.45860356
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2677869   0.09697691  0.07652965  0.08009309 -0.13356425  0.37086317
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.25109974  0.08510971  0.09249425  0.06891943 -0.1293009   0.32619384
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.23986585  0.08248477  0.11026014  0.0845815  -0.14127682  0.3183104
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22684619  0.07757747  0.13265225  0.08739373 -0.15984915  0.31357828
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.21645054  0.07457931  0.14384234  0.08520467 -0.18688117  0.3242573
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2163359   0.07491125  0.14384234  0.08520467 -0.18688117  0.3257913
  1.0001532 ], shape=(7,), dtype=float32)
  
  
  ***
  
Episode 1:
tf.Tensor(
[ 0.32279104 -0.02118294  0.10182465 -0.17781603 -0.07830916  0.92143136
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.32263812 -0.02116953  0.10147302 -0.17779927 -0.07833081  0.9214926
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.32269126 -0.02117206  0.09343403 -0.1577201  -0.06394028  0.9230828
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.32882315 -0.01661764  0.0793895  -0.12339178 -0.05369333  0.93491197
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.33928514 -0.00367324  0.06859627 -0.1083837  -0.04356351  0.9626468
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.35203257  0.00418845  0.05980398 -0.10522091 -0.03459513  0.99152076
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.36499232  0.01324488  0.04976716 -0.08840615 -0.05829217  1.0142933
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.3755168   0.0179571   0.03874521 -0.05821661 -0.0535942   1.0159351
  1.0001532 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.3784917   0.01799399  0.02234769 -0.0156917  -0.03717303  1.0111723
  0.94534314], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.37925002  0.00401973  0.01485474 -0.0101526  -0.03373506  0.9569919
  0.65510666], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.38016194  0.00104693  0.015121   -0.02091405 -0.03173233  0.95597
  0.35027087], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.37571532  0.00310263  0.01220647 -0.02205616 -0.0219237   0.9570732
  0.15414065], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.3732069   0.00284098  0.02293168 -0.04824463 -0.03015185  0.9558962
  0.0815745 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.36921954  0.01200177  0.03071779 -0.07897902 -0.00970274  0.99007064
  0.0714388 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.36710778  0.01936721  0.05679273 -0.16383518 -0.01503566  1.0062165
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.3538595   0.0332942   0.07845419 -0.21034881 -0.03057368  1.0257424
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.33374137  0.04099501  0.09001112 -0.20775773 -0.06234048  1.0066078
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2998944   0.05563567  0.08942233 -0.1789732  -0.07874786  1.0160948
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.27728555  0.06084988  0.08182407 -0.16033338 -0.07178819  1.0011059
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2506308   0.06661142  0.0626005  -0.11104646 -0.05623956  0.9646525
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.23295224  0.08099333  0.04485331 -0.06118909 -0.05019638  0.9914577
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22253737  0.09005947  0.02315    -0.03227186 -0.04303008  0.97353125
  0.07112513], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2244058   0.087626    0.01592931 -0.04946928 -0.05404015  0.9522454
  0.1008246 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22292508  0.0841941   0.01347664 -0.03363365 -0.05316353  0.9400495
  0.27666032], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22322309  0.0842143   0.01373289 -0.03652061 -0.05125298  0.9512835
  0.5570889 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2236137   0.08452257  0.01459748 -0.03598881 -0.05159936  0.9562339
  0.8408564 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.22362088  0.08454861  0.02020695 -0.0305167  -0.06460879  0.9470927
  1.0080036 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.2228096   0.07947867  0.04825119  0.01748569 -0.11165888  0.9055245
  1.0041072 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.21848957  0.07613246  0.07808639 -0.02628601 -0.12426008  0.90235144
  1.0008161 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.21668243  0.07715948  0.09221683 -0.09002657 -0.12491859  0.90423095
  1.0008161 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.21691391  0.07661936  0.09257805 -0.09318552 -0.12493232  0.89766574
  1.0008161 ], shape=(7,), dtype=float32)
tf.Tensor(
[ 0.21657135  0.07686681  0.09254072 -0.09258138 -0.12347458  0.90061545
  1.0008161 ], shape=(7,), dtype=float32)
"""