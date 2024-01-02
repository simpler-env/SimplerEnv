import mani_skill2.envs, gymnasium as gym

def build_maniskill2_env(env_name, instruction=None, **kwargs):
    env = gym.make(env_name, **kwargs)
    
    # Get task description
    obj_name = ' '.join(env.obj.name.split('_')[1:])
    if instruction is not None:
        task_description = instruction
    elif env_name == 'PickSingleYCBIntoBowl-v0':
        task_description = f"place {obj_name} into red bowl"
    elif env_name in ['GraspSingleYCBInScene-v0', 'GraspSingleYCBSomeInScene-v0']:
        task_description = f"pick {obj_name}"
    elif env_name in ['GraspSingleYCBFruitInScene-v0']:
        task_description = "pick fruit"
    elif env_name in ['GraspSingleYCBCanInScene-v0', 'GraspSingleYCBTomatoCanInScene-v0']:
        task_description = "pick can"
    elif 'CokeCan' in env_name:
        task_description = "pick coke can"
    elif env_name in ['GraspSinglePepsiCanInScene-v0', 'GraspSingleUpRightPepsiCanInScene-v0']:
        task_description = "pick pepsi can"
    elif env_name == 'GraspSingleYCBBoxInScene-v0':
        task_description = "pick box"
    elif env_name == 'KnockSingleYCBBoxOverInScene-v0':
        task_description = "knock box over"
    else:
        raise NotImplementedError()
    
    print(task_description)
    
    return env, task_description