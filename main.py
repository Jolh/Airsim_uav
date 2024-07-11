import argparse
from xuance.common import get_configs
from airsim_env import AirSimDroneEnv
from xuance.environment import make_envs, REGISTRY_ENV
from xuance.torch.agents import PPOCLIP_Agent

configs_dict = get_configs(file_dir="airsim.yaml")
configs = argparse.Namespace(**configs_dict)

REGISTRY_ENV['AirSimDroneEnv'] = AirSimDroneEnv
envs = make_envs(configs)  # Make parallel environments.
Agent = PPOCLIP_Agent(config=configs, envs=envs)  # Create a PPO agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.