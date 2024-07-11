import airsim
import gym
import numpy as np
from gym.spaces import Box
from xuance.environment import RawMultiAgentEnv


class AirSimDroneEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(AirSimDroneEnv, self).__init__()
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.drone = airsim.MultirotorClient(ip=env_config.ip_address)
        self.setup_flight()

        self.target_pos = np.array([-19, -14, -4])
        x, y, z = self.drone.simGetVehiclePose().position
        self.target_dist_prev = np.linalg.norm(np.array([x, y, z]) - self.target_pos)
        self.env_id = env_config.env_id
        self.observation_space = Box(low=0, high=255, shape=[84, 84, 3], dtype=np.uint8, seed=env_config.seed)
        self.action_space = gym.spaces.Discrete(9)
        self.max_episode_steps = 2000
        self._episode_step = 0

        self.info = {"collision": False}
        self.collision_time = 0

        self.space_range_x = [-10.0, 10.0]
        self.space_range_y = [-10.0, 10.0]
        self.space_range_z = [0.1, 10.0]

    def reset(self):
        info = {}
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs, info

    def render(self):
        return self.get_obs()

    def close(self):
        return

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        self._episode_step += 1
        truncated = False if self._episode_step < self.max_episode_steps else True
        return obs, reward, done, truncated, info

    def get_obs(self):
        obs = self.get_rgb_image()
        return obs, {}

    def do_action(self, select_action):
        speed = 0.4
        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)

        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=1).join()

        # # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def compute_reward(self):
        reward = 0
        done = 0

        # Target distance based reward
        x, y, z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([x, y, z]) - self.target_pos)
        reward += (self.target_dist_prev - target_dist_curr) * 20
        self.target_dist_prev = target_dist_curr
        # Alignment reward
        if target_dist_curr < 0.3:
            reward += 12
            done = 1
        if z < self.space_range_z[0] + 0.05:
            reward -= 10
        if (x < self.space_range_x[0]) or (x > self.space_range_x[1]) or (y < self.space_range_y[0]) or (
                y > self.space_range_y[1]) or (z < self.space_range_z[0]) or (
                z > self.space_range_z[1]):  # out of range
            done = 1

        return reward, done

    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(2, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

        # Sometimes no image returns from api
        try:
            return img2d.reshape([84, 84, 3])
        except:
            return np.zeros([84, 84, 3])
