import gym
from stable_baselines3 import PPO
import random

import argparse
import yaml

import csv
import numpy as np
import os
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True, help='Path to the config yaml')
    return ap.parse_args()

if __name__ == '__main__':
  args = parse_args()

  # load config
  config = yaml.full_load(open(args.config_path, 'r'))
  n_cpu = 6

  env = gym.make(config['env_name'])

  env.configure({
    "controlled_vehicles": random.choice(range(1,39))+1,
    "vehicles_count": random.choice(range(10))+1, # add more IDM vehicles if needs
    "observation": {
      "type": "MultiAgentObservation",
      "observation_config": {
        "type": "Kinematics",
      }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
        "type": "DiscreteMetaAction",
        },
    },
    "create_vehicles_ego_idm": True, # True to create ego vehicle as IDM
    "ego_collision_weight": config['ego_collision_weight'],
  })
  env.reset()

  model = PPO('MultiInputPolicy', env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                learning_rate=5e-4,
                batch_size=32,
                gamma=0.8,
                n_steps=32 * 12 // n_cpu,
                verbose=2,
                n_epochs=100,
                tensorboard_log="highway_multiagent/")

  env.set_dqn_model(model)
  model.learn(int(2e4))
  model.save(config['model_path']) 

  print('start inference')
  # inference
  ttc_threshold = config['ttc_threshold']
  save_result_csv = config['save_result_csv']
  # number of episodes
  T = config['num_episodes']
  num_steps = config['num_steps']
  eps = 0
  patience = 3
  episode_reward = 0
  rewards = []
  while eps < T:
    age = 0
    done = False
    obs = env.reset()
    previous_info = False
    prev_info = None
    for i in range(num_steps):# done:
      action = model.predict(obs)
      
      # convert action to tuple
      action = tuple(action[0])
      next_obs, reward, done, info = env.step(action)
      episode_reward += reward
      rewards.append(episode_reward)
      current_crashed = info['crashed']


      if current_crashed and not previous_info:
        if prev_info is None:
          prev_info = info
        
        ego_speed = info['speed']
        ego_acceleration = info['ego_action']['acceleration']
        ego_ast_action = info['ego_ast_action']
        lane_index_ = info['current_lane_index'][2]

        # get crashed vehicle info
        crashed_veh_info = info['crashed_veh_info']
        crashed_speed = crashed_veh_info['crashed_veh_speed']
        crashed_lane_index = crashed_veh_info['crashed_veh_lane_index'][2]
        crashed_front = crashed_veh_info['front']
        crashed_distance = crashed_veh_info['crashed_distance']
        crashed_acceleration = crashed_veh_info['action']['acceleration']
        crashed_ast_action = crashed_veh_info['ast_action']
        with open(save_result_csv, 'a', newline='') as file:
          writer = csv.writer(file)
          # append to csv file the done, reward
          writer.writerow([
            eps, i, ego_speed, ego_acceleration, ego_ast_action, lane_index_,
            crashed_speed, crashed_acceleration, crashed_lane_index, crashed_ast_action, crashed_front, crashed_distance,
            ])
        # save picture
        os.makedirs(config['save_pic'], exist_ok=True)
        img_file = os.path.join(config['save_pic'], 'episode_{}_step_{}.png'.format(eps, i))
        img = env.render(mode='rgb_array')
        plt.imsave(img_file, img)

        previous_info = True
      if not current_crashed:
        previous_info = False

      prev_obs = next_obs
      prev_reward = reward
      prev_info = info
      prev_done = done
      obs = next_obs
      env.render()
      if done:
        episode_reward = 0.0
        obs = env.reset()
      
    eps = eps + 1

  # save rewards
  with open(config['reward_file'], 'wb') as f:
    np.save(f, rewards)