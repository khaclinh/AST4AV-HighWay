import numpy as np
from gym.envs.registration import register
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
import math
from stable_baselines3.common.policies import obs_as_tensor

import random
from highway_env.envs.common.graphics import EnvViewer
import torch

Observation = np.ndarray

class HighwayEnvEgoIBDMMOBILReward(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    print('HighwayEnvEgoIBDMMOBILReward')
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
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
            "lanes_count": 4,
            "vehicles_count": 0,
            "controlled_vehicles": 40,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 2,
            "collision_reward": 0,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [30, 40],
            "offroad_terminal": False,
            "ttc_threash_hold": 1.5,
            "create_vehicles_ego_idm": True
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        if self.config["create_vehicles_ego_idm"]:
            self._create_vehicles_ego_idm()
            #print("create_vehicles_ego_idm")
        else:
            self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=30,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _create_vehicles_ego_idm(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        #print("other_per_controlled:" + str(other_per_controlled))
        
        self.controlled_vehicles = []
        #rendom step to create ego vehicle
        rnd=random.choice(range(self.config["controlled_vehicles"]))
        for i in range(self.config["controlled_vehicles"]):
            #create controlled vehicles
            vehicle = Vehicle.create_random(
                self.road,
                speed=30,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            # Create ego vehicle as HDM in the step selected rendomly. 
            # Ego vehicle will we created between controlled vehicles rendomly.
            if i==rnd:
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                self.ego_veh = self.road.vehicles[-1]
                self.ego_veh.color = (255, 255, 0) 

        #create other HDM vehicles if needed
        for _ in range(self.config["vehicles_count"]):
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
            vehicle.randomize_behavior()
            vehicle.color=(100,100,100)
            self.road.vehicles.append(vehicle)
        #print(self.road.vehicles)

    def set_dqn_model(self, model):
        self.model = model

    def cal_ttc(self, a_f, v_f, a_l, v_l, d):
        """
            This function is aim to calculate the time to collision between two vehicles.
            a_f: acceleration of following vehicle
            v_f: velocity of following vehicle
            a_l: acceleration of leading vehicle
            v_l: velocity of leading vehicle
            d: distance between two vehicles
            return: ttc: time to collision
            ttc can be determined by the following equation:
            v_f*t + 1/2*a_f*t^2 = d + v_l*t + 1/2*a_l*t^2
        """
        a = 1/2*(a_f-a_l)
        b = v_f - v_l
        c = d
        delta = b**2 - 4*a*c
        if delta < 0:
            return 100
        else:
            t1 = (-b + np.sqrt(delta))/(2*a)
            t2 = (-b - np.sqrt(delta))/(2*a)
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
            elif t1 > 0:
                return t1
            elif t2 > 0:
                return t2
            else:
                return 100
        

    def time_to_collision(self):
        """
        This function is aim to calculate the time to collision between ego vehicle and other vehicles.
        """
        front_ttc, rare_ttc, front_ttc_r, rare_ttc_r, front_ttc_l, rare_ttc_l = 100, 100, 100, 100, 100, 100
        # v = self.vehicle
        v = self.ego_veh
        ego_lane_index=v.lane_index[2]
        #print("ego_lane_index:" + str(ego_lane_index))
        side_lane = self.road.network.side_lanes(v.lane_index)
        #print("side_lane:" + str(side_lane))
        side_lane.append(v.lane_index)
        #print("side_lane:" + str(side_lane))

        # ego speed
        ego_speed = v.speed
        # ego acceleration
        ego_acceleration = v.action['acceleration']

        for lane in side_lane:

            f, r, n_front, n_rear = self.road.neighbour_vehicles_ast(self.ego_veh, lane)
            # increase number of front and rear vehicles for each lane
            
            front_speed = rear_speed = 0
            front_acceleration = rear_acceleration = 0
            if f != None:
                # get front vehicle speed
                front_speed = f.speed
                # get front vehicle acceleration
                front_acceleration = f.action['acceleration']
            if r != None:
                # get rear vehicle speed
                rear_speed = r.speed
                # get rear vehicle acceleration
                rear_acceleration = r.action['acceleration']
            

            if f == None:
                continue
            if f.lane_index[2]==ego_lane_index and f!=None:
                front_spasing=v.lane_distance_to(f)
                # compute the time to collision between ego vehicle and front vehicle
                front_ttc = self.cal_ttc(ego_acceleration, ego_speed, front_acceleration, front_speed, front_spasing)
               

            if f.lane_index[2]==ego_lane_index and r!=None:
                rare_spasing=v.lane_distance_to(r)
                # compute the time to collision between ego vehicle and rear vehicle
                rare_ttc = self.cal_ttc(rear_acceleration, rear_speed, ego_acceleration, ego_speed, rare_spasing)

            if f.lane_index[2]>ego_lane_index and f!=None:
                front_spasing_r=v.lane_distance_to(f)
                # compute the time to collision between ego vehicle and front vehicle
                front_ttc_r = self.cal_ttc(ego_acceleration, ego_speed, front_acceleration, front_speed, front_spasing_r)

            if f.lane_index[2]>ego_lane_index and r!=None:
                rare_spasing_r=v.lane_distance_to(r)
                # compute the time to collision between ego vehicle and rear vehicle
                rare_ttc_r = self.cal_ttc(rear_acceleration, rear_speed, ego_acceleration, ego_speed, rare_spasing_r)

            if f.lane_index[2]<ego_lane_index and f!=None:
                front_spasing_l=v.lane_distance_to(f)
                # compute the time to collision between ego vehicle and front vehicle
                front_ttc_l = self.cal_ttc(ego_acceleration, ego_speed, front_acceleration, front_speed, front_spasing_l)

                
            if f.lane_index[2]<ego_lane_index and r!=None:
                rare_spasing_l=v.lane_distance_to(r)
                # compute the time to collision between ego vehicle and rear vehicle
                rare_ttc_l = self.cal_ttc(rear_acceleration, rear_speed, ego_acceleration, ego_speed, rare_spasing_l)

        spacing_ret = front_ttc, rare_ttc, front_ttc_r, rare_ttc_r, front_ttc_l, rare_ttc_l
        return spacing_ret
    
    def cal_prob_other(self, veh_ttc, ttc_threshold):
        if veh_ttc <= ttc_threshold:
            prob_col = 0
        else:
            prob_col = (veh_ttc - ttc_threshold) / veh_ttc
        return prob_col

    def prob_collision_of_other(self, other_veh: MDPVehicle, ttc_threshold=1.5):
        """
        This function is aim to calculate the time to collision between ego vehicle and other vehicles.
        """
        front_ttc, rare_ttc, front_ttc_r, rare_ttc_r, front_ttc_l, rare_ttc_l = 100, 100, 100, 100, 100, 100
        lane_index=other_veh.lane_index[2]
        side_lane = self.road.network.side_lanes(other_veh.lane_index)
        side_lane.append(other_veh.lane_index)

        count_veh = 0
        prob_col = 0
        # ego speed
        veh_speed = other_veh.speed
        # ego acceleration
        veh_acceleration = other_veh.action['acceleration']
        
        for lane in side_lane:
            f, r = self.road.neighbour_vehicles(other_veh, lane)
            
            front_speed = rear_speed = 0
            front_acceleration = rear_acceleration = 0
            if f != None:
                # get front vehicle speed
                front_speed = f.speed
                # get front vehicle acceleration
                front_acceleration = f.action['acceleration']
            if r != None:
                # get rear vehicle speed
                rear_speed = r.speed
                # get rear vehicle acceleration
                rear_acceleration = r.action['acceleration']
            

            if f == None:
                continue
            if f.lane_index[2]==lane_index and f!=None:
                front_spasing=other_veh.lane_distance_to(f)
                # compute the time to collision between ego vehicle and front vehicle
                front_ttc = self.cal_ttc(veh_acceleration, veh_speed, front_acceleration, front_speed, front_spasing)
                if f != self.ego_veh:
                    count_veh += 1
                    prob_col += self.cal_prob_other(front_ttc, ttc_threshold)
               

            if f.lane_index[2]==lane_index and r!=None:
                rare_spasing=other_veh.lane_distance_to(r)
                # compute the time to collision between ego vehicle and rear vehicle
                rare_ttc = self.cal_ttc(rear_acceleration, rear_speed, veh_acceleration, veh_speed, rare_spasing)
                if r != self.ego_veh:
                    count_veh += 1
                    prob_col += self.cal_prob_other(front_ttc, ttc_threshold)

            if f.lane_index[2]>lane_index and f!=None:
                front_spasing_r=other_veh.lane_distance_to(f)
                # compute the time to collision between ego vehicle and front vehicle
                front_ttc_r = self.cal_ttc(veh_acceleration, veh_speed, front_acceleration, front_speed, front_spasing_r)
                if f != self.ego_veh:
                    count_veh += 1
                    prob_col += self.cal_prob_other(front_ttc_r, ttc_threshold)

            if f.lane_index[2]>lane_index and r!=None:
                rare_spasing_r=other_veh.lane_distance_to(r)
                # compute the time to collision between ego vehicle and rear vehicle
                rare_ttc_r = self.cal_ttc(rear_acceleration, rear_speed, veh_acceleration, veh_speed, rare_spasing_r)
                if r != self.ego_veh:
                    count_veh += 1
                    prob_col += self.cal_prob_other(rare_ttc_r, ttc_threshold)

            if f.lane_index[2]<lane_index and f!=None:
                front_spasing_l=other_veh.lane_distance_to(f)
                # compute the time to collision between ego vehicle and front vehicle
                front_ttc_l = self.cal_ttc(veh_acceleration, veh_speed, front_acceleration, front_speed, front_spasing_l)
                if f != self.ego_veh:
                    count_veh += 1
                    prob_col += self.cal_prob_other(front_ttc_l, ttc_threshold)

                
            if f.lane_index[2]<lane_index and r!=None:
                rare_spasing_l=other_veh.lane_distance_to(r)
                # compute the time to collision between ego vehicle and rear vehicle
                rare_ttc_l = self.cal_ttc(rear_acceleration, rear_speed, veh_acceleration, veh_speed, rare_spasing_l)
                if r != self.ego_veh:
                    count_veh += 1
                    prob_col += self.cal_prob_other(rare_ttc_l, ttc_threshold)

        
        if prob_col == 0:
            prob_col = -10e-7
        else:
            prob_col = math.log(prob_col / count_veh)
        
        return prob_col

    
    def _reward(self, action: Action) -> float:
        ego_collision_weight = self.config["ego_collision_weight"]
        vehicle_list = self.road.close_vehicles_to(self.ego_veh,
                                                         self.PERCEPTION_DISTANCE,
                                                         count=20,
                                                         )
        # vehicle_list = self.controlled_vehicles
        ttc_threash_hold=self.config["ttc_threash_hold"]
        v = self.ego_veh
        ego_prob_colision=0
        tau = -10000

        # 1. calculate collision probability of ego vehicle
        ttc = self.time_to_collision()
        for i in range(len(ttc)):
            if ttc[i]<ttc_threash_hold:
                tmp_prob = 1 
            else:
                tmp_prob = ttc_threash_hold/ttc[i]
            ego_prob_colision += tmp_prob
        ego_prob_colision = 0 if len(ttc) == 0 else ego_prob_colision/len(ttc)


        if ego_prob_colision==0:
            ego_prob_col= tau
        else:
            ego_prob_col=math.log(ego_prob_colision)

        # 2. calculate collision probability of other vehicles
        other_prob_col = 0
        count_veh = 0
        for agent_action, agent_veh in zip(action, vehicle_list):
            if agent_veh == self.ego_veh:
                continue
            count_veh = count_veh + 1
            # agent_veh.act(agent_action)
            other_prob_col += self.prob_collision_of_other(agent_veh, ttc_threash_hold)

        if count_veh == 0:
            other_prob_col = tau
        else:
            other_prob_col = other_prob_col/count_veh

        # Finally, calculate the reward
        if v.crashed:
            reward = 0
            # self.reset()
        else:
            reward = ego_collision_weight * ego_prob_col + (1-ego_collision_weight) * other_prob_col

        if self.steps > 500 and not v.crashed:
            reward = -(10000+1000*min(ttc)/ttc_threash_hold)
            self.reset()
        
        reward = 0 if not self.ego_veh.on_road else reward
        self.render()
        return reward

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.ego_veh is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        
        obs = self.observation_type.observe()
        self._simulate(action)

        reward = self._reward(action)
        terminal = self._is_terminal()
        
        info = self._info(obs, action)

        return obs, reward, terminal, info
    def _is_no_crash(self) -> bool:
        """The episode is over and the ego vehicle notncrashed and the time is out."""
        # return not self.vehicle.crashed and \
        #     self.time >= self.config["duration"]
        return not self.ego_veh.crashed and \
            self.time >= self.config["duration"] 

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.ego_veh.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.ego_veh.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""

        # return float(self.vehicle.crashed)
        return float(self.ego_veh.crashed)
    
    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """

        # find other vehicle which crashed with ego vehicle
        tem_distance = 1000 # metter
        front = None
        crashed_veh = None
        crashed_veh_info = {
                    "crashed_veh_speed": None,
                    "crashed_veh_lane_index": None,
                    "front": None,
                    "action": None,
                    "ast_action": None
                }
        if self.ego_veh.crashed:
            for veh in self.road.vehicles:
                # check if veh is different from ego vehicle
                if veh != self.ego_veh and veh.crashed:
                    # find the closest vehicle
                    distance = np.linalg.norm(veh.position - self.ego_veh.position)
                    if  distance < tem_distance:
                        tem_distance = np.linalg.norm(veh.position - self.ego_veh.position)
                        crashed_veh = veh
            # get the information of crashed vehicle
            
            if crashed_veh is not None:
                front = 1
                # check crashed_veh is front or back of ego vehicle by longitudinal position
                if crashed_veh.position[0] < self.ego_veh.position[0]:
                    front = 0

                crashed_veh_info = {
                    "crashed_veh_speed": crashed_veh.speed,
                    "crashed_veh_lane_index": crashed_veh.lane_index,
                    "front": front,
                    "action": crashed_veh.action,
                    "ast_action": crashed_veh.ast_action,
                    "crashed_distance": tem_distance
                }
        

        # get ego acceleration
        # ego_acceleration = self.ego_veh.speed - self.ego_veh.last_speed
        info = {
            "speed": self.ego_veh.speed,
            "crashed": self.ego_veh.crashed and self.time < self.config["duration"],
            "current_lane_index": self.ego_veh.lane_index,
            "crashed_veh_info": crashed_veh_info,
            "ego_action": self.ego_veh.action,
            "ego_ast_action": self.ego_veh.ast_action
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass
        return info

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode
        if self.viewer is None:
            self.viewer = EnvViewer(self)
        self.viewer.observer_vehicle = self.ego_veh
        self.enable_auto_render = True
        self.viewer.display()
        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

register(
    id='HighwayEnvEgoIBDMMOBILReward',
    entry_point='highway_env.envs:HighwayEnvEgoIBDMMOBILReward',
)

