from math import floor, sqrt
import json
from sim_utils import *
from actor_critic_agent import ActorCriticAgent
import numpy as np
import tensorflow as tf

class Environment:
    def __init__(self):
        self.state_height = 45
        self.state_width = 3
        self.episodes = 100
        self.episode = 1

    def set_ego_vehicle_state(self, grid, decoded_data):
        # Ego Vehicle localization Data
        car_d = decoded_data[1]['d']
        car_speed = decoded_data[1]['speed']
        ego_car_lane = int(floor(car_d/4))
        grid[31:35, ego_car_lane] = car_speed / 100.0
        return grid
    

    def set_surrounding_vehicle_state(self, grid, decoded_data):
        # Sensor Fusion Data, a list of all other cars on the same side of the road.
        sensor_fusion = decoded_data[1]['sensor_fusion']
        car_s = decoded_data[1]['s']
        for i in range(len(sensor_fusion)):
            vx = sensor_fusion[i][3]
            vy = sensor_fusion[i][4]
            s = sensor_fusion[i][5]
            d = sensor_fusion[i][6]
            check_speed = sqrt(vx * vx + vy * vy)
            car_lane = int(floor(d / 4))
            if 0 <= car_lane < 3:
                s_dis = s - car_s
                if -36 < s_dis < 66:
                    pers = - int(floor(s_dis / 2.0)) + 30
                    grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237
        return grid


    def get_input_to_agent(self, grid, decoded_data):
        state = np.zeros((self.state_height, self.state_width))
        car_speed = decoded_data[1]['speed']
        car_d = decoded_data[1]['d']
        ego_car_lane = int(floor(car_d/4))
        state[:, :] = grid[3:48, :]
        state = np.reshape(state, [-1, 1, self.state_height, self.state_width])
        pos = [car_speed / 50, 0, 0]
        if ego_car_lane == 0:
            pos = [car_speed / 50, 0, 1]
        elif ego_car_lane == 1:
            pos = [car_speed / 50, 1, 1]
        elif ego_car_lane == 2:
            pos = [car_speed / 50, 1, 0]
        pos = np.reshape(pos, [1, 3])
        return state, pos
            
    def get_reward(self, act, decoded_data, grid, lane):
        if act == 0:
            reward = (2 * ((decoded_data[3] - 25.0) / 5.0))  # - abs(ego_car_lane - 1))
        else:
            reward = (2 * ((decoded_data[3] - 25.0) / 5.0)) - 10.0

        if grid[3:31, lane].sum() > 27 and act != 0:
            reward = -30.0
        if decoded_data[2] < -10:
                reward = float(decoded_data[2])  # check for collision

        reward = reward / 10.0
        return reward


    def run(self):
        server = setup_server()
        agent = ActorCriticAgent(self.state_height, self.state_width)
        while self.episode <= self.episodes:    
            conn, sim = launch_simulator(server)
            try:
                data = conn.recv(2000)
            except Exception as e:
                close_all(sim)
                continue
            data = bytes.decode(data)
            j = json.loads(data)
            grid = np.ones((self.state_height + 6, self.state_width))
            grid = self.set_ego_vehicle_state(grid, j)
            grid = self.set_surrounding_vehicle_state(grid, j)
            state, pos = self.get_input_to_agent(grid, j)
            action = 0
            mess_out = str(action)
            mess_out = str.encode(mess_out)
            conn.sendall(mess_out)
            flag = False
            agent.reinitialize()
            while True:
                try:
                    data = conn.recv(2000)
                except Exception as e:
                    pass
                while not data:
                    try:
                        data = conn.recv(2000)
                    except Exception as e:
                        pass
                data = bytes.decode(data)
                if data == "over":
                    agent.save(self.episode)
                    print("weight saved")
                    print("episode: {}".format(self.episode))
                    with open('/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/train.txt', 'a') as f:
                        f.write(" episode {}\n".format(self.episode))
                    close_all(sim)
                    conn.close()
                    self.episode += 1
                    break
                try:
                    j = json.loads(data)
                except Exception as e:
                    close_all(sim)
                    break
                car_d = j[1]['d']
                ego_car_lane = int(floor(car_d/4))
            
                last_state = state
                last_pos = pos
                last_act = action
                last_lane = ego_car_lane

                car_speed = j[1]['speed']
                if car_speed == 0:
                    mess_out = str(0)
                    mess_out = str.encode(mess_out)
                    conn.sendall(mess_out)
                    continue
                last_reward = self.get_reward(last_act, j, grid, last_lane)
                grid = np.ones((51, 3))
                grid = self.set_ego_vehicle_state(grid, j)
                grid = self.set_surrounding_vehicle_state(grid, j)
                state, pos = self.get_input_to_agent(grid, j)
                print("episode: {}, last_action:{}, last_reward:{:.4}, speed:{:.3}"
                                                .format(self.episode, last_act, last_reward, float(car_speed)))

                
                if flag:
                    agent.update([last_state, last_pos], [state, pos], last_reward)
                flag = True

                action = agent.actor.act([state, pos])                

                print("Took action ", action)
                
                mess_out = str(action)
                mess_out = str.encode(mess_out)
                conn.sendall(mess_out)
