# coding=utf-8
import socket  # socket模块
import json
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from math import floor, sqrt
import tensorflow as tf
import subprocess
import time
import psutil
import pyautogui
from pynput.mouse import Listener
import os

from multiprocessing import Pool
from tensorflow.python.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


class Actor:

    def __init__(self, state_height, state_width, action_size):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.gamma = 0.90  # discount rate
        self.lamda = 0.80  # eligibility trace decay
        self.epsilon = 0.3  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9  # init with pure exploration
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.i = 1
        self.z = [np.zeros(arr.shape) for arr in self.model.trainable_variables]
        # print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG {}".format(self.z))
        self.act_prob = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input1 = Input(shape=(1, self.state_height, self.state_width))
        conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
                       input_shape=(1, self.state_height, self.state_width))(input1)
        conv2 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        conv3 = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        state1 = Flatten()(conv3)
        input2 = Input(shape=(3,))
        state2 = concatenate([input2, state1])
        state2 = Dense(256, activation='relu')(state2)
        state2 = Dense(64, activation='relu')(state2)
        out_put = Dense(self.action_size, activation='softmax')(state2)
        model = Model(inputs=[input1, input2], outputs=out_put)
        self.optimizer = Adam(lr=self.learning_rate)
        return model

    def act(self, state):
        act_prob = self.model(state)
        # print("############################Policy#################")
        # print(act_prob)
        act_value = np.random.choice(self.action_size, p=np.squeeze(act_prob))
        
        self.act_prob = tf.math.log(act_prob[0,act_value])
        
        return act_value  # returns action
    

    def reset(self):
        self.i = 1
        self.z = [np.zeros(arr.shape) for arr in self.model.trainable_variables]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, advantage, tape):
        log_grad = tape.gradient(self.act_prob, self.model.trainable_variables)
        # print("log_grad shape : {},".format(log_grad))
        # self.z = self.gamma * self.lamda * self.z + self.i * log_grad
        updated_grad = self.z
        for l in range(len(self.z)):
            self.z[l] = self.gamma * self.lamda * self.z[l] + self.i * log_grad[l] 
            updated_grad[l] = self.z[l] * advantage[0][0]
        # print ("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUuuuu {}".format(updated_grad))
        # print("z shape : {}".format(self.z.shape))
        # print("z is {}".format(self.z))
        # print("advantage is {}".format(advantage))
        # updated_grad = self.z * advantage
        # print("############################log Grad#################")
        # print(self.act_prob)
        self.optimizer.apply_gradients(zip(updated_grad, self.model.trainable_variables))
        self.i = self.gamma * self.i


class Critic:

    def __init__(self, state_height, state_width, action_size):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.gamma = 0.90  # discount rate
        self.lamda = 0.80  # eligibility trace decay
        self.epsilon = 0.3  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9  # init with pure exploration
        self.learning_rate = 0.00025
        self.i = 1
        self.model = self._build_model()
        self.z = [np.zeros(arr.shape) for arr in self.model.trainable_variables]

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input1 = Input(shape=(1, self.state_height, self.state_width))
        conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
                       input_shape=(1, self.state_height, self.state_width))(input1)
        conv2 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid')(conv1)
        conv3 = Conv2D(3, 1, strides=1, activation='relu', padding='valid')(conv2)
        state1 = Flatten()(conv3)
        input2 = Input(shape=(3,))
        state2 = concatenate([input2, state1])
        state2 = Dense(256, activation='relu')(state2)
        state2 = Dense(64, activation='relu')(state2)
        out_put = Dense(1)(state2)
        model = Model(inputs=[input1, input2], outputs=out_put)
        self.optimizer = Adam(lr=self.learning_rate)
        return model

    def reset(self):
        self.i = 1
        self.z = [np.zeros(arr.shape) for arr in self.model.trainable_variables]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, v_state, advantage, tape):
        # print("############################ADVANTAGE#################")
        # print(advantage)
        # print("############################V_STATE#################")
        # print(v_state)
        grad = tape.gradient(v_state, self.model.trainable_variables)
        # self.z = self.gamma * self.lamda * self.z + grad
        updated_grad = self.z
        for l in range(len(self.z)):
            self.z[l] = self.gamma * self.lamda * self.z[l] + self.i*grad[l] 
            updated_grad[l] = self.z[l] * advantage[0][0]
        self.optimizer.apply_gradients(zip(updated_grad, self.model.trainable_variables))
        self.i=self.i*self.gamma


def connect(ser):
    conn, addr = ser.accept()  # 接受TCP连接，并返回新的套接字与IP地址
    print('Connected by', addr)  # 输出客户端的IP地址
    return conn


def open_ter(loc):
    os.system("gnome-terminal -e 'bash -c \"cd " + loc + " && ./path_planning; exec bash\"'")
    time.sleep(1)
    # return sim


def kill_terminal():
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == "gnome-terminal-server":
            os.kill(pid, 9)


def close_all(sim):
    if sim.poll() is None:
        sim.terminate()
        sim.wait()
    time.sleep(2)
    kill_terminal()


def _on_click_(x, y, button, pressed):
    return pressed


class ActorCriticAgent:
    def __init__(self):
        self.EPISODES = 100
        self.location = "/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/build"

        self.HOST = '127.0.0.1'
        self.PORT = 1234
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
        self.server.bind((self.HOST, self.PORT))  # 套接字绑定的IP与端口
        self.server.listen(1)  # 开始TCP监听

        self.state_height = 45
        self.state_width = 3
        self.action_size = 3
        self.actor = Actor(self.state_height, self.state_width, self.action_size)
        self.critic = Critic(self.state_height, self.state_width, self.action_size)
        # self.actor.load("/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/data/actor/episode48_actor.h5")
        # self.critic.load("/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/data/critic/episode48_critic.h5")
        self.episode =1
        self.gamma = 0.9
        self.num_lane_changes_list = []
        self.avg_velocity_list = []
        self.num_collisions_list = []
        # print("fine so far initialized agent")
        self.main_loop()

    def main_loop(self):
        # print("In main loop")
        while self.episode <= self.EPISODES:
            # 开启程序
            pool = Pool(processes=2)
            result = []
            result.append(pool.apply_async(connect, (self.server,)))
            pool.apply_async(open_ter, (self.location,))
            pool.close()
            pool.join()
            conn = result[0].get()
            sim = subprocess.Popen(
                '/home/prajwal/Autonomous-Driving/decision-making-CarND/term3_sim_linux/term3_sim.x86_64')
            while not Listener(on_click=_on_click_):
                pass

            while not Listener(on_click=_on_click_):
                pass

            time.sleep(2)
            pyautogui.click(x=1913, y=1426, button='left')
            time.sleep(6)
            pyautogui.click(x=1708, y=1711, button='left')
            try:
                data = conn.recv(2000)  # 把接收的数据实例化
            except Exception as e:
                close_all(sim)
                continue
            while not data:
                try:
                    data = conn.recv(2000)
                except Exception as e:
                    close_all(sim)
                    continue
            data = bytes.decode(data)
            # print(data)
            j = json.loads(data)

            # 初始化状态信息
            # Main car's localization Data
            # car_x = j[1]['x']
            # car_y = j[1]['y']
            car_s = j[1]['s']
            car_d = j[1]['d']
            car_yaw = j[1]['yaw']
            car_speed = j[1]['speed']
            # Sensor Fusion Data, a list of all other cars on the same side of the road.
            sensor_fusion = j[1]['sensor_fusion']
            grid = np.ones((51, 3))
            ego_car_lane = int(floor(car_d / 4))
            grid[31:35, ego_car_lane] = car_speed / 100.0

            # sensor_fusion_array = np.array(sensor_fusion)
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

            state = np.zeros((self.state_height, self.state_width))
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
            # print(state)
            action = 0
            mess_out = str(action)
            mess_out = str.encode(mess_out)
            conn.sendall(mess_out)

            # 开始训练过程
            flag = False
            self.actor.reset()
            self.critic.reset()
            self.num_lane_changes = 0
            self.avg_velocity = 0
            self.num_collisions = 0
            i = 0
            while True:
                # print("pass")
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
                if data == "over":  # 此次迭代结束
                    print("Num lane changes : {}".format(self.num_lane_changes))
                    print("Average Velocity : {}".format(self.avg_velocity))
                    print("Num Collisions : {}".format(self.num_collisions))
                    self.num_lane_changes_list.append(self.num_lane_changes)
                    self.avg_velocity_list.append(self.avg_velocity)
                    self.num_collisions_list.append(self.num_collisions)
                    with open('/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/ac.txt', 'a') as f:
                        f.write(" episode {} num_lane_changes {}\n".format(self.episode, self.num_lane_changes))
                        f.write(" episode {} avg_velocity {}\n".format(self.episode, self.avg_velocity))
                        f.write(" episode {} num_collisions {}\n".format(self.episode, self.num_collisions))
                    self.actor.save(
                        "/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/data/actor/episode" + str(
                            self.episode) + "_actor.h5")
                    self.critic.save(
                        "/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/data/critic/episode" + str(
                            self.episode) + "_critic.h5")
                    # print("weight saved")
                    # print("episode: {}, epsilon: {}".format(episode, agent.epsilon))
                    close_all(sim)
                    conn.close()  # 关闭连接
                    self.episode = self.episode + 1
                    break
                try:
                    j = json.loads(data)
                except Exception as e:
                    close_all(sim)
                    break
                # print(j)
                # *****************在此处编写程序*****************
                last_state = state
                # print(last_state)
                last_pos = pos
                last_act = action
                # print(last_act)
                last_lane = ego_car_lane
                # **********************************************

                # Main car's localization Data
                # car_x = j[1]['x']
                # car_y = j[1]['y']
                car_s = j[1]['s']
                car_d = j[1]['d']
                car_yaw = j[1]['yaw']
                car_speed = j[1]['speed']

                # print(car_s)
                if car_speed == 0:
                    mess_out = str(0)
                    mess_out = str.encode(mess_out)
                    conn.sendall(mess_out)
                    continue
                # Sensor Fusion Data, a list of all other cars on the same side of the road.
                sensor_fusion = j[1]['sensor_fusion']
                ego_car_lane = int(floor(car_d / 4))
                if last_act == 0:
                    last_reward = (2 * ((j[3] - 25.0) / 5.0))  # - abs(ego_car_lane - 1))
                else:
                    last_reward = (2 * ((j[3] - 25.0) / 5.0)) - 10.0
                if grid[3:31, last_lane].sum() > 27 and last_act != 0:
                    last_reward = -30.0

                grid = np.ones((51, 3))
                grid[31:35, ego_car_lane] = car_speed / 100.0
                # sensor_fusion_array = np.array(sensor_fusion)
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
                    if j[2] < -10:
                        last_reward = float(j[2])  # reward -50, -100
                        self.num_collisions += 1
                print("J[2] is : {}".format(j[2]))
                last_reward = last_reward / 10.0
                state = np.zeros((self.state_height, self.state_width))
                state[:, :] = grid[3:48, :]
                state = np.reshape(state, [-1, 1, self.state_height, self.state_width])
                # print(state)
                pos = [car_speed / 50, 0, 0]
                if ego_car_lane == 0:
                    pos = [car_speed / 50, 0, 1]
                elif ego_car_lane == 1:
                    pos = [car_speed / 50, 1, 1]
                elif ego_car_lane == 2:
                    pos = [car_speed / 50, 1, 0]
                pos = np.reshape(pos, [1, 3])
                print("last_action:{}, last_reward:{:.4}, speed:{:.3}".format(last_act, last_reward,
                                                                              float(car_speed)))

                if flag:
                    with tf.GradientTape() as criticTape:
                        v_s = self.critic.model([last_state, last_pos])
                    advantage = last_reward + self.gamma * self.critic.model.predict(
                        [state, pos]) - self.critic.model.predict([last_state, last_pos])
                    self.actor.train(advantage, actorTape)
                    self.critic.train(v_s, advantage, criticTape)
                with tf.GradientTape() as actorTape:
                    action = self.actor.act([state, pos])
                print("Took action ", action)
                if action != 0:
                    self.num_lane_changes += 1
                flag = True
                # **********************************************
                i += 1
                self.avg_velocity = (self.avg_velocity * (i-1) + car_speed) / i
                mess_out = str(action)
                mess_out = str.encode(mess_out)
                conn.sendall(mess_out)
        with open('/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/src/train/ac_list.txt', 'a') as f:
            f.write(" num_lane_changes_list {}\n".format(self.num_lane_changes_list))
            f.write(" avg_velocity_list {}\n".format(self.avg_velocity_list))
            f.write(" num_collisions_list {}\n".format(self.num_collisions_list))


print("calling agent")
ActorCriticAgent()
