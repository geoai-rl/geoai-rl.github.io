#-*- coding:utf-8 -*-

"""
- date : 2021.1.26
- author : Jiwon Jang

- Train agent to find the shortest path in the stochastic environment using deep Q-learning.

[reference]
- Lee et. al. 2020. Learning reinforcement learning with Python and Keras. WikiBooks
- http://www.yes24.com/Product/Goods/44136413
- https://github.com/rlcode/reinforcement-learning-kr

[Dependency]
- OS: MAC OS Big SUR
- python: 3.7.4
- Keras: 2.3.1
- numpy: 1.17.2

"""

import time
import numpy as np
import tkinter as tk # tk gui toolkit
from PIL import ImageTk, Image
from tqdm import tqdm

import sys
import copy
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

PhotoImage = ImageTk.PhotoImage
UNIT = 50  # number of pixel
HEIGHT = 20
WIDTH = 20

np.random.seed(1)

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r'] # define action
        self.action_size = len(self.action_space)
        self.title('DQN - Consumer Satisfaction model')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        # negative reward setting - obstacles (triangle)
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)

        self.set_reward([0, 10], -1)
        self.set_reward([1, 11], -1)
        self.set_reward([2, 12], -1)

        # positive reward setting - circle
        self.set_reward([2, 0], 1)
        self.set_reward([2, 1], 1)
        self.set_reward([3, 2], 1)
        self.set_reward([4, 2], 1)
        self.set_reward([4, 4], 1)

        self.set_reward([7, 7], 5)
        self.set_reward([8, 8], 5)
        self.set_reward([9, 9], 5)

        self.set_reward([17, 17], 10)
        self.set_reward([18, 18], 10)
        self.set_reward([19, 19], 10)


    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # generate grid world
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = []
        self.goal = []
        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0])

        canvas.pack()

        return canvas

    def load_images(self):
        filedir = '/Users/jiwonjang/python_tutorial/ve_env/masterThesis/reinforcement_learning/reinforcement-learning-kr-master/1-grid-world/img/'
        rectangle = PhotoImage(Image.open(filedir + "rectangle.png").resize((30, 30))) # agent
        triangle = PhotoImage(Image.open(filedir + "triangle.png").resize((30, 30))) # negative reward
        circle = PhotoImage(Image.open(filedir + "circle.png").resize((30, 30))) # positive reward

        return rectangle, triangle, circle

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward > 0: # positive reward
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2])

            self.goal.append(temp['figure'])

        elif reward < 0: # negative reward
            temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state

        self.rewards.append(temp)

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()

        # negative reward setting
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)

        self.set_reward([0, 10], -1)
        self.set_reward([1, 11], -1)
        self.set_reward([2, 12], -1)

        # positive reward setting
        self.set_reward([2, 0], 1)
        self.set_reward([2, 1], 1)
        self.set_reward([3, 2], 1)
        self.set_reward([4, 2], 1)
        self.set_reward([4, 4], 1)

        self.set_reward([7, 7], 5)
        self.set_reward([8, 8], 5)
        self.set_reward([9, 9], 5)

        self.set_reward([17, 17], 10)
        self.set_reward([18, 18], 10)
        self.set_reward([19, 19], 10)

    def reset(self):
        # reset the Environment

        self.update() # update method가 가지는 기능을 모르겠음 - tk 클래스에 할당되어 있는 메소드임
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        print('reset')
        print(x, y)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()

        return self.get_state()

    def get_state(self):

        location = self.coords_to_state(self.canvas.coords(self.rectangle))

        agent_x = location[0]
        agent_y = location[1]

        states = list()

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y)

            if reward['reward'] < 0:
                states.append(-1)
                states.append(reward['direction'])

            else:
                states.append(1)

        return states

    def move_const(self, target):

        s = self.canvas.coords(target['figure'])
        base_action = np.array([0, 0])

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:
            target['direction'] = 1
        elif s[0] == UNIT / 2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT

        if (target['figure'] is not self.rectangle
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)

        return [x, y]

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] >= 1:
                new_rewards.append(temp)
                continue

            temp['coords'] = self.move_const(temp)
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)

        return new_rewards

    def move(self, target, action):
        s = self.canvas.coords(target)
        base_action = np.array([0, 0])

        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(target, base_action[0], base_action[1])
        s_ = self.canvas.coords(target)

        return s_

    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']
                if reward['reward'] == 1:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list

    def step(self, action):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action)
        check = self.check_if_reward(self.coords_to_state(next_coords))

        done = check['if_goal']
        reward = check['rewards']

        self.canvas.tag_raise(self.rectangle)

        s_ = self.get_state()

        return s_, reward, done

    def render(self):
        # 게임 속도 조정
        time.sleep(0.05)
        self.update()

"""
Define deep Q-learning Model (DQN)
"""

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        # DQN hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.005
        self.batch_size = 64
        self.train_start = 1000

        # reply memory
        self.memory = deque(maxlen=2000)

        # generate deep-learning network
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if self.load_model:
            self.model.load_weights('./save_model/DQN_0401.h5')


    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save interaction history
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # batch learning
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


"""
Start Simulation
"""

EPISODES = 4000 # number of episodes

if __name__ == "__main__":
    action_space = [0, 1, 2, 3, 4] # define action
    action_size = len(action_space)
    state_size = 57

    env = Env()
    agent = DQNAgent(state_size, action_size)

    global_step = 0
    scores, episodes = [], []

    for e in tqdm(range(EPISODES)):
        print('episode:', e)
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        for i in range(200):
            global_step += 1
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)
            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state
            score += reward

            if i==199:
                agent.update_target_model()
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/DQN_.png")
                print("episode:", e, "  score:", score, "global_step", global_step,
                    " memory length:", len(agent.memory), "  epsilon:", agent.epsilon)


        if e % 100 == 0:
            agent.model.save_weights("./save_model/DQN_.h5")
