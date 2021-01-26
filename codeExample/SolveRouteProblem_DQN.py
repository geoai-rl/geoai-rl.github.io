#-*- coding:utf-8 -*-

"""
- date : 2021.1.26
- author : Jiwon Jang

- solve routing problem using reinforcement learning_rate

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

import copy
import pylab
import random
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

PhotoImage = ImageTk.PhotoImage
UNIT = 50
HEIGHT = 5
WIDTH = 5

np.random.seed(1)

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('DeepSARSA - Victim Model(Static Environment)')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        # negative reward setting
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)

        # positive reward setting
        self.set_reward([2, 0], 1)
        self.set_reward([2, 1], 1)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = []
        self.goal = []

        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0])

        canvas.pack()

        return canvas

    def load_images(self):
        filedir = '/Users/jiwonjang/Dropbox/RL_evacuation/img/'
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

        # positive reward setting
        self.set_reward([2, 0], 1)
        self.set_reward([2, 1], 1)
        self.set_reward([3, 2], 1)

    def reset(self):
        self.update() # update method가 가지는 기능을 모르겠음
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
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
        elif s[0] == UNIT / 2: # UNIT / 2 -> 25.0
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT

        if (target['figure'] is not self.rectangle
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]): # (WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT -> (200, 200)
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
        base_action = np.array([0, 0]) # x, y

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
        time.sleep(0.05)
        self.update()

"""
Define Deep Q-learning
"""

class DeepSARSAgent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3] # ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.population = 100
        self.state_size = 25

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999 # exploration
        self.epsilon_min = 0.005 # exploration
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]

        if done:
            target[action] = reward # 즉각적인 보상
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])


        target = np.reshape(target, [1, 5])
        self.model.fit(state, target, epochs=1, verbose=0)

# Start simulation

EPISODES = 3000

if __name__ == "__main__":
    env = Env()
    agent = DeepSARSAgent()

    global_step = 0
    scores, episodes = [], []

    print('state size:', agent.state_size)

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
            next_state = np.reshape(next_state, [1, agent.state_size])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)
            state = next_state
            score += reward
            state = copy.deepcopy(next_state)

            if i==199:
                scores.append(score)
                episodes.append(e)
