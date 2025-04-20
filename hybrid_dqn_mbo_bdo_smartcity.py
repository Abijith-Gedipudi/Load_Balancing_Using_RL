import numpy as np
import random
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from collections import deque

# Environment Setup
class SmartCityEnv(gym.Env):
    def __init__(self):
        super(SmartCityEnv, self).__init__()
        self.state_space = 10
        self.action_space = 4
        self.state = np.random.rand(self.state_space)

    def step(self, action):
        reward = np.random.rand() * 10 - action
        next_state = np.random.rand(self.state_space)
        done = np.random.choice([True, False], p=[0.1, 0.9])
        return next_state, reward, done, {}

    def reset(self):
        self.state = np.random.rand(self.state_space)
        return self.state

# Deep Q-Network (DQN) Model
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(np.array([state]), verbose=0)[0])

# MBO, BDO & Hybrid Optimization
class MigratingBirdsOptimization:
    def optimize(self, agent_weights):
        return [w + np.random.uniform(-0.1, 0.1, w.shape) for w in agent_weights]

class BottlenoseDolphinOptimization:
    def refine(self, agent_weights, agent):
        return [w + np.random.uniform(-0.05, 0.05, w.shape) for w in agent_weights]

class HybridOptimization:
    def __init__(self, mbo, bdo):
        self.mbo = mbo
        self.bdo = bdo

    def optimize(self, agent):
        agent_weights = agent.model.get_weights()
        optimized_weights = self.mbo.optimize(agent_weights)
        refined_weights = self.bdo.refine(optimized_weights, agent)
        return refined_weights

# Training the Agent
env = SmartCityEnv()
agent = DQNAgent(state_size=env.state_space, action_size=env.action_space)
mbo, bdo = MigratingBirdsOptimization(), BottlenoseDolphinOptimization()
hybrid_optimizer = HybridOptimization(mbo, bdo)

num_episodes = 100
rewards_history = []

for episode in range(num_episodes):
    state, total_reward, done = env.reset(), 0, False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state, total_reward = next_state, total_reward + reward
    rewards_history.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Save & Plot
agent.model.save("hybrid_marl_mbo_bdo.h5")
plt.plot(rewards_history)
plt.show()
