import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class LoRaDRL:
    def __init__(self, state_size, action_size, sfSet, powSet, freqSet):
        self.state_size = state_size  
        self.action_size = action_size  
        self.num_channels = len(freqSet)

        
        self.gamma = 0.7   # discount factor for future rewards
        self.epsilon = 1.0  
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.00005
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.memory = deque(maxlen=30000)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        
        self.SF_values = sfSet
        self.power_levels = powSet  
        self.channels = freqSet
        # Relative constants for assigning weights to different components of the reward function
        self.alpha = 1.0   # hyperparameter for tuning PDR
        self.beta = 0.5   # hyperparameter for tuning airtime
        self.gamma_p = 0.3   # hyperparameter for tuning power usage

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # The correct keyword for learning rate declaration now is learning_rate
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        current_q = self.model.predict(states)

        future_q = self.target_model.predict(next_states)

        for i in range(len(minibatch)):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(future_q[i])

        self.model.fit(states, current_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def calculate_reward(self, PDR, airtime, power_chosen=0):
        power_max = max(self.power_levels)
        power_min = min(self.power_levels)
        # power_reward = (power_max - power_chosen) / (power_max - power_min)

        # reward = (self.alpha * PDR -
        #          self.beta * airtime +
        #          self.gamma_p * power_reward)
        reward = (self.alpha * PDR - self.beta * airtime)

        return reward

    def get_phy_parameters(self, action_idx):
        sf_idx = action_idx % len(self.SF_values)
        remaining = action_idx // len(self.SF_values)
        power_idx = remaining % len(self.power_levels)
        channel_idx = remaining // len(self.power_levels) % self.num_channels

        return {
            'SF': self.SF_values[sf_idx],
            'power': self.power_levels[power_idx],
            'channel': self.channels[channel_idx]
        }

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


########## test script ##########
def main():
    # Define dummy sets
    sfSet = [7, 8, 9]
    powSet = [2, 5, 8]
    freqSet = [868100, 868300]
    state_size = 4
    action_size = len(sfSet) * len(powSet) * len(freqSet)

    # Initialize agent
    agent = LoRaDRL(state_size, action_size, sfSet, powSet, freqSet)
    print("Agent initialized.")

    # Create a dummy state
    state = np.random.rand(state_size)
    print("Dummy state:", state)

    # Test action selection
    action = agent.act(state)
    print("Selected action:", action)
    print("Decoded action:", agent.get_phy_parameters(action))

    # Test reward calculation
    reward = agent.calculate_reward(PDR=0.8, airtime=0.5, power_chosen=5)
    print("Calculated reward:", reward)

    # Test memory and replay
    next_state = np.random.rand(state_size)
    agent.remember(state, action, reward, next_state, False)
    # Fill memory to batch size for replay
    for _ in range(agent.batch_size):
        s = np.random.rand(state_size)
        a = np.random.randint(0, action_size)
        r = np.random.rand()
        ns = np.random.rand(state_size)
        d = np.random.choice([True, False])
        agent.remember(s, a, r, ns, d)
    print("Memory filled. Running replay...")
    agent.replay()
    print("Replay complete.")

    # # Test saving and loading weights
    # agent.save("test_weights.h5")
    # print("Weights saved.")
    # agent.load("test_weights.h5")
    # print("Weights loaded.")

if __name__ == "__main__":
    main()