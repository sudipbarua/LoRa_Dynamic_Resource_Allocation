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
        self.epsilon_decay = 0.995
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
        self.alpha = 10.0   # hyperparameter for tuning PDR
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
        # epsilon greedy action selection
        if np.random.rand() <= self.epsilon:
            print("[LoRaDRL act] Random action selected")
            return random.randrange(self.action_size)

        print("[LoRaDRL act] Predicting action based on current state")
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        # print("[LoRaDRL replay] Starting replay...")
        if len(self.memory) < self.batch_size:
            # print("[LoRaDRL replay] Not enough data for continuing train/retrain. Current memory size:", len(self.memory))
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        current_q = self.model.predict(states)
        # print(f"[LoRaDRL replay] Current Q-values calculated {current_q}")
        future_q = self.target_model.predict(next_states)
        # print(f"[LoRaDRL replay] Future Q-values calculated {future_q}")
        for i in range(len(minibatch)):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(future_q[i])

        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon after each replay: expomential decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print(f"[LoRaDRL replay] Epsilon decayed to {self.epsilon}")

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


class AdaptiveResourceAllocation(LoRaDRL):
    def __init__(self, state_size, action_size, sfSet, powSet, freqSet):
        self.state_size = state_size  
        self.action_size = action_size  
        self.num_channels = len(freqSet)
        
        self.gamma = 0.7   # discount factor for future rewards
        self.epsilon = 1.0  
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 128
        self.memory = deque(maxlen=30000)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.SF_values = sfSet
        self.power_levels = powSet  
        self.channels = freqSet
        # Relative constants for assigning weights to different components of the reward function
        self.alpha = 10.0   # hyperparameter for tuning PDR
        self.beta = 0.5   # hyperparameter for tuning airtime
        
    def _build_model(self):
        return super()._build_model()

    def calculate_reward(self, PDR, airtime, pktLost):
        if pktLost:
            return -10.0  # Negative reward for lost packets
        else:
            return super().calculate_reward(PDR, airtime)


class LoRaQLAgent(AdaptiveResourceAllocation):
    def __init__(self, state_size, action_size, sfSet, powSet, freqSet):
        super().__init__(state_size, action_size, sfSet, powSet, freqSet)
        self.name = "LoRaQLAgent"
        # Defines the bins for each continuous state parameter.
        # Example: [[-120,-100,-80,-60], [-10,0,10,20], [0,100,500,1000]]
        # Each inner list defines the upper bounds of bins for a parameter.
        self.state_bins = state_bins
        self.action_size = action_size
        self.num_channels = len(freqSet) # Number of available channels

        # Calculate the number of discrete states based on the defined bins
        self.num_discrete_states = 1
        self.bin_counts = []
        for bins in self.state_bins:
            num_bins_for_param = len(bins) + 1 # Number of bins is thresholds + 1
            self.num_discrete_states *= num_bins_for_param
            self.bin_counts.append(num_bins_for_param)
        # Q-table: stores Q-values for each state-action pair
        # Initialized with zeros. Dimensions: (num_discrete_states, action_size)
        self.q_table = np.zeros((self.num_discrete_states, self.action_size))

    def _get_discrete_state(self, continuous_state):
        """
        Converts a continuous state vector into a single discrete integer index.

        Args:
            continuous_state (np.array or list): The continuous state vector (e.g., [RSSI, SNR, Distance]).

        Returns:
            int: The discrete integer index representing the state.
        """
        discrete_indices = []
        for i, param_value in enumerate(continuous_state):
            # Find which bin the parameter value falls into
            # np.digitize returns the index of the bin to which each value belongs.
            # Bins are defined by `self.state_bins[i]`.
            # Example: if bins = [-100, -80], value -90 -> index 1 (falls into bin (-100, -80])
            bin_index = np.digitize(param_value, self.state_bins[i])
            discrete_indices.append(bin_index)

        # Combine individual discrete indices into a single unique integer
        # This uses a base-N conversion, where N is the number of bins for each parameter.
        # Example: if bin_counts = [5, 4, 4] and discrete_indices = [1, 2, 3]
        # index = (1 * 4 * 4) + (2 * 4) + 3 = 16 + 8 + 3 = 27
        discrete_state_index = 0
        multiplier = 1
        for i in reversed(range(len(discrete_indices))):
            discrete_state_index += discrete_indices[i] * multiplier
            multiplier *= self.bin_counts[i] # Multiply by the number of bins in the current dimension

        # Ensure the index is within bounds of the Q-table
        return int(np.clip(discrete_state_index, 0, self.num_discrete_states - 1))
    
    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple (state, action, reward, next_state, done) in the replay memory.
        States are discretized before storage.

        Args:
            state (np.array): The current continuous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next continuous state.
            done (bool): True if the episode has ended, False otherwise.
        """
        discrete_state = self._get_discrete_state(state)
        discrete_next_state = self._get_discrete_state(next_state)
        self.memory.append((discrete_state, action, reward, discrete_next_state, done))

    def act(self, continuous_state):
        """
        Selects an action based on the current continuous state using an epsilon-greedy policy.
        The continuous state is first discretized.

        Args:
            continuous_state (np.array): The current continuous state.

        Returns:
            int: The index of the selected action.
        """
        discrete_state = self._get_discrete_state(continuous_state)

        # Epsilon-greedy action selection: explore randomly or exploit learned Q-values
        if np.random.rand() <= self.epsilon:
            print("[LoRaQLearningAgent act] Random action selected")
            return random.randrange(self.action_size)

        print("[LoRaQLearningAgent act] Selecting action from Q-table based on discrete state")
        # Exploit: choose the action with the highest Q-value for the current discrete state
        return np.argmax(self.q_table[discrete_state, :])

    def replay(self):
        # Updating the Q-table using the Bellman equation
        # Unusual naming. But we keep it for maintaining the consistency with the parent classes
        
        # Ensure enough experiences are in memory to form a batch
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size) # Sample a random mini-batch

        for state, action, reward, next_state, done in minibatch:
            # Get the current Q-value for the (state, action) pair
            current_q_value = self.q_table[state, action]

            if done:
                # If the episode ended, the target Q-value is just the reward
                target_q_value = reward
            else:
                # Calculate the maximum Q-value for the next state
                max_future_q = np.max(self.q_table[next_state, :])
                # Bellman equation for Q-learning update
                target_q_value = reward + self.gamma * max_future_q

            # Update the Q-value using the Q-learning formula
            self.q_table[state, action] = current_q_value + self.learning_rate * (target_q_value - current_q_value)

        # Decay epsilon after each learning step to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print(f"[LoRaQLearningAgent learn] Epsilon decayed to {self.epsilon:.4f}")

    def save_q_table(self, filename="q_table.npy"):
        """
        Saves the Q-table to a file using NumPy's save function.

        Args:
            filename (str): The name of the file to save the Q-table to.
        """
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        """
        Loads the Q-table from a file using NumPy's load function.

        Args:
            filename (str): The name of the file to load the Q-table from.
        """
        try:
            self.q_table = np.load(filename)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Q-table file '{filename}' not found. Initializing with zeros.")
            # Q-table remains zeros as initialized in __init__ if file not found


class DQNAgent(AdaptiveResourceAllocation):
    def __init__(self, state_size, action_size, sfSet, powSet, freqSet):
        super().__init__(state_size, action_size, sfSet, powSet, freqSet)

    def replay(self):
        # print("[LoRaDRL replay] Starting replay...")
        if len(self.memory) < self.batch_size:
            # print("[LoRaDRL replay] Not enough data for continuing train/retrain. Current memory size:", len(self.memory))
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        current_q = self.model.predict(states)
        # print(f"[LoRaDRL replay] Current Q-values calculated {current_q}")
        future_q = self.model.predict(next_states)
        # print(f"[LoRaDRL replay] Future Q-values calculated {future_q}")
        for i in range(len(minibatch)):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(future_q[i])

        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon after each replay: expomential decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print(f"[LoRaDRL replay] Epsilon decayed to {self.epsilon}")


class AraSysOptimizerAgent(AdaptiveResourceAllocation):
    def __init__(self, state_size, action_size, sfSet, powSet, freqSet):
        super().__init__(state_size, action_size, sfSet, powSet, freqSet)
        self.alpha = 10  # Tuning parameter for PRR
        self.gamma_p = 100  # Tuning parameter for energy consumption 

    def calculate_reward(self, prrSys, avgErgPerPkt, pktLost):
        return (self.alpha * prrSys * int(not pktLost)) / (self.gamma_p * avgErgPerPkt)

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