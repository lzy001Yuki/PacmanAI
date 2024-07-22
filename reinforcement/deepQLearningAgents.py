import os.path

import model
from qlearningAgents import PacmanQAgent
from backend import ReplayMemory
import layout
import copy
import torch
import numpy as np

class PacmanDeepQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", target_update_rate=300, doubleQ=True, **args):
        PacmanQAgent.__init__(self, **args)
        self.model = None
        self.target_model = None
        self.target_update_rate = target_update_rate
        self.update_amount = 0
        self.epsilon_explore = 0.0001
        self.epsilon0 = 0.0001
        self.minimal_epsilon = 0.0001
        self.epsilon = self.epsilon0
        self.discount = 0.95
        self.update_frequency = 3
        self.counts = None
        self.replay_memory = ReplayMemory(50000)
        self.min_transitions_before_training = 100
        self.td_error_clipping = 10

        # Initialize Q networks:
        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input
        self.state_dim = self.get_state_dim(layout_instantiated)
        self.initialize_q_networks(self.state_dim)

        self.doubleQ = doubleQ
        if self.doubleQ:
            self.target_update_rate = -1

    def get_state_dim(self, layout):
        pac_ft_size = 2
        ghost_ft_size = 2 * layout.getNumGhosts()
        food_capsule_ft_size = layout.width * layout.height
        return pac_ft_size + ghost_ft_size + food_capsule_ft_size

    def get_features(self, state):
        pacman_state = np.array(state.getPacmanPosition())
        ghost_state = np.array(state.getGhostPositions())
        capsules = state.getCapsules()
        food_locations = np.array(state.getFood().data).astype(np.float32)
        for x, y in capsules:
            food_locations[x][y] = 2
        return np.concatenate((pacman_state, ghost_state.flatten(), food_locations.flatten()))

    def initialize_q_networks(self, state_dim, action_dim=5):
        import model
        self.model = model.DeepQNetwork(state_dim, action_dim)
        self.target_model = model.DeepQNetwork(state_dim, action_dim)
        if os.path.exists('para.best.bin'):
            print('has initialized--------------------')
            checkpoint = torch.load('para.best.bin')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.model.optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            self.target_model.optimizer.load_state_dict(checkpoint['target_model_optimizer_state_dict'])
            self.replay_memory = checkpoint['memory']
            #print(self.model.state_dict())

    def save_model(self, filename="para.bin"):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'target_model_state_dict': self.target_model.state_dict(),
                    'model_optimizer_state_dict': self.model.optimizer.state_dict(),
                    'target_model_optimizer_state_dict': self.target_model.optimizer.state_dict(),
                    'memory': self.replay_memory},filename)
        #print(self.model.state_dict())

    def getQValue(self, state, action):
        """
          Should return Q(state,action) as predicted by self.model
        """
        feats = self.get_features(state)
        legalActions = self.getLegalActions(state)
        action_index = legalActions.index(action)
        state = torch.tensor(np.array([feats]).astype("float64"), dtype=torch.double)
        return self.model.run(state).data[0][action_index]


    def shape_reward(self, reward):
        if reward > 100:
            reward = 10
        elif reward > 0 and reward < 10:
            reward = 2
        elif reward == -1:
            reward = 0
        elif reward < -100:
            reward = -10
        return reward


    def compute_q_targets(self, minibatch, network = None, target_network=None, doubleQ=False):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            float: Loss value
        """
        if network is None:
            network = self.model
        if target_network is None:
            target_network = self.target_model
        states = np.vstack([x.state for x in minibatch])
        states = torch.tensor(states, dtype=torch.double)
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        next_states = torch.tensor(next_states)
        done = np.array([x.done for x in minibatch])

        Q_predict = network.run(states).data.detach().numpy()
        Q_target = np.copy(Q_predict )
        state_indices = states.int().detach().numpy()
        state_indices = (state_indices[:, 0], state_indices[:, 1])
        exploration_bonus = 1 / (2 * np.sqrt((self.counts[state_indices] / 100)))

        replace_indices = np.arange(actions.shape[0])
        action_indices = np.argmax(network.run(next_states).data, axis=1)
        target = rewards + exploration_bonus + (1 - done) * self.discount * target_network.run(next_states).data[replace_indices, action_indices].detach().numpy()

        Q_target[replace_indices, actions] = target

        if self.td_error_clipping is not None:
            Q_target = Q_predict + np.clip(
                     Q_target - Q_predict, -self.td_error_clipping, self.td_error_clipping)

        return Q_target

    def update(self, state, action, nextState, reward):
        legalActions = self.getLegalActions(state)
        action_index = legalActions.index(action)
        done = nextState.isLose() or nextState.isWin()
        reward = self.shape_reward(reward)

        if self.counts is None:
            x, y = np.array(state.getFood().data).shape
            self.counts = np.ones((x, y))

        state = self.get_features(state)
        nextState = self.get_features(nextState)
        self.counts[int(state[0])][int(state[1])] += 1

        transition = (state, action_index, reward, nextState, done)
        self.replay_memory.push(*transition)


        if len(self.replay_memory) < self.min_transitions_before_training:
            self.epsilon = self.epsilon_explore
        else:
            self.epsilon = max(self.epsilon0 * (1 - self.update_amount / 20000), 0)


        if len(self.replay_memory) > self.min_transitions_before_training and self.update_amount % self.update_frequency == 0:
            minibatch = self.replay_memory.pop(self.model.batch_size)
            states = np.vstack([x.state for x in minibatch])
            states = torch.tensor(states.astype("float64"), dtype=torch.double)
            Q_target1 = self.compute_q_targets(minibatch, self.model, self.target_model, doubleQ=self.doubleQ)
            Q_target1 = torch.tensor(Q_target1.astype("float64"), dtype=torch.double)

            if self.doubleQ:
                Q_target2 = self.compute_q_targets(minibatch, self.target_model, self.model, doubleQ=self.doubleQ)
                Q_target2 = torch.tensor(Q_target2.astype("float64"), dtype=torch.double)
            
            self.model.gradient_update(states, Q_target1)
            if self.doubleQ:
                self.target_model.gradient_update(states, Q_target2)

        if self.target_update_rate > 0 and self.update_amount % self.target_update_rate == 0:
            self.target_model.set_weights(copy.deepcopy(self.model.parameters))

        self.update_amount += 1


    def final(self, state):
        """Called at the end of each game."""
        PacmanQAgent.final(self, state)
