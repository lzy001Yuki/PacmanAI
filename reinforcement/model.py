
"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Module
from torch.nn import Linear
from torch import tensor, double, optim
from torch.nn.functional import relu, mse_loss



class DeepQNetwork(Module):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        super(DeepQNetwork, self).__init__()
        # Remember to set self.learning_rate, self.numTrainingGames,
        # and self.batch_size!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 0.005
        self.numTrainingGames = 5000
        self.batch_size = 128



        self.layer1 = Linear(state_dim, 256)
        self.layer2 = Linear(256, 512)
        self.layer3 = Linear(512, 128)
        self.layer4 = Linear(128, action_dim)

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        self.double()


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        Q_target_tensor = tensor(Q_target, dtype=double, device=self.device)
        prediction = self.forward(states)
        return mse_loss(prediction, Q_target_tensor)


    def forward(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        if states.device.type != self.device.type:
            states = states.to(self.device)
        output1 = self.layer1(states)
        input2 = relu(output1)
        output2 = self.layer2(input2)
        input3 = relu(output2)
        output3 = self.layer3(input3)
        input4 = relu(output3)
        output4 = self.layer4(input4)
        return output4

    
    def run(self, states):
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        You can look at the ML project for an idea of how to do this, but note that rather
        than iterating through a dataset, you should only be applying a single gradient step
        to the given datapoints.

        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """

        self.optimizer.zero_grad()
        loss = self.get_loss(states, Q_target)
        loss.backward()
        self.optimizer.step()