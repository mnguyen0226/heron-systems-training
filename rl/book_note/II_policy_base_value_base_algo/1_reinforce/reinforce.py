from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Policy Network with simple 1 layer MLP wiht 64 hidden unit
        
        Parameters
        ----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        """
        super().__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ] 
        self.model = nn.Sequential(*layers) # NN
        self.onpolicy_reset() # reset policy
        self.train() # set training mode

    def onpolicy_reset(self):
        """Reset the log probability and rewards
        """
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        """Feed forward of the NN

        Parameters
        ----------
        x: 
            input to NN
        """
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        """action of agent on the environment with the given state

        Parameters
        ----------
        state:
            state of the environment
        """
        x = torch.from_numpy(state.astype(np.float32)) # to tensor
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action.item()

def train(pi, optimizer):
    """Implements the update step
    The loss is expressed as sum of the negative log prob multiplied by the return
    Pytorch optimizer minimize the loss while we want to maximize the objective

    Parameters
    ----------
    optimizer:
        type of optimizer
    """
    # Inner gradient-ascent loop of REINFORCE algorithms
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # the return   
    future_ret = 0.0

    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets # gradient term; negative for maximzing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward() # backprob, compute gradient
    optimizer.step() # gradient-ascent, update the weight
    return loss

def main():
    """Construct CartPole environment, the policy network Pi, and an optimizer.
    Run training loop for 300 episodes, as the training progresses, the total 
    reward per episode should increase towards 200.
    The environment is solve when the total reward is above 195
    """
    env = gym.make("CartPole-v0")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr = 0.01)
    for epi in range(300):
        state = env.reset()
        for t in range(200): # cartpole max time step is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break

        loss = train(pi, optimizer) # train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset() # On policy: clear memory after training
        print(f"Episode {epi}, loss: {loss}, total_rewards: {total_reward}, solved: {solved}")

if __name__ == "__main__":
    main()
