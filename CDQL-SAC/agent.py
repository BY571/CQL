import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import IQN, Actor
import numpy as np
import math
import copy


class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = IQN(state_size, action_size, hidden_size, seed=1).to(device)
        self.critic2 = IQN(state_size, action_size, hidden_size, seed=2).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = IQN(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = IQN(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1.get_qvalues(states, actions_pred.squeeze(0))   
        q2 = self.critic2.get_qvalues(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        with torch.no_grad():
            actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        qs1 = self.critic1.get_qvalues(obs_q, actions_pred)
        qs2 = self.critic2.get_qvalues(obs_q, actions_pred)
        
        return qs1-log_pis, qs2-log_pis
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic.get_qvalues(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, _ = self.actor_local.evaluate(next_states)
            #next_action = next_action.unsqueeze(1).repeat(1, 10, 1).view(next_action.shape[0] * 10, next_action.shape[1])
            #temp_next_states = next_states.unsqueeze(1).repeat(1, 10, 1).view(next_states.shape[0] * 10, next_states.shape[1])
            
            Q_target1_next, _ = self.critic1_target(next_states, next_action) #.view(states.shape[0], 10, 1).max(1)[0].view(-1, 1)
            # batch_size, num_tau, 1    
            Q_target2_next, _ = self.critic2_target(next_states, next_action) #.view(states.shape[0], 10, 1).max(1)[0].view(-1, 1)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next).transpose(1,2)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu().unsqueeze(-1) + (gamma * (1 - dones.cpu().unsqueeze(-1)) * Q_target_next.cpu()) 


        # Compute critic loss
        q1, taus1 = self.critic1(states, actions)
        q2, taus2 = self.critic2(states, actions)
        assert Q_targets.shape == (256, 1, 32), "have shape: {}".format(Q_targets.shape)
        assert q1.shape == (256, 32, 1)
        
        # Quantile Huber loss
        td_error1 = Q_targets - q1.cpu()
        td_error2 = Q_targets - q2.cpu()
        
        assert td_error1.shape == (256, 32, 32), "wrong td error shape"
        huber_l_1 = calculate_huber_loss(td_error1, 1.0)
        huber_l_2 = calculate_huber_loss(td_error2, 1.0)
        
        quantil_l_1 = abs(taus1.cpu() - (td_error1.detach() < 0).float()) * huber_l_1 / 1.0
        quantil_l_2 = abs(taus2.cpu() - (td_error2.detach() < 0).float()) * huber_l_2 / 1.0

        critic1_loss = quantil_l_1.sum(dim=1).mean(dim=1).mean()
        critic2_loss = quantil_l_2.sum(dim=1).mean(dim=1).mean()

        
        # CQL addon

        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int (random_actions.shape[0] / states.shape[0])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat, next_states.shape[1])
        
        current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
        
        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0], num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0], num_repeat, 1)

        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)
        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)      
        
        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        
        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        

        cql1_scaled_loss = (torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp - q1.mean()) * self.cql_weight
        cql2_scaled_loss = (torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp - q2.mean()) * self.cql_weight
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss
