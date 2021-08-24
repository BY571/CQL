import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import Critic, Actor


class CQLSAC():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        args,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-3
        hidden_size = args.layer_size

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=args.lr_a) 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, device, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=args.lr_a)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, device, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, device, hidden_size).to(device)
        
        self.critic1_target = Critic(state_size, action_size, device, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, device, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr_c, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr_c, weight_decay=0) 

    
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

        q1 = self.critic1(states, actions_pred.squeeze(0))   
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, log_pis_next = self.actor_local.evaluate(next_states)

            Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
            Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

            # take the mean of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu())) 





        # Compute critic loss
        q1 = self.critic1(states, actions).cpu()
        q2 = self.critic2(states, actions).cpu()
        
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)
        
        # CQL addon
        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        current_actions, log_pis = self.actor_local.evaluate(states)
        new_actions, new_log_pis = self.actor_local.evaluate(next_states)
        
        q1_random = self.critic1(states, random_actions)
        q2_random = self.critic2(states, random_actions)
        
        q1_a_s = self.critic1(current_actions, states)
        q2_a_s = self.critic2(current_actions, states)
        
        q1_next_a_s = self.critic1(new_actions, states)
        q2_next_a_s = self.critic2(new_actions, states)
        
        cat_q1 = torch.cat([q1_random, q1, q1_next_a_s, q1_a_s], 1)
        cat_q2 = torch.cat([q2_random, q2, q2_next_a_s, q2_a_s], 1)
        
        cql1_loss = torch.logsumexp(cat_q1 / 10, dim=1).mean() - q1.mean()
        cql2_loss = torch.logsumexp(cat_q2 / 10, dim=1).mean() - q2.mean()
        
        total_c1_loss = critic1_loss + cql1_loss
        total_c2_loss = critic2_loss + cql2_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            actor_loss, log_pis = self.calc_policy_loss(states, self.alpha)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Compute alpha loss
            alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

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
