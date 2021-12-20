import gym
import d4rl
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save, collect_random
import random
from agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    
    args = parser.parse_args()
    return args

def prep_dataloader(env_id="halfcheetah-medium-v2", batch_size=256, seed=1):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if  k is not "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    if "halfcheetah" in env_id:
        eval_env = gym.make("HalfCheetah-v2")
    eval_env.seed(seed)
    return dataloader, eval_env

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataloader, env = prep_dataloader(env_id=config.env, batch_size=config.batch_size, seed=config.seed)

    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batches = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="CQL-offline", name=config.run_name, config=config):
        
        agent = CQLSAC(state_size=env.observation_space.shape[0],
                        action_size=env.action_space.shape[0],
                        tau=config.tau,
                        hidden_size=config.hidden_size,
                        learning_rate=config.learning_rate,
                        temp=config.temperature,
                        with_lagrange=config.with_lagrange,
                        cql_weight=config.cql_weight,
                        target_action_gap=config.target_action_gap,
                        device=device)

        wandb.watch(agent, log="gradients", log_freq=10)
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        eval_reward = evaluate(env, agent)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        for i in range(1, config.episodes+1):

            for batch_idx, experience in enumerate(dataloader):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn((states, actions, rewards, next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                eval_reward = evaluate(env, agent)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches,))
            
            wandb.log({
                       "Average10": np.mean(average10),
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Lagrange Alpha Loss": lagrange_alpha_loss,
                       "CQL1 Loss": cql1_loss,
                       "CQL2 Loss": cql2_loss,
                       "Bellman error 1": bellmann_error1,
                       "Bellman error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Lagrange Alpha": lagrange_alpha,
                       "Batches": batches,
                       "Episode": i})

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
