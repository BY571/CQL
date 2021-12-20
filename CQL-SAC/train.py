

import gym
import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random, evaluate
import random
from agent import CQLSAC

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC", help="Run name, default: CQL-SAC")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
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

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    
    env.seed(config.seed)
    eval_env.seed(config.seed + 123)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    with wandb.init(project="CQL", name=config.run_name, config=config):
        
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

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
        collect_random(env=env, dataset=buffer, num_samples=10000)
        
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)
        eval_reward = evaluate(eval_env, agent)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Steps": steps}, step=steps)
        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(buffer.sample())
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break


            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Lagrange Alpha Loss": lagrange_alpha_loss,
                       "CQL1 Loss": cql1_loss,
                       "CQL2 Loss": cql2_loss,
                       "Bellman error 1": bellmann_error1,
                       "Bellman error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Lagrange Alpha": lagrange_alpha,
                       "Steps": steps,
                       "Episode": i,
                       "Buffer size": buffer.__len__()})
        
            if i % config.eval_every == 0:
                eval_reward = evaluate(eval_env, agent)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Steps": steps}, step=steps)

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="CQL-DQN", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
