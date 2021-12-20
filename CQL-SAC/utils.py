import torch
import numpy as np

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

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