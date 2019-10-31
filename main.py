import glob
import os
import time
from collections import deque

import numpy as np
import torch

from arguments import get_args
from environment.vec_env import get_vec_normalize
from running.evaluation import evaluate
from running.training import instantiate, perform_update


LAST_N_REWARDS = 10


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + '_eval'
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    envs, actor_critic, agent, rollouts = instantiate(args, device)
    rollouts.to(device)

    episode_rewards = deque(maxlen=LAST_N_REWARDS)

    start_t = time.time()
    n_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for update_number in range(n_updates):
        value_loss, action_loss, dist_entropy = perform_update(
            args, envs, actor_critic, agent, rollouts,
            update_number, n_updates, episode_rewards)

        # save for every interval-th episode or for the last epoch
        if (update_number % args.save_interval == 0
                or update_number == n_updates - 1) and args.save_dir:
            save_model(args, envs, actor_critic)

        if update_number % args.log_interval == 0 and len(episode_rewards) > 1:
            log_progress(args, update_number, episode_rewards, time.time() - start_t,
                         value_loss, action_loss, dist_entropy)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and update_number % args.eval_interval == 0):
            ob_rms = get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


def save_model(args, envs, actor_critic):
    save_path = os.path.join(args.save_dir, args.algo)
    os.makedirs(save_path, exist_ok=True)

    torch.save([
        actor_critic,
        getattr(get_vec_normalize(envs), 'ob_rms', None)
    ], os.path.join(save_path, args.env_name + ".pt"))


def log_progress(args, update_number, episode_rewards, time_elapsed,
                 value_loss, action_loss, dist_entropy):
    total_num_steps = (update_number + 1) * args.num_processes * args.num_steps
    print(f"Updates {update_number}, \t"
          f"Timesteps {total_num_steps}, \t"
          f"FPS {int(total_num_steps / time_elapsed)}")
    print(f"Last {len(episode_rewards)} episodes reward: "
          f"min {min(episode_rewards):.1f}, \t"
          f"median {np.median(episode_rewards):.1f}, \t"
          f"avg {np.mean(episode_rewards):.1f}, \t"
          f"max {max(episode_rewards):.1f}")
    print(f"value loss {value_loss:.2f}, \t"
          f"action loss {action_loss:.2f}, \t"
          f"entropy {dist_entropy:.2f}")
    print()


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


if __name__ == "__main__":
    main()
