""" Gathers arguments, runs learning steps, defines reporting and other I/O """
import json
from sys import argv

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as progress_bar

from configs.structure import Config
from configs.paths import MODEL_CHECKPOINTS, LOGS, VIDEOS
from running.training import instantiate, perform_update
from running.visualization import write_summary
from environment.visualization import record_rollout

LAST_N_REWARDS = 10


def main(config_path: str):
    config = read_config(config_path)

    # Set random seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Set hardware configuration
    torch.set_num_threads(1)
    device = torch.device('cuda:0' if config.cuda else 'cpu')

    envs, policy, agent, rollouts, lr_decay = instantiate(config, device)
    rollouts.to(device)

    writer = SummaryWriter('outputs/logs')

    for update_number in progress_bar(range(config.num_updates)):
        is_last_update = (update_number == config.num_updates - 1)

        value_loss, action_loss, dist_entropy, episode_number_history = perform_update(
            config, envs, policy, agent, rollouts)

        lr_decay.step(update_number)

        if update_number % config.log_interval == 0:
            write_summary(config, envs, rollouts, episode_number_history, writer, update_number)

        if is_last_update: #update_number % config.eval_interval == 0:
            record_rollout(config, policy, VIDEOS)

        # Save for every interval-th episode or for the last epoch
        # if (update_number % config.save_interval == 0
        #         or update_number == config.num_updates - 1):
        #     save_model(MODEL_CHECKPOINTS, envs, policy)
        rollouts.clear()

    print('Did', config.num_updates, 'updates successfully.')


def read_config(config_path: str) -> Config:
    with open(config_path) as f:
        dict_obj = json.load(f)
    config = Config()
    for k, v in dict_obj.items():
        setattr(config, k, v)
    return config


# def save_model(args, envs, policy):
#     # TODO also save config, and place them all in an "experiment", with the experiment's name being the time it started running
#     save_path = MODEL_CHECKPOINTS
#     os.makedirs(save_path, exist_ok=True)
#
#     torch.save([
#         policy,
#         None,  # getattr(get_vec_normalize(envs), 'ob_rms', None)
#     ], os.path.join(save_path, args.env_name + ".pt"))


def log_progress(update_number, rewards_history, value_loss, action_loss, dist_entropy):
    print(f"Updates {update_number}, \t")

    multi_sided_rewards = np.array(rewards_history)
    print(f"Last {len(rewards_history)} episodes reward: "
          f"min {np.min(multi_sided_rewards, axis=0)}, \t"
          f"median {np.median(multi_sided_rewards, axis=0)}, \t"
          f"avg {np.mean(multi_sided_rewards, axis=0)}, \t"
          f"max {np.max(multi_sided_rewards, axis=0)}")
    print(f"value loss {value_loss:.2f}, \t"
          f"action loss {action_loss:.2f}, \t"
          f"entropy {dist_entropy:.2f}")
    print()


if __name__ == "__main__":
    main(argv[1])
