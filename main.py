""" Orchestrates I/O and learning """

from sys import argv
import torch
from tqdm import tqdm as progress_bar

from running.experiments_setup import setup
from running.training import instantiate, perform_update
from running.visualization import write_logs

LAST_N_REWARDS = 10


def main(config_path: str):
    config, logs_writer, video_recorder, model_saver = setup(config_path)

    # Set random seed
    torch.manual_seed(config.seed)

    # Instantiate
    env, policies, storages = instantiate(config)

    # Call updates
    for update_number in progress_bar(range(config.num_updates)):
        is_last_update = (update_number == config.num_updates - 1)

        perform_update(config, env, policies, storages)

        # if update_number % config.log_interval == 0 or is_last_update:
        #     write_logs(config, env, policy, rollouts, episode_number_history, logs_writer, update_number)

        if update_number % config.eval_interval == 0 or is_last_update:
            video_recorder(env, policies, update_number)

        if update_number % config.save_interval == 0 or is_last_update:
            model_saver(policies, update_number)

    logs_writer.close()


if __name__ == "__main__":
    main(argv[1])
