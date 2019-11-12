""" Gathers arguments, runs learning steps, defines reporting and other I/O """
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
    torch.cuda.manual_seed_all(config.seed)

    # Set hardware configuration
    torch.set_num_threads(1)
    device = torch.device('cuda:0' if config.cuda else 'cpu')

    envs, policy, agent, rollouts, lr_decay = instantiate(config, device)
    rollouts.to(device)

    for update_number in progress_bar(range(config.num_updates)):
        is_last_update = (update_number == config.num_updates - 1)

        value_loss, action_loss, dist_entropy, episode_number_history = perform_update(
            config, envs, policy, agent, rollouts)

        lr_decay.step(update_number)

        if update_number % config.log_interval == 0 or is_last_update:
            write_logs(config, envs, policy, rollouts, episode_number_history, logs_writer, update_number)

        if update_number % config.eval_interval == 0 or is_last_update:
            video_recorder(policy, update_number)

        if update_number % config.save_interval == 0 or is_last_update:
            model_saver({'policy': policy, 'update_number': update_number, 'lr_decay': lr_decay}, update_number)

        rollouts.clear()

    logs_writer.close()
    print('Did', config.num_updates, 'updates successfully.')


if __name__ == "__main__":
    main(argv[1])
