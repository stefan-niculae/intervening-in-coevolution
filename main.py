""" Orchestrates I/O and learning """

from sys import argv
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as progress_bar

from running.training import instantiate, perform_update
from running.evaluation import evaluate
from running.visualization import log_layers, log_scalars
from running.utils import paths, do_this_iteration, save_code
from environment.visualization import create_animation
from configs.structure import read_config, save_config


def main(config_path: str):
    # Read the experiment configuration
    config = read_config(config_path)

    # Generate experiment file structure
    logs_dir, config_save_path, video_path, checkpoint_path, code_save_path = paths(config_path)

    # Read config
    save_config(config, config_save_path)
    save_code(code_save_path)

    # Set random seed
    torch.manual_seed(config.seed)

    # Instantiate components
    env, policies, storages = instantiate(config)
    logs_writer = SummaryWriter(logs_dir)

    # Main loop
    for update_number in progress_bar(range(config.num_unguided_updates), 'Random'):
        # Collect rollouts and update weights
        training_history = perform_update(config, env, policies, storages)

        # Write progress summaries
        if do_this_iteration(config.log_interval, update_number, config.num_unguided_updates):
            log_layers(policies, logs_writer, update_number)
            log_scalars(training_history, logs_writer, update_number)

        # Evaluate and record video
        if do_this_iteration(config.eval_interval, update_number, config.num_unguided_updates):
            for deterministic in [True, False]:
                env_history = evaluate(env, policies, deterministic)
                suffix = 'deterministic' if deterministic else 'sampling'
                create_animation(env_history, video_path % (update_number, suffix))

        # Checkpoint current model weights
        if do_this_iteration(config.save_interval, update_number, config.num_unguided_updates):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                torch.save(policies, checkpoint_path % update_number)

    # TODO log final

    # Flush logs
    logs_writer.close()


if __name__ == "__main__":
    main(argv[1])
