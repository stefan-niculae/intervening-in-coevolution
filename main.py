""" Orchestrates I/O and learning """

from time import time
from sys import argv
import warnings

import torch

from tqdm import tqdm as progress_bar

with warnings.catch_warnings():
    # Silence tensorflow (2.0) deprecated usages of numpy
    warnings.simplefilter('ignore', FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from running.training import instantiate, perform_update
from running.evaluation import simulate_episode
from running.visualization import log_layers, log_scalars, log_comparisons, log_hyperparams_and_metrics
from running.utils import paths, do_this_iteration, save_code, save_model
from environment.visualization import create_animation
from configs.structure import read_config, save_config
from intervening.scheduling import SAMPLE, DETERMINISTIC, SCRIPTED, action_source_names
from tuning.comparison import read_models, play_against_others


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

    if config.viz_scripted_mode:
        [*env_history, end_reason] = simulate_episode(env, policies, SCRIPTED)
        create_animation(env_history, video_path % (0, 'scripted'))
        return

    with warnings.catch_warnings():
        # Silence tensorflow (2.0) deprecated usages of numpy
        warnings.simplefilter('ignore', FutureWarning)
        logs_writer = SummaryWriter(logs_dir)

    if config.compare_interval > 0:
        comparison_policies, _ = read_models(config.comparison_models_dir)

    try:
        start_time = time()

        # Main training loop
        for update_number in progress_bar(range(config.num_iterations), 'Training'):
            # Collect rollouts and update weights
            training_history = perform_update(config, env, policies, storages)

            # Write progress summaries
            if do_this_iteration(config.log_interval, update_number, config.num_iterations):
                log_layers(policies, logs_writer, update_number)
                log_scalars(training_history, logs_writer, update_number, env)

            # Evaluate and record video
            if do_this_iteration(config.eval_interval, update_number, config.num_iterations):
                for sampling_method in [SAMPLE, DETERMINISTIC]:
                    [*env_history, _] = simulate_episode(env, policies, sampling_method)
                    create_animation(env_history, video_path % (update_number, action_source_names[sampling_method]))

            # Checkpoint current model weights
            if do_this_iteration(config.save_interval, update_number, config.num_iterations):
                save_model(policies, checkpoint_path % update_number)

            # Evaluate against other models
            if do_this_iteration(config.compare_interval, update_number, config.num_iterations):
                won_statuses, rewards = play_against_others(env, policies, comparison_policies, config.comparison_num_episodes)
                log_comparisons(won_statuses, rewards, logs_writer, update_number)

    except KeyboardInterrupt:
        print('Stopped training, finishing up...')

    # Save final weights
    if config.save_interval > 0:
        save_model(policies, checkpoint_path % update_number)

    if config.eval_interval > 0:
        [*env_history, _] = simulate_episode(env, policies, SAMPLE)
        create_animation(env_history, video_path % (update_number, action_source_names[SAMPLE]))

    # Save hyperparams and metrics (comparisons against others and themselves)
    selves_wons, selves_rewards     = play_against_others(env, policies, [policies],          config.comparison_num_episodes)
    if config.compare_interval > 0:
        others_wons, others_rewards = play_against_others(env, policies, comparison_policies, config.comparison_num_episodes)
    else:
        others_wons, others_rewards = None, None
    log_hyperparams_and_metrics(config,
                                selves_wons, selves_rewards,
                                others_wons, others_rewards,
                                logs_writer, start_time)

    # TODO log final
    # Flush logs
    logs_writer.close()


if __name__ == "__main__":
    main(argv[1])
