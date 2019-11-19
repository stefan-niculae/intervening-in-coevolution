import numpy as np
from pathlib import Path
from datetime import datetime
import os
import os.path
from shutil import copytree, ignore_patterns, make_archive, rmtree


ROOT_DIR = Path('outputs/')

TRAINED_MODELS_DIR = 'trained_models'
VIDEOS_DIR         = 'videos'
LOG_DIR            = 'logs'


def paths(config_path: str) -> (str, str, str, str):
    config_name = os.path.basename(config_path)
    config_name = os.path.splitext(config_name)[0]
    time_string = datetime.now().strftime('%d %b %H.%M.%S')  # colons mess paths up, square brackets mess glob up

    experiment_dir = ROOT_DIR / f'{time_string} - {config_name}'
    logs_dir    = experiment_dir / LOG_DIR
    models_dir  = experiment_dir / TRAINED_MODELS_DIR
    videos_dir  = experiment_dir / VIDEOS_DIR

    videos_path = videos_dir / 'rollout-%d (%s).gif'
    models_path = models_dir / 'checkpoint-%d.tar'
    config_save_path = experiment_dir / 'config.json'
    code_save_path   = experiment_dir / 'code'  # zip added automatically at the end!

    for dir_path in [experiment_dir, logs_dir, videos_dir, models_dir]:
        os.makedirs(str(dir_path), exist_ok=True)

    return logs_dir, config_save_path, str(videos_path), str(models_path), code_save_path


def do_this_iteration(interval: int, current_iteration: int, total_iterations: int):
    # Zero disables
    if interval == 0:
        return False

    # Do at the last iteration
    if current_iteration == total_iterations - 1:
        return True

    # Don't do at the first step
    if current_iteration == 0:
        return False

    # Every `interval` steps
    return current_iteration % interval == 0


def save_code(save_path: str):
    """ Saves all files not present in .gitignore """
    with open('.gitignore') as f:
        ignored = f.read().splitlines()
    ignored.append('.git')

    # Copy file tree recursively
    copytree('.', save_path, ignore=ignore_patterns(*ignored))

    # Archive it
    make_archive(save_path, format='zip', root_dir=save_path)

    # Remove the temporary, unarchived folder
    rmtree(save_path)


class EpisodeAccumulator:
    def __init__(self, *shape):
        self.history = []
        self.current = np.zeros(shape)

    def episode_over(self):
        self.history.append(self.current.copy())
        self.current[:] = 0

    def final_history(self, drop_last: bool) -> np.array:
        history = np.array(self.history)
        if drop_last:
            return history[:-1]
        else:
            return history
