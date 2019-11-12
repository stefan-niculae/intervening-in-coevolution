import os
import os.path
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from configs.structure import read_config, save_config
from environment.visualization import create_animation
from running.evaluation import evaluate


_ROOT_DIR = Path('outputs/')

_TRAINED_MODELS_DIR = 'trained_models'
_VIDEOS_DIR         = 'videos'
_LOG_DIR            = 'logs'


def paths(config_path):
    config_name = os.path.basename(config_path)
    config_name = os.path.splitext(config_name)[0]
    time_string = datetime.now().strftime('%d %b %H.%M.%S')  # colons mess paths up, square brackets mess glob up

    experiment_dir = _ROOT_DIR / f'{time_string} - {config_name}'
    logs_dir    = experiment_dir / _LOG_DIR
    models_dir  = experiment_dir / _TRAINED_MODELS_DIR
    videos_dir  = experiment_dir / _VIDEOS_DIR

    videos_path = videos_dir / 'rollout-%d.gif'
    models_path = models_dir / 'checkpoint-%d.tar'
    config_save_path = experiment_dir / 'config.json'

    for dir_path in [experiment_dir, logs_dir, videos_dir, models_dir]:
        os.makedirs(str(dir_path), exist_ok=True)

    return logs_dir, config_save_path, str(videos_path), str(models_path)


def save_model(dict_to_save, save_path):
    torch.save(dict_to_save, save_path)


def load_model(checkpoint_path):
    # TODO resumable training
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def setup(config_path):
    config = read_config(config_path)

    logs_dir, config_save_path, videos_path, models_path = paths(config_path)

    logs_writer = SummaryWriter(logs_dir)
    save_config(config, config_save_path)

    def record_rollout_f(policy, update_step):
        maps, pos2ids = evaluate(config, policy)
        create_animation(maps, pos2ids, videos_path % update_step)
    save_model_f = lambda policy, step: save_model(policy, models_path % step)

    return config, logs_writer, record_rollout_f, save_model_f

