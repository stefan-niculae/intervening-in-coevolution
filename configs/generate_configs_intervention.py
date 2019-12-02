from sys import argv
from copy import deepcopy
import itertools
import json
from os import makedirs

from configs.structure import read_config


DEFAULT_ROOT_DIR = 'intervention-comparison'

fixed_params = {
    'scenario': '9x9,2v2,pacman-1',
    "treasure_collection_limit": 2,
    'time_limit': 80,

    'algorithm': 'ppo',
    "discount": 0.96,

    'state_representation': 'grid',
    'encoder': 'conv',
    'encoder_layer_size': 'hardcoded',
    'decoder_layer_size': 'hardcoded',
    'activation': 'leaky_relu',
    'batch_norm': True,

    "entropy_coef_milestones":   [    0,    150,   300,    400],
    "entropy_coef_values":       [0.006,  0.003, 0.001, 0.0001],
    "lr_milestones":             [    0,    200],
    "lr_values":                 [0.005,  0.001],
    "max_grad_norm": 5,

    "num_iterations": 501,
    "num_transitions": 4000,
    "batch_size": 512,
    "num_epochs": 8,

    "log_interval": 1,
    "eval_interval": 25,
    "save_interval": 100,

    'first_no_adjustment': 50,
}

translator = {}

param_grid = {
    'seed': [1, 2],
    'winrate_threshold': [None, .6, .66, .7, .8],
    'adjust_lr_to': [None, 0],
    'adjust_mi_to': [None, .2],
    'adjust_uniform_to': [None, .33],
    'adjust_scripted_to': [None, .5],
}


def is_valid(d: dict):
    adjustments_enabled = [
        'adjust_lr_to' in d,
        'adjust_mi_to' in d,
        'adjust_uniform_to' in d,
        'adjust_scripted_to' in d,
    ]
    num_enabled = sum(adjustments_enabled)
    if num_enabled > 1:
        return False

    if 'winrate_threshold' in d:
        t = d['winrate_threshold']
        if 'adjust_mi_to' in d:
            return t == .66
        if 'adjust_uniform_to' in d:
            return t == .66
        if 'adjust_scripted_to' in d:
            return t in [.6, .7]
        if 'adjust_lr_to' in d:
            return t in [.6, .7, .8]

    return (num_enabled == 1) == ('winrate_threshold' in d)


def carthesian_product(d):
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        d = dict(zip(keys, instance))
        for k, v in deepcopy(d).items():
            if v is None:
                del d[k]
        if is_valid(d):
            yield d


def comb_name(comb: dict,
              remove_from_key=('adjust_', '_to', 'winrate_')) -> str:
    d = {}
    for k, v in comb.items():
        for to_remove in remove_from_key:
            k = k.replace(to_remove, '')

        if v is True:
            v = 'T'
        elif v is False:
            v = 'F'

        d[k] = v
    return '; '.join(f'{k}={v}'
                     for k, v in d.items())


def parse(comb: dict) -> dict:
    translated = {}
    for k, v in comb.items():
        if k in translator:
            translations = translator[k]
            translated.update(translations[v])
        else:
            translated[k] = v
    return {**translated, **fixed_params}


if __name__ == '__main__':
    if len(argv) > 1:
        root_dir = argv[1]
    else:
        root_dir = DEFAULT_ROOT_DIR

    makedirs(root_dir, exist_ok=True)

    for i, comb in enumerate(carthesian_product(param_grid)):
        name = comb_name(comb)  # before parsing
        comb = parse(comb)

        path = f'{root_dir}/#{i} - {name}.json'
        with open(path, 'w') as f:
            json.dump(comb, f, indent=2)

        # Check that it can be read successfully
        read_config(path)

    print(f'Generated {i} configs')
