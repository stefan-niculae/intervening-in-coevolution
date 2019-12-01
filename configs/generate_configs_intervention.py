from sys import argv
from copy import deepcopy
import itertools
import json
from os import makedirs

from configs.structure import read_config


DEFAULT_ROOT_DIR = 'generated'

fixed_params = {
    # TODO fill in with best params
    'algorithm': 'pg',
    'state_representation': 'coordinates',
    'encoder': 'fc',
    'num_encoder_layers': 1,
    'encoder_layer_size': 8,
    'num_decoder_layers': 1,
    'decoder_layer_size': 4,
    'activation': 'relu',
    'lr': 0.01,
    'entropy_coef': 0.001,

    'scenario': '9x9,2v2,pacman-1',
    'treasure_collection_limit': 2,
    'time_limit': 100,

    'batch_norm': False,
    'layer_norm': False,

    'max_run_time': 1,
    'num_iterations': 250,
    'num_transitions': 4000,
    'batch_size': 512,
    'num_epochs': 5,

    'log_interval': 1,
    'save_interval': 50,
    'eval_interval': 0,
}

translator = {}

param_grid = {
    'seed': [1, 2, 3],
    'winrate_threshold': [None, .6, .7, .8],
    'adjust_lr_to': [None, 0, 0.0001],
    'adjust_mi_to': [None, 1, .25],
    'adjust_uniform_to': [None, .25, .66],
    'adjust_scripted_to': [None, .25, .66],  # TODO!!! for the loser
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

    fixed_name = 'best-model'
    output_dir = root_dir + '/' + fixed_name
    makedirs(output_dir, exist_ok=True)

    for i, comb in enumerate(carthesian_product(param_grid)):
        name = comb_name(comb)  # before parsing
        comb = parse(comb)

        path = f'{output_dir}/#{i} - {name}.json'
        with open(path, 'w') as f:
            json.dump(comb, f, indent=2)

        # Check that it can be read successfully
        read_config(path)

    print(f'Generated {i} configs')
