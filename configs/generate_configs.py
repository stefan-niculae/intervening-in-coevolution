from sys import argv
from copy import deepcopy
import itertools
import json
from os import makedirs

from configs.structure import read_config


DEFAULT_ROOT_DIR = 'generated'

fixed_params = {
    'scenario': '9x9,2v2,pacman-1',
    'treasure_collection_limit': 2,
    'time_limit': 100,
    'seed': 0,

    'batch_norm': True,
    'layer_norm': True,

    'num_iterations': 250,
    'num_transitions': 4000,
    'batch_size': 512,
    'num_epochs': 5,

    'log_interval': 1,
    'save_interval': 50,
    'eval_interval': 0,
}

param_grid = {
    'algorithm':    ['pg', 'ppo'],
    'architecture': ['grid→conv', 'coords→fc'],
    'conv_kernel_size': [None, 1, 3],
    'activation':   ['relu', 'tanh'],
    'encoder':      ['small', 'medium', 'large'],
    'decoder':      ['small', 'large'],
    'entropy_coef': [0.001, 0.01],
    'lr':           [0.001, 0.01],
}

translator = {
    'architecture': {
        'grid→conv': {'state_representation': 'grid',        'encoder': 'conv'},
        'coords→fc': {'state_representation': 'coordinates', 'encoder': 'fc'},
    },
    'encoder': {
        'small':  {'num_encoder_layers': 2, 'encoder_layer_size': 32},
        'medium': {'num_encoder_layers': 3, 'encoder_layer_size': 64},
        'large':  {'num_encoder_layers': 4, 'encoder_layer_size': 64},
    },
    'decoder': {
        'small':  {'num_decoder_layers': 2, 'decoder_layer_size': 16},
        'large':  {'num_decoder_layers': 3, 'decoder_layer_size': 32},
    },
}


def is_valid(d: dict):
    specifies_kernel = 'conv_kernel_size' in d
    is_convolutional = 'conv' in d['architecture']
    return specifies_kernel == is_convolutional


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
              only_values=('algorithm', 'architecture', 'activation', 'scenario'),
              remove_from_key=('num_', '_interval', '_size', 'conv_')) -> str:
    d = {}
    for k, v in comb.items():
        for to_remove in remove_from_key:
            k = k.replace(to_remove, '')

        if v is True:
            v = 'T'
        elif v is False:
            v = 'F'

        d[k] = v
    return '; '.join(str(v) if k in only_values else f'{k}={v}'
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

    fixed_name = comb_name(fixed_params)
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
