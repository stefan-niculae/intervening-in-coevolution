import numpy as np

from collections import namedtuple
from environment.ThievesGuardiansEnv import (
    EMPTY    as _,
    WALL     as W,
    THIEF    as T,
    GUARDIAN as G,
    TREASURE as S,
)

ScenarioConfig = namedtuple('ScenarioConfig', 'width height n_thieves n_guardians wall_density time_limit fixed_map')


_fixed_scenario_maps = {
    '4x4-test-movements': [
        [_, _, _, _],
        [_, T, _, _],
        [_, _, G, _],
        [_, _, _, S],
    ],
    '4x4-test-movements-walls': [
        [_, W, _, _],
        [W, T, W, _],
        [_, W, G, W],
        [_, _, W, _],
    ],
    '4x4-test-rewards': [
        [_, S, _, _],
        [_, T, _, _],
        [_, G, _, _],
        [_, _, _, _],
    ],
    '4x4-thief-treasure': [
        [T, _, _, _],
        [_, _, _, _],
        [_, _, _, _],
        [_, _, _, S],
    ],

    '4x4-thief-guardian-treasure': [
        [T, _, _, G],
        [_, _, _, _],
        [_, _, _, _],
        [_, _, _, S],
    ],

    '6x6-2xthief-2xguardian': [
        [T, _, _, _, _, G],
        [T, _, _, _, _, G],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, S],
    ],

    # TODO
    '6x6-2xthief-2xguardian-walls': [
        [T, _, W, _, _, G],
        [T, _, _, W, _, G],
        [_, _, _, _, _, _],
        [_, _, W, W, _, _],
        [_, _, _, _, _, W],
        [_, _, _, _, _, S],
    ],
}


def fixed_scenario_config(name):
    map = np.array(_fixed_scenario_maps[name])
    return ScenarioConfig(
        width=map.shape[0],
        height=map.shape[1],
        n_thieves=np.sum(map == T),
        n_guardians=np.sum(map == G),
        wall_density=(map == W).mean(),
        time_limit=999,
        fixed_map=map,
    )


random_scenario_configs = {
    's': ScenarioConfig(
        width=5,
        height=5,
        n_thieves=1,
        n_guardians=1,
        wall_density=0.,
        time_limit=50,
        fixed_map=None,
    ),

    'm': ScenarioConfig(
        width=8,
        height=8,
        n_thieves=2,
        n_guardians=2,
        wall_density=.2,
        time_limit=150,
        fixed_map=None,
    ),

    'l': ScenarioConfig(
        width=12,
        height=12,
        n_thieves=3,
        n_guardians=3,
        wall_density=.4,
        time_limit=400,
        fixed_map=None,
    )
}
