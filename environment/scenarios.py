from dataclasses import dataclass
import numpy as np

from environment.thieves_guardians_env import (
    EMPTY    as _,
    WALL     as W,
    THIEF    as T,
    GUARDIAN as G,
    TREASURE as S,
)


@dataclass
class Scenario:
    width: int
    height: int
    n_thieves: int
    n_guardians: int
    wall_density: float
    fixed_map: np.array


_fixed_scenario_maps = {
    'test-kill': [
        [T, T, G, _],
    ],

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

    '3x3-test-2thief-1guardian': [
        [T, _, G],
        [T, _, _],
        [_, _, S],
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

    '6x6-thief-guardian': [
        [T, _, _, _, _, _],
        [_, _, _, _, _, G],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, S],
    ],

    '6x6-2xthief-2xguardian': [
        [T, _, _, _, _, G],
        [T, _, _, _, _, G],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, S],
    ],

    # TODO wall layout
    '6x6-2xthief-2xguardian-walls': [
        [T, _, W, _, _, G],
        [T, _, _, W, _, G],
        [_, _, _, _, _, _],
        [_, _, W, W, _, _],
        [_, _, _, _, _, W],
        [_, _, _, _, _, S],
    ],
}


def generate_fixed_scenario(name) -> Scenario:
    map = np.array(_fixed_scenario_maps[name])
    return Scenario(
        width=map.shape[0],
        height=map.shape[1],
        n_thieves=np.sum(map == T),
        n_guardians=np.sum(map == G),
        wall_density=(map == W).mean(),
        fixed_map=map,
    )


random_scenario_configs = {
    's': Scenario(
        width=5,
        height=5,
        n_thieves=1,
        n_guardians=1,
        wall_density=0.,
        fixed_map=None,
    ),

    'm': Scenario(
        width=8,
        height=8,
        n_thieves=2,
        n_guardians=2,
        wall_density=.0,
        fixed_map=None,
    ),

    'l': Scenario(
        width=12,
        height=12,
        n_thieves=3,
        n_guardians=3,
        wall_density=.4,
        fixed_map=None,
    )
}


def random_cell(x_range, y_range) -> (int, int):
    x = np.random.randint(*x_range)
    y = np.random.randint(*y_range)
    return x, y


def _random_empty_cell(map, quadrant_ranges, quadrant_idx) -> (int, int):
    ranges = quadrant_ranges[quadrant_idx]

    failsafe = 0
    x, y = random_cell(*ranges)
    while map[x, y] != _:
        x, y = random_cell(*ranges)
        failsafe += 1
        if failsafe == 1000:
            raise Exception(f'Could not find an open space in quadrant {quadrant_idx}')
    return x, y


def _compute_quadrants(W, H) -> [((int, int), (int, int))]:
    """
    Split the map into four equal quadrants

    Returns:
        for each of the four quadrants, an x range and a y range
    """
    H2 = H // 2
    W2 = W // 2
    top = (0, H2)
    bottom = (H2, H)
    left = (0, W2)
    right = (W2, W)
    return [
        (left, top),
        (left, bottom),
        (right, top),
        (right, bottom),
    ]


def generate_random_map(scenario_name: str) -> np.array:
    scenario = random_scenario_configs[scenario_name[-1]]
    quadrant_ranges = _compute_quadrants(scenario.width, scenario.height)

    map = np.zeros((scenario.width, scenario.height))
    map[:, :] = _

    # Place walls
    wall_mask = np.random.rand(scenario.width, scenario.height) < scenario.wall_density
    map[wall_mask] = W

    # Pick areas for the teams and treasure
    thieves_quad, guardians_quad, treasure_quad = np.random.choice(4, size=3, replace=False)

    # Place treasure
    treasure_pos = _random_empty_cell(map, quadrant_ranges, treasure_quad)
    map[treasure_pos] = S  # TODO walls don't block off objects

    # Place the thieves and guardians
    for avatar_id in range(scenario.n_thieves):
        x, y = _random_empty_cell(map, quadrant_ranges, thieves_quad)
        map[x, y] = T

    for avatar_id in range(scenario.n_thieves, scenario.n_thieves + scenario.n_guardians):
        x, y = _random_empty_cell(map, quadrant_ranges, guardians_quad)
        map[x, y] = G

    return map
