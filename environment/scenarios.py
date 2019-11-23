from dataclasses import dataclass
import numpy as np

from environment.thieves_guardians_env import EMPTY, WALL, THIEF, GUARDIAN, TREASURE

_ = EMPTY
W = WALL
T = THIEF
G = GUARDIAN
S = TREASURE


@dataclass
class Scenario:
    width: int
    height: int
    num_thieves: int
    num_guardians: int
    num_treasures: int
    num_walls: int
    fixed_map: np.array


_fixed_scenario_maps = {
    '4x4,1v0': [
        [T, _, _, _],
        [_, _, _, _],
        [_, _, _, _],
        [_, _, _, S],
    ],

    '4x4,1v0,2t': [
        [T, _, _, _],
        [_, _, _, _],
        [_, _, _, _],
        [S, _, _, S],
    ],

    '4x4,1v1': [
        [T, _, _, G],
        [_, _, _, _],
        [_, _, _, _],
        [_, _, _, S],
    ],

    '6x6,1v1': [
        [T, _, _, _, _, _],
        [_, _, _, _, _, G],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, S],
    ],

    '6x6,2v2': [
        [T, _, _, _, _, G],
        [T, _, _, _, _, G],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, _],
        [_, _, _, _, _, S],
    ],

    '9v9,2v3,pacman-1': [
        [T, _, _, _, W, _, _, _, S],
        [_, W, W, _, W, _, W, W, _],
        [_, W, _, _, _, _, _, W, _],
        [_, W, _, W, W, W, _, W, _],
        [_, _, _, G, G, G, _, _, _],
        [_, W, _, W, W, W, _, W, _],
        [_, W, _, _, _, _, _, W, _],
        [_, W, W, _, W, _, W, W, _],
        [S, _, _, _, W, _, _, _, T],
    ],

    '13x13,2v2,pacman-2': [
        [W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, S, _, _, _, _, W, _, _, _, _, _, S],
        [W, _, W, W, W, _, W, _, W, _, _, _, _],
        [W, _, W, _, _, _, W, _, W, W, W, W, _],
        [W, _, _, _, _, _, _, _, _, _, _, _, _],
        [W, W, W, _, W, W, W, W, W, _, W, W, W],
        [_, _, _, _, _, G, W, G, _, _, _, _, _],
        [W, W, W, _, W, _, W, _, W, _, W, W, W],
        [W, _, _, _, W, _, W, _, W, _, _, _, W],
        [W, _, W, _, W, _, _, _, W, _, _, _, W],
        [W, _, W, _, W, _, _, W, W, W, W, _, W],
        [W, T, _, _, _, _, _, _, _, _, _, T, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W],
    ],

    '3x3,2v1': [
        [T, _, G],
        [T, _, _],
        [_, _, S],
    ],
}


def generate_fixed_scenario(name) -> Scenario:
    map = np.array(_fixed_scenario_maps[name])
    return Scenario(
        width=map.shape[0],
        height=map.shape[1],
        num_thieves=np.sum(map == THIEF),
        num_guardians=np.sum(map == GUARDIAN),
        num_treasures=np.sum(map == TREASURE),
        num_walls=np.sum(map == WALL),
        fixed_map=map,
    )


random_scenario_configs = {
    's': Scenario(
        width=5,
        height=5,
        num_thieves=1,
        num_guardians=1,
        num_walls=0,
        num_treasures=1,
        fixed_map=None,
    ),

    'm': Scenario(
        width=8,
        height=8,
        num_thieves=2,
        num_guardians=2,
        num_treasures=1,
        num_walls=5,
        fixed_map=None,
    ),

    'l': Scenario(
        width=12,
        height=12,
        num_thieves=3,
        num_guardians=3,
        num_treasures=1,
        num_walls=25,
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

    map = np.full((scenario.width, scenario.height), EMPTY)

    # Place walls
    wall_xs = np.arange(scenario.width,  size=scenario.num_walls)
    wall_ys = np.arange(scenario.height, size=scenario.num_walls)
    map[wall_xs, wall_ys] = WALL
    # TODO check that a path exists between thieves/guardians/treasures

    # Pick areas for the teams and treasure
    thieves_quad, guardians_quad, treasures_quad = np.random.choice(4, size=3, replace=False)

    # Place treasures
    for _ in range(scenario.num_treasures):
        treasure_pos = _random_empty_cell(map, quadrant_ranges, treasures_quad)
        map[treasure_pos] = TREASURE

    # Place the thieves and guardians
    for _ in range(scenario.num_thieves):
        x, y = _random_empty_cell(map, quadrant_ranges, thieves_quad)
        map[x, y] = THIEF

    for _ in range(scenario.num_thieves, scenario.num_thieves + scenario.num_guardians):
        x, y = _random_empty_cell(map, quadrant_ranges, guardians_quad)
        map[x, y] = GUARDIAN

    return map
