import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors
from matplotlib.patches import Patch

from environment.ThievesGuardiansEnv import (
    EMPTY as E,
    WALL as W,
    THIEF as T,
    GUARDIAN as G,
    TREASURE as S,
)


CELL_TYPES = [E, W, T, G, S]  # must increment from zero
TYPE_COLORS = {
    E: 'lightgrey',
    W: 'dimgrey',
    T: 'tab:purple',
    G: 'tab:cyan',
    S: 'tab:orange',
}
TYPE_NAMES = {
    E: 'Empty',
    W: 'Wall',
    T: 'Thief',
    G: 'Guardian',
    S: 'Treasure',
}


PLOT_SIZE = (12, 8)  # in "inches"
FPS = 1
DPI = 80


def plot_instance(args: tuple):
    ax, cmap, norm, step, map, pos2id = args

    ax.set_title(f'Step {step}')

    # Each cells shows the id of the avatar that is there
    display_map = np.full(map.shape, fill_value='')
    for pos, id in pos2id.items():
        display_map[pos] = id  # FIXME this only displays the first character

    # Color each cell
    c = ax.pcolor(map, edgecolors='white', linewidths=3, cmap=cmap, norm=norm, vmin=min(CELL_TYPES), vmax=max(CELL_TYPES), )
    c.update_scalarmappable()

    # Remove avatar id text objects at previous locations
    if not hasattr(plot_instance, 'previous_texts'):
        plot_instance.previous_texts = []
    for text in plot_instance.previous_texts:
        text.set_visible(False)

    # Draw text inside each cell
    for p, bg, text in zip(c.get_paths(), c.get_facecolors(), display_map.flatten()):
        # Black for ligther background and white for darker ones
        if np.all(bg[:3] > .5):
            text_color = 'black'
        else:
            text_color = 'white'

        x, y = p.vertices[:-2, :].mean(axis=0)
        text_obj = ax.text(x, y, text, ha='center', va='center', color=text_color)
        plot_instance.previous_texts.append(text_obj)


def create_animation(maps, pos2ids, save_path):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    fig.set_size_inches(PLOT_SIZE)

    # Setup sizes
    plt.rcParams.update({'font.size': 25})  # due to a bug this needs to be ran twice

    w, h = maps[0].shape
    # Place the ticks at the middle of each cell
    ax.set_yticks(np.arange(w) + .5, minor=False)
    ax.set_xticks(np.arange(h) + .5, minor=False)
    # Set tick labels to cell row/column
    ax.set_xticklabels(range(w), minor=False)
    ax.set_yticklabels(range(h), minor=False)
    # Hide all the ticks (but not tick labels)
    ax.tick_params(axis='both', which='both', length=0)
    # Remove all spines
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_visible(False)

    # Turn everything upside down because in a coordinate system (0, 0) is bottom left
    ax.invert_yaxis()

    # Show legend entries for each cell type
    legend_elements = [
        Patch(facecolor=color, label=TYPE_NAMES[t])
        for t, color in TYPE_COLORS.items()
    ]
    ax.legend(
        handles=legend_elements,
        title='Cell types',

        # Place legend box outside the plot
        loc='lower left',  # anchor position
        bbox_to_anchor=(1, 0),
    )

    # Setup colors
    cmap = matplotlib.colors.ListedColormap([TYPE_COLORS[t] for t in CELL_TYPES])
    bounds = np.array(CELL_TYPES) - .5  # for n colors there are n+1 bounds
    norm = matplotlib.colors.BoundaryNorm([*bounds, max(CELL_TYPES) + .5], len(CELL_TYPES))

    # Create the animation
    args = [(ax, cmap, norm, i, maps[i], pos2ids[i]) for i in range(len(maps))]
    animation = FuncAnimation(fig, plot_instance, args)

    # Save the gif
    animation.save(save_path, 'imagemagick', FPS, DPI)


def test():
    # Test inputs
    maps = [
        np.array([
            [T, E, E, E, E],
            [W, E, E, E, E],
            [W, E, E, W, G],
            [W, E, E, W, W],
            [G, E, E, E, S],
        ]),

        np.array([
            [E, T, E, E, E],
            [W, E, E, E, G],
            [W, E, E, W, E],
            [W, E, E, W, W],
            [G, E, E, E, S],
        ]),
    ]

    pos2ids = [
        {
            (0, 0): 0,
            (2, 4): 1,
            (4, 0): 2,
        },

        {
            (0, 1): 0,
            (1, 4): 1,
            (4, 0): 2,
        },
    ]

    create_animation(maps, pos2ids, 'test-animation.gif')


if __name__ == '__main__':
    test()
