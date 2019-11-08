import glob
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Patch
import matplotlib
import numpy as np
import gym
from IPython import display as ipythondisplay
from IPython.display import HTML
from gym.wrappers import Monitor


class EnvVisualizationWrapper(gym.Wrapper):

    def render(self, *args, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'matplot'
        if mode != 'matplot' and mode != 'rgb_array':
            self.unwrapped.render()
            return
        # Definitions
        E = 0  # empty
        W = 1  # wall
        T = 2  # thief
        G = 3  # guardian
        S = 4  # treasure

        cell_types = [E, W, T, G, S]  # must increment from zero
        type_colors = {
            E: 'lightgrey',
            W: 'dimgrey',
            T: 'tab:purple',
            G: 'tab:cyan',
            S: 'tab:orange',
        }

        type_names = {
            E: 'Empty',
            W: 'Wall',
            T: 'Thief',
            G: 'Guardian',
            S: 'Treasure',
        }

        # Configuration
        step = self.unwrapped.elapsed_time
        w = self.unwrapped.width  # width
        h = self.unwrapped.height  # height
        map = self.unwrapped.map
        id2team = self.unwrapped.id2team
        pos2id = self.unwrapped.pos2id

        # Each cells shows the id of the avatar that is there
        display_map = np.full(map.shape, fill_value='')
        for pos, id in pos2id.items():
            display_map[pos] = id  # FIXME this only displays the first character

        # Setup colors
        cmap = matplotlib.colors.ListedColormap([type_colors[t] for t in cell_types])
        bounds = np.array(cell_types) - .5  # for n colors there are n+1 bounds
        norm = matplotlib.colors.BoundaryNorm([*bounds, max(cell_types) + .5], len(cell_types))
        # Setup sizes
        plt.rcParams.update({'font.size': 25})  # due to a bug this needs to be ran twice
        # Show legend entries for each cell type
        legend_elements = [
            Patch(facecolor=color, label=type_names[t])
            for t, color in type_colors.items()
        ]
        # Setup plot

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.set_title(f'Step {step}')

        # Color each cell
        c = ax.pcolor(
            map,
            edgecolors='white',
            linewidths=3,
            cmap=cmap,
            norm=norm,
            vmin=min(cell_types),
            vmax=max(cell_types),
        )

        # Draw text inside each cell
        c.update_scalarmappable()
        for p, bg, text in zip(c.get_paths(), c.get_facecolors(), display_map.flatten()):
            # Black for ligther background and white for darker ones
            if np.all(bg[:3] > .5):
                text_color = 'black'
            else:
                text_color = 'white'

            x, y = p.vertices[:-2, :].mean(axis=0)
            ax.text(x, y, text, ha='center', va='center', color=text_color)

        # Place the ticks at the middle of each cell
        ax.set_yticks(np.arange(w) + .5, minor=False)
        ax.set_xticks(np.arange(h) + .5, minor=False)

        # Set tick labels to cell row/column
        ax.set_xticklabels(range(w), minor=False)
        ax.set_yticklabels(range(h), minor=False)

        # Hide all the ticks (but not tick labels)
        ax.tick_params(axis='both', which='both', length=0)

        # Turn everything upside down because in a coordinate system (0, 0) is bottom left
        ax.invert_yaxis()

        # Remove all spines
        for loc in ['top', 'bottom', 'left', 'right']:
            ax.spines[loc].set_visible(False)

        # Show the legend
        ax.legend(
            handles=legend_elements,
            title='Cell types',

            # Place legend box outside the plot
            loc='lower left',  # anchor position
            bbox_to_anchor=(1, 0),
        )
        if mode == 'rgb_array':
            fig.canvas.draw()
            return np.array(fig.canvas.renderer.buffer_rgba())
        return

def wrap_env_video(env):
  env = MultiAgent_VideoMonitor(env, './video', force=True)
  return env

def gen_wrapped_env():
    return wrap_env_video(make_env("ThievesGuardiansEnv",0,0,None,False)())

class MultiAgent_VideoMonitor(Monitor):
    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done
        # if done and self.env_semantics_autoreset:
        #    # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
        #    self.reset_video_recorder()
        #    self.episode_id += 1
        #    self._flush()

        # Record stat
        # self.stats_recorder.after_step(observation, sum(reward), done, info)
        # Record video

        #TODO: I have used multipled capture_frame() to reduce the frame rate of the video, but make this more expensive (in a constant amount)
        self.video_recorder.capture_frame()
        self.video_recorder.capture_frame()
        self.video_recorder.capture_frame()
        self.video_recorder.capture_frame()
        self.video_recorder.capture_frame()
        self.video_recorder.capture_frame()
        return done


# This function plots videos of rollouts (episodes) of a given policy and environment
def log_policy_rollout(policy, pytorch_policy=False):
    # Create environment with flat observation
    env = gen_wrapped_env()

    # Initialize environment
    observation = env.reset()

    done = False
    episode_reward = 0
    episode_length = 0

    # Run until done == True
    while not done:
        # Take a step
        # if pytorch_policy:
        #    observation = torch.tensor(observation, dtype=torch.float32)
        #    action = policy.act(observation)[0].data.cpu().numpy()
        # else:
        #    action = policy.act(observation)[0]
        action = policy.act(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1

    print('Total reward:', episode_reward)
    print('Total length:', episode_length)
    env.close()
    show_video()


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")