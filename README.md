# intervening-in-coevolution

Team project for USC course CS-566 _Deep Learning_

## Installation

Requires Python 3.7. Tested on MacOS and Windows 

1. Python packages `pip install -r requirements.txt`
2. Imagemagick
 - MacOS: `brew install imagemagick`


## Development

Inspiration sources:
- PG & PPO: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- GAE: https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L53-L66

- Each team gets their own `Policy`
  - has a `Controller` 
    - has an `.actor` and `.critic`
  - defines how to update the `Controller` based on transitions
- Each avatar gets its own `RolloutsStorage`
  - stores transitions
  - computes returns
  - samples transitions


## Running

1. Run `python main.py configs/your-config.json`
   - Edit `configs/your-config.json` as you see fit   
3. Check training progress `tensorboard --logdir=outputs`
   - and videos in `outputs/<experiment name>/videos`
