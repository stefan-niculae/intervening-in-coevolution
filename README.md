# intervening-in-coevolution

Team project for USC course CS-566 _Deep Learning_

[Presentation link](https://docs.google.com/presentation/d/15h6Y2-9z8D5KjHKlFErx_rsckPDSoI42xj4B3SkkwIQ)

[Blog post](https://stefan-niculae.github.io/intervening-in-coevolution)


## Installation

Requires Python 3.7. Tested on MacOS and Windows 

1. Python packages `pip install -r requirements.txt`
2. Imagemagick
   - MacOS: `brew install imagemagick`
3. Create virtualenv:  `python -m venv <path-to-env>`
   - activate it with `source <path-to-env>/bin/activate` 
3. Install python kernel on Jupyter: `python -m ipykernel install --name <kernel-name>` 


## Development

Inspiration sources:
- PG & PPO: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- GAE: https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L53-L66
- SAC: https://github.com/pranz24/pytorch-soft-actor-critic/ and https://github.com/quantumiracle/SOTA-RL-Algorithms/ (for discrete action space)


## Running
1. Generate configs: `python configs/generate_configs.py`
   - Check file to specify parameter combinations
2. Start training: `python main.py <path-to-your-config>.json`
   - The format of the config is detailed in `configs/structure.py`   
3. Check training throughout training: 
   - start Tensorboard: `tensorboard --logdir=outputs` and access  localhost:6006 
   - and videos in `outputs/<experiment name>/videos`
4. Compare model performances: run the `tuning/comparison.ipynb` notebook
   - start a Jupyter runtime: `jupyter lab` and access localhost:8888/lab
   - model checkpoints can be found in `outputs/<experiment-name>/trained_models/checkpoint-*.tar`
   - place `.tar` files in `tuning/comparison_models` (or a different directory which you can specify in the notebook)
