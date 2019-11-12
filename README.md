# intervening-in-coevolution

Team project for USC course CS-566 _Deep Learning_

## Installation

Requires Python 3.7

1. Python packages `pip install -r requirements.txt`
2. Imagemagick
 - MacOS: `brew install imagemagick`


## Development

If your model uses convolutions, [torch.nn.Conv2d](https://pytorch.org/docs/stable/nn.html#conv2d) expects channels first.

Agent algorithms source: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


## Running

1. Run `python main.py configs/your-config.json`
   - Edit `configs/your-config.json` as you see fit   
3. Check training progress `tensorboard --logdir=outputs`
   - and videos in `outputs/<experiment name>/videos`
