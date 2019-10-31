# intervening-in-coevolution

Team project for USC course CS-566 _Deep Learning_

## Installation

Tested on Python 3.6

1. MPI and CMake (from [baselines](https://github.com/openai/baselines))
    - MacOS: `brew install cmake openmpi`
    - Ubuntu: `sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`
2. Install Baselines: pip install git+https://github.com/openai/baselines.git (tested on 0.1.6)
    - the package available in pip (`pip install baselines`) requires a Mujoco license
2. Python packages `pip install -r requirements.txt`
    - likely has some unused packages in there as well
