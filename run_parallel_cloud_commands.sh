# Install requirements
sudo `which python3.7` -m venv venv37
source venv37/bin/activate
sudo git clone https://github.com/stefan-niculae/intervening-in-coevolution.git
sudo `which pip` install -r intervening-in-coevolution/requirements.txt


# Run 16 processes from batch-1 configuration
cd intervening-in-coevolution
sudo `which python` run_parallel.py 16 configs/hyperparams_tuning/batch-1/
