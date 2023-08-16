
# Reference RL Algorithm Implementation

## About
This repository outlines several reinforcement learning algorithms
and uses them to solve common OpenAI Gym environment tasks.

The algorithm outlined is Proximal Policy Optimization (PPO). The implementation
of Dreamer is currently not working. Feel free to contribute in case you want
to help implementing more algorithms.

## Quickstart

```sh
git clone https://github.com/Bonifatius94/rl-algos
cd rl-algos
```

### Local Deployment
```sh
python3 -m pip install virtualenv
virtualenv venv --python=python3

source venv/bin/activate

pip install -r build_requirements.txt
pip install -r runtime_requirements.txt

deactivate
```

### Run Training With Visualization
```sh
python3 train_interactive.py
```

### Docker Deployment

```sh
docker-compose build && docker-compose run \
    rlalgos-cuda python ./train_headless.py
```
