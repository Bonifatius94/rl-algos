
# Reference RL Algorithm Implementation

## About
This repository outlines several reinforcement learning algorithms
and uses them to solve common OpenAI Gym environment tasks.

## Quickstart

```sh
git clone https://github.com/Bonifatius94/rl-algos
cd rl-algos
```

### Local Deployment

```sh
python3 -m pip install -r requirements.txt
```

```sh
python3 algos/ppo.py
```

### Docker Deployment

```sh
docker-compose build && docker-compose run \
    rlalgos-cuda python ./algos/ppo.py
```
