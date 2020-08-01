# Neural agents for OpenAI gym

This repo consists of multiple implementations of neural agents that work with some of the [OpenAI Gym classical control](https://gym.openai.com/envs/#classic_control) environments. The agents have primarily been developed for use with the NES training algorithm (see below).

Each agent has the following methods:
```python
set_weights()
get_action()
```

### Gym environments
- The [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem is a classical underactuated control example. The task is to balance a pole connected to a cart such that it stays upright. The system is underactuated since we cannot manipulate the pole angle directly, but instead we must move the cart in order to effect the pendulum.
- The [MountainCar](https://gym.openai.com/envs/MountainCar-v0/) problem is a task where we have an underactuated car on a hill and would like the car to get over the hill. For this we need to build up momentum.
- Pendulum
- Acrobot

Resources
- https://gym.openai.com/envs/#classic_control
- https://github.com/openai/gym/wiki


### Install
Install: `pip install -r requirements.txt`

Tested with Python 3.8.2

# Natural Evolution Strategies (NES)

**Natural Evolution Strategies (NES)** is an evolutionary inspired optimization algorithm that can be used for adjusting the parameters of an ANN. It works by creating a fixed standard deviation Gaussian distribution around some _current_ parameter set, and testing the fitness of a set of parameters within this distribution. The _current_ parameters are updated each iteration. The optimization process is similar to hill climbing, but uses instead a population of agents where each agent represents a single parameter vector in the set.

### Background Reading

- [A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)
- [Evolving Stable Strategies](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/)
- [OpenAI: Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://openai.com/blog/evolution-strategies/)

### Inspiration

- [JorgeCeja/evolution-strategies](https://github.com/JorgeCeja/evolution-strategies)
- [karpathy/nes.py](https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d)  
- [flyman3046/es-CartPole.py](https://gist.github.com/flyman3046/d37680eeaac469a4030c690ae65b0419)
- [hardmaru/estools](https://github.com/hardmaru/estool)

### Different implementations in the NES Jupyter notebook

- numpy (no framework)
- [alirezamika/evostra](https://github.com/alirezamika/evostra)
- pyTorch
- [uber-research/EvoGrad](https://github.com/uber-research/EvoGrad) (uses pyTorch)

---

## Uber Research EvoGrad package for NES

The EvoGrad package uses PyTorch to create networks and performs NES to optimize the networks parameters/weights.

See
- [Introducing EvoGrad: A Lightweight Library for Gradient-Based Evolution](https://eng.uber.com/evograd/)
- [uber-research/EvoGrad](https://github.com/uber-research/EvoGrad)

Standalone install:
```bash
pip install evograd
```