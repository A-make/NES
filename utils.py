import numpy as np

def create_agent(framework, environment, layers):
    if framework == 'numpy':
        import agents.numpy as gents
    elif framework == 'torch':
        import agents.torch as gents
    elif framework == 'evostra':
        import agents.evostra as gents
    if environment == 'CartPole-v1':
        return gents.CartPoleAgent(layers)
    elif environment == 'MountainCar-v0':
        return gents.MountainCarAgent(layers)
    elif environment == 'Pendulum-v0':
        return gents.PendulumAgent(layers)
    elif environment == 'Acrobot-v1':
        return gents.AcrobotAgent(layers)

def sigmoid(x):
	return 1 / (1+np.exp(-x))

def maxmin(x):
	return np.clip(x,-1,1)

def relu(x):
	return np.clip(x,0,None)

def step(x):
	return x > 0

if __name__ == "__main__":
    from simulate import simulate_single, simulate_batch
    import gym

    environments = {'CartPole-v1':                  [4,1],
                    'MountainCar-v0':               [2,8,3],
                    'Pendulum-v0':                  [3,32,16,1],
                    'Acrobot-v1':                   [6,32,16,1],
                    'gym_custom:CartPoleSwingUp-v0':[1]
                    }
    envName = list(environments)[0] #env.unwrapped.spec.id
    print(envName)
    env = gym.make(envName)
    layers = environments[envName]
    framework = 'numpy'
    agent = create_agent(framework, envName, layers)