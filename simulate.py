import numpy as np

step_counter = 0

def step_reward(env, obs):
	envName = env.unwrapped.spec.id
	if envName == 'CartPole-v1':
		return 1											# reward staying alive (balancing)
	elif envName == 'MountainCar-v0':
		reward = -1											# punish not finishing
		reward += np.abs(obs[1])*100 						# reward high speeds
		#reward += np.abs(obs[0])*10 						# reward reaching goal area
		# if obs[0] > -0.6 and obs[0] < -0.4: reward = -2 	# punish staying in start area
		return reward
	elif envName == 'Pendulum-v0':
		return 3*obs[1] 									# reward equal to height of pendulum sin(theta)
	elif envName == 'Acrobot-v1':
		pass
	elif envName == 'gym_custom:CartPoleSwingUp-v0':
		return 0.001*(4.8-np.abs(obs[0]))/4.8 - np.abs(obs[2])

def stop_condition(env, total_reward, done):
	envName = env.unwrapped.spec.id
	if envName == 'CartPole-v1':
		return done
	elif envName == 'MountainCar-v0':
		return done
	elif envName == 'Pendulum-v0':
		global step_counter 
		step_counter += 1
		if total_reward < -4000:
			step_counter = 0
			return True
		elif step_counter > 200:
			step_counter = 0
			return True
		return False
	elif envName == 'Acrobot-v1':
		pass
	elif envName == 'gym_custom:CartPoleSwingUp-v0':
		pass

'''SIMULATE'''

def render(env, agent):
	obs = env.reset()
	done = False
	while not done:
		env.render()
		action = agent.get_action(obs)
		obs, _, done, _ = env.step(action)

def simulate_single(env, agent):
	accumilated_reward = 0.0
	obs = env.reset()
	done = False
	while not done:
		action = agent.get_action(obs)
		obs, reward, done, info = env.step(action)
		#accumilated_reward += reward
		#accumilated_reward += 1
		accumilated_reward += step_reward(env, obs)
		done = stop_condition(env, accumilated_reward, done)
	return accumilated_reward

def simulate_batch(env, agent, batch_weights):
	rewards = []
	for weights in batch_weights:
		agent.set_weights(weights)
		#for t in range(RUNS_PER_INDIVIDUAL):
		rewards.append(simulate_single(env, agent))
	return np.array(rewards)