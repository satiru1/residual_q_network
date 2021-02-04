from agent import Agents, CommAgents
from worker import RolloutWorker, CommRolloutWorker
from buffer import ReplayBuffer
import os
import matplotlib.pyplot as plt
import numpy as np


class Runner:
	def __init__(self, env, args):
		self.env = env

		# communication agent
		# TODO see g2anet here
		if args.alg.find('commnet') > -1:
			# communication alg; difference mainly lies on the presence of weigths for each action
			self.agents = CommAgents(args)
			self.rolloutWorker = CommRolloutWorker(env, self.agents, args) 
		else:
			# no communication
			self.agents = Agents(args)
			self.rolloutWorker = RolloutWorker(env, self.agents, args)
			#self.evaluateWorker = RolloutWorker(self.env_evaluate, self.agents, args)
		
		# coma doesnt use replay buffer, its on policy
		if args.alg.find('coma') == -1:
			self.buffer = ReplayBuffer(args)

		self.args = args

		self.save_path = self.args.result_dir + '/' + args.alg
		if not os.path.exists(self.save_path):
		    os.makedirs(self.save_path)


	def run(self, num):
		plt.figure()
		plt.axis([0, self.args.n_epochs, 0, 100])
		win_rates = []
		episode_rewards = []
		train_steps = 0

		# train for n_epochs
		for epoch in range(self.args.n_epochs):
			if epoch % 100 == 0:
				print('Run {}, train epoch {}/{}'.format(num, epoch, self.args.n_epochs))

			# evaluate every evaluate_cycle
			if epoch % self.args.evaluate_cycle == 0:
				win_rate, episode_reward = self.evaluate(epoch_num=epoch)
				episode_rewards.append(episode_reward)
				win_rates.append(win_rate)
				#print(len(episode_rewards))

				plt.cla()
				plt.subplot(2, 1, 1)
				plt.plot(range(len(win_rates)), win_rates)
				plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
				plt.ylabel('win_rate')

				plt.subplot(2, 1, 2)
				plt.plot(range(len(episode_rewards)), episode_rewards)
				plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
				plt.ylabel('episode_rewards')

				plt.savefig(self.save_path + '/plt_{}_{}_{}ep.png'.format(num, self.args.env, self.args.n_epochs), format='png')
				np.save(self.save_path + '/episode_rewards_{}_{}_{}ep'.format(num, self.args.env, self.args.n_epochs), episode_rewards)
				np.save(self.save_path + '/win_rates_{}_{}_{}ep'.format(num, self.args.env, self.args.n_epochs), win_rate)

			episodes = []

			# run for n_episodes for each epoch, i.e, generates n_episodes (sequences of states that end with a terminal state)
			for episode_idx in range(self.args.n_episodes):
				#with open("/home/rafael/Documents/pp4_agents_log.txt", "a") as f:
				#	f.write("EPISODE\n")
				episode, _, _ = self.rolloutWorker.generate_episode(episode_idx)  # returns episode sample and sum of rewards for this episode and won bool
				episodes.append(episode)


			episode_batch = episodes[0]
			episodes.pop(0)

			# put observations of all the generated epsiodes together
			for episode in episodes:
			    for key in episode_batch.keys():
			        episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
			
			# again, coma doesnt need buffer, so wont store the episodes sampled
			if self.args.alg.find('coma') > -1:
				# added this line, because i was getting errors in coma.py when giving a tensor of bool; apparently, for vdn and qmix this is converted to a tensor of float in buffer processing
				episode_batch['terminated'] = episode_batch['terminated'].astype(float)
				
				self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
				train_steps += 1
			else:
			    self.buffer.store_episode(episode_batch)
			    for train_step in range(self.args.train_steps):
			        mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
			        self.agents.train(mini_batch, train_steps)
			        train_steps += 1

		plt.cla()
		plt.subplot(2, 1, 1)
		plt.plot(range(len(win_rates)), win_rates)
		plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
		plt.ylabel('win_rate')

		plt.subplot(2, 1, 2)
		plt.plot(range(len(episode_rewards)), episode_rewards)
		plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
		plt.ylabel('episode_rewards')

		plt.savefig(self.save_path + '/plt_{}_{}_{}ep.png'.format(num, self.args.env, self.args.n_epochs), format='png')
		np.save(self.save_path + '/episode_rewards_{}_{}_{}ep'.format(num, self.args.env, self.args.n_epochs), episode_rewards)
		np.save(self.save_path + '/win_rates_{}_{}_{}ep'.format(num, self.args.env, self.args.n_epochs), win_rate)

	def evaluate(self, epoch_num=None):
		win_counter = 0
		episode_rewards = 0
		for epoch in range(self.args.evaluate_epoch):
			# changed added this to log agents
			#with open("/home/rafael/Documents/pp4_agents_log.txt", "a") as f:
				#f.write("EPISODE\n")
			_, episode_reward, won = self.rolloutWorker.generate_episode(evaluate=True) #, epoch_num=epoch_num, eval_epoch=epoch) # changed added this
			episode_rewards += episode_reward
			if won:  # if env ended in winning state 
				win_counter += 1
			#print(win_counter, episode_reward)
		#print("tot_counter: ", win_counter)
		return win_counter / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch



