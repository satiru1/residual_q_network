import torch
import os
from base_net import RNN
from qmix_net import QMixNet
from commnet import CommNet
import torch.nn as nn

class QMix:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		input_shape = self.obs_shape

		# input dimension for rnn according to the params
		if args.last_action:
		    input_shape += self.n_actions
		if args.reuse_network:
		    input_shape += self.n_agents


		# changed this if/else block
		if args.alg == 'qmix':
			self.eval_rnn = RNN(input_shape, args)  # each agent picks a net of actions
			self.target_rnn = RNN(input_shape, args)
			print("QMIX initialized")
		elif args.alg == 'qmix+commnet':
			self.eval_comm = CommNet(input_shape, args) # communication network t be used in get_action_weights in agent.py
			self.target_comm = CommNet(input_shape, args)
			self.eval_rnn = RNN(input_shape, args)  # each agent picks a net of actions
			self.target_rnn = RNN(input_shape, args)
			#changed this
			self.criterion = nn.CrossEntropyLoss()
			print("QMIX+COMMNET initialized")
		else:
			raise Exception("No such algorithm")

		self.eval_qmix_net = QMixNet(args)  # netowrk that mixes up agents Q values 
		self.target_qmix_net = QMixNet(args)  # target network, as in DQN
		self.args = args

		self.model_dir = args.model_dir + '/' + args.alg

		if self.args.load_model:
		    if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
		        path_rnn = self.model_dir + '/rnn_net_params.pkl'
		        path_qmix = self.model_dir + '/qmix_net_params.pkl'
		        self.eval_rnn.load_state_dict(torch.load(path_rnn))
		        self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
		        print('Successfully loaded the model: {} and {}'.format(path_rnn, path_qmix))
		    else:
		    	raise Exception("No such model!")


		# make parameters of target and eval the same
		self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
		self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

		if args.alg == 'qmix+commnet':
			#changed added this line
			self.target_comm.load_state_dict(self.eval_comm.state_dict())


		self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())

		# changed adde this
		if args.alg == 'qmix+commnet':
			self.eval_comm_parameters = list(self.eval_comm.parameters())

		if args.optimizer == "RMS":
		    self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
		    #changed added this
		    if args.alg == 'qmix+commnet':
		    	self.comm_optimizer = torch.optim.RMSprop(self.eval_comm_parameters, lr=args.lr)

		# during learning one should keep an eval_hidden and a target_hidden for each agent of each episode
		self.eval_hidden = None
		self.target_hidden = None
		#chnaged: added this line
		self.eval_comm_hidden = None
		self.target_comm_hidden = None

	def learn(self, batch, max_episode_len, train_step, epsilon=None):
		'''
			batch: batch with episode batches from before to train the model
			max_episode_len: len of the longest episode batch in batch
			train_step: it is used to control and update the params of the target network

			------------------------------------------------------------------------------

			the extracted data is 4D, with meanings 1-> n_episodes, 2-> n_transitions in the episode, 
			3-> data of multiple agents, 4-> obs dimensions
			hidden_state is related to the previous experience (RNN ?) so one cant randomly extract
			experience to learn, so multiple episodes are extracted at a time and then given to the
			nn one at a time   
		'''

		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		self.init_hidden(episode_num)

		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		state, state_next, actions, reward, avail_actions, avail_actions_next, terminated = batch['state'], batch['state_next'], \
																							batch['actions'], batch['reward'], \
																							batch['avail_actions'], batch['avail_actions_next'], \
																							batch['terminated']

		# used to set the td error of the filled experiments to 0, not to affect learning
		mask = 1 - batch["padded"].float()  

		# gets q value corresponding to each agent, dimensions are (episode_number, max_episode_len, n_agents, n_actions)
		q_evals, q_targets = self.get_q_values(batch, max_episode_len)

		#print(q_evals.shape, actions.shape)
		# get q value corresponding to each agent and remove last dim (3)
		q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)

		#print(q_evals)
		#print("-----")
		#print(weights)
		#if self.args.alg == 'qmix+commnet':
			# changed added this
			#q_evals = q_evals + weights

		#changed added this block
		#scalar_prod = 0
		#if self.args.alg == 'qmix+commnet':
			#q_evals_aux = q_evals.reshape(-1)
			#weights_aux = weights.reshape(-1)
			#scalar_prod = torch.dot(q_evals_aux, weights_aux)

		# get q value corresponding to each agent and remove last dim (3)
		#q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)

		# get real q_target
		# unavailable actions dont matter, low value
		q_targets[avail_actions_next == 0.0] = - 9999999
		q_targets = q_targets.max(dim=3)[0]

		# mixes values with qmix
		q_total_eval = self.eval_qmix_net(q_evals, state)
		q_total_target = self.target_qmix_net(q_targets, state_next)

		# gets the weights for the actions of each agent in communication changed
		# trains weights net here
		if self.args.alg == 'qmix+commnet':
			weights = self.train_weights(batch, max_episode_len, q_total_eval, q_total_target)  # changed added two last arguments so i can use the same loss for the weights
			#weights = self.train_weights(batch, max_episode_len)
			#weights = self._get_action_prob(batch, max_episode_len, self.args.epsilon)

		targets = reward + self.args.gamma * q_total_target * (1 - terminated)

		td_error = (q_total_eval - targets.detach())
		masked_td_error = mask * td_error 

		# there are still useless experiments, so the avg is according the number of real experiments
		#changed added this block
		if self.args.alg == 'qmix+commnet':
			loss = (masked_td_error ** 2).sum() / mask.sum()
		else:	
			loss = (masked_td_error ** 2).sum() / mask.sum()
		#changed added this line
		#loss_comm = loss

		self.optimizer.zero_grad()
		# changed added retain_graph arg
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
		self.optimizer.step()

		# changed added this
		'''if self.args.alg == 'qmix+commnet':
			self.comm_optimizer.zero_grad()
			loss_comm.backward()
			torch.nn.utils.clip_grad_norm_(self.eval_comm_parameters, self.args.grad_norm_clip)
			self.comm_optimizer.step()'''		


		# update target networks
		if train_step > 0 and train_step % self.args.target_update_cycle == 0:
		    self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
		    self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
		    self.target_comm.load_state_dict(self.eval_comm.state_dict())


	# changed added this function; changed the loss to get q_total values instead of the weights so we can train the nets to minimize the same loss
	def train_weights(self, batch, max_episode_len, q_total_eval, q_total_target):
		actions, avail_actions_next, reward, terminated = batch['actions'], batch['avail_actions_next'], batch['reward'], batch['terminated']

		mask = 1 - batch["padded"].float()  
		weights, weights_target = self.get_weights(batch, max_episode_len)

		weights = torch.gather(weights, dim=3, index=actions).squeeze(3)

		#weights_target[avail_actions_next == 0.0] = - 9999999
		weights_target = weights_target.max(dim=3)[0]
		
		targets = reward + self.args.gamma * weights_target * (1 - terminated)  # changed replaced weghts_target with q_total_target

		td_error = (weights - targets.detach())  # changed replaced weights with q_total_eval
		masked_td_error = mask * td_error 

		# there are still useless experiments, so the avg is according the number of real experiments
		loss = (masked_td_error ** 2).sum() / mask.sum()
		#changed added this line
		#loss_comm = loss

		self.comm_optimizer.zero_grad()
		loss.backward(retain_graph=True)
		torch.nn.utils.clip_grad_norm_(self.eval_comm_parameters, self.args.grad_norm_clip)
		self.comm_optimizer.step()	

		return weights


	# changed: added this function
	# this should be used to get the weights for commnet
	def _get_action_prob(self, batch, max_episode_len, epsilon):
		episode_num = batch['obs'].shape[0]
		avail_actions = batch['avail_actions']  # coma doesnt need the target actor
		action_prob = []
		for transition_idx in range(max_episode_len):
			inputs = self._get_actor_inputs(batch, transition_idx)

			outputs, self.eval_comm_hidden = self.eval_comm(inputs, self.eval_comm_hidden)
			outputs = outputs.view(episode_num, self.n_agents, -1)
			prob = torch.nn.functional.softmax(outputs, dim=-1)
			action_prob.append(prob)

		action_prob = torch.stack(action_prob, dim=1).cpu()
		
		action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])  # number of actions that can be selected
		action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
		action_prob[avail_actions == 0] = 0.0

		# regularize probability of actions that cant be performed
		action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)

		# set to 0 again to avoid errors
		action_prob[avail_actions == 0] = 0.0

		return action_prob

	# changed added this function
	def _get_actor_inputs(self, batch, transition_idx):
		# decentralised actor, decentralised execution; actor -> policy, maps states to actions
		# take the experience of the transition_idx on all the episodes
		obs, actions_onehot = batch['obs'][:, transition_idx], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs = []
		inputs.append(obs)

		if self.args.last_action:
			if transition_idx == 0:
				inputs.append(torch.zeros_like(actions_onehot[:, transition_idx]))
			else:
				inputs.append(actions_onehot[:, transition_idx - 1])
		if self.args.reuse_network:
			# same as above
			inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

		inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)

		return inputs


	def get_q_values(self, batch, max_episode_len):
		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		q_evals, q_targets = [], []
		for transition_idx in range(max_episode_len):
		    inputs, inputs_next = self._get_inputs(batch, transition_idx)  # add last action and agent_id to the obs
		    
		    q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # The input dimension is (40,96), and the resulting q_eval dimension is (40,n_actions)
		    q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

		    # Change the q_eval dimension back to (8, 5(n_agents), n_actions)
		    q_eval = q_eval.view(episode_num, self.n_agents, -1)
		    q_target = q_target.view(episode_num, self.n_agents, -1)
		    q_evals.append(q_eval)
		    q_targets.append(q_target)

		'''
		q_eval and q_target are lists containing max_episode_len arrays with dimensions (episode_number, n_agents, n_actions)
		convert the lists into arrays of (episode_number, max_episode_len, n_agents, n_actions)
		'''
		#print(np.shape(q_evals))
		#print(q_evals)

		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)
		return q_evals, q_targets

	# changed added this function
	def get_weights(self, batch, max_episode_len):
		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		q_evals, q_targets = [], []
		for transition_idx in range(max_episode_len):
		    inputs, inputs_next = self._get_inputs(batch, transition_idx)  # add last action and agent_id to the obs
		    
		    q_eval, self.eval_comm_hidden = self.eval_comm(inputs, self.eval_comm_hidden)  # The input dimension is (40,96), and the resulting q_eval dimension is (40,n_actions)
		    q_target, self.target_comm_hidden = self.target_comm(inputs_next, self.target_comm_hidden)

		    # Change the q_eval dimension back to (8, 5(n_agents), n_actions)
		    q_eval = q_eval.view(episode_num, self.n_agents, -1)
		    q_target = q_target.view(episode_num, self.n_agents, -1)
		    q_evals.append(q_eval)
		    q_targets.append(q_target)

		'''
		q_eval and q_target are lists containing max_episode_len arrays with dimensions (episode_number, n_agents, n_actions)
		convert the lists into arrays of (episode_number, max_episode_len, n_agents, n_actions)
		'''
		#print(np.shape(q_evals))
		#print(q_evals)

		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)
		return q_evals, q_targets


	def _get_inputs(self, batch, transition_idx):
		# gets the experience of the transition_idx on all episodes, actions_onehot take all out because the last one is used TODO ?
		obs, obs_next, actions_onehot = batch['obs'][:, transition_idx], \
		                          batch['obs_next'][:, transition_idx], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs, inputs_next = [], []
		inputs.append(obs)
		inputs_next.append(obs_next)


		# adds last action and agent number to obs
		if self.args.last_action:
		    if transition_idx == 0:  # if it is the first transition, let the previous action be a 0 vector
		        inputs.append(torch.zeros_like(actions_onehot[:, transition_idx]))
		    else:
		        inputs.append(actions_onehot[:, transition_idx - 1])
		    inputs_next.append(actions_onehot[:, transition_idx])

		if self.args.reuse_network:  # uses one network for all agents TODO: see differences
			
			'''
			TODO: read and refrase
			Because the current obs 3D data, each dimension 
			represents (episode number, agent number, obs dimension), add the corresponding vector directly on dim_1
			That is, for example, adding (1, 0, 0, 0, 0) to agent_0 means the number 0 in 5 agents. 
			And the data of agent_0 happens to be in the 0th row, then you need to add 
			The agent number happens to be an identity matrix, that is, the diagonal is 1, and the rest are 0
			'''

			inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
			inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

		'''
		TODO: read and refrase
		It is necessary to put three of the obs together, and the data of episode_num episodes 
		and self.args.n_agents agents are combined into 40 (40,96) data.

		Because all agents here share a neural network, each data is brought 
		with its own number, so it is still its own data
		'''

		inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
		inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)

		# TODO note from github: Check if inputs_next is equivalent to moving inputs backward
		return inputs, inputs_next


	def init_hidden(self, episode_num):
		# initializes eval_hidden and target_hidden for each agent of each episode, as in DQN there is a net and a target net to stabilize learning

		self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
		self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

		# changed added this if clause
		if self.args.alg == 'qmix+commnet':
			self.eval_comm_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
			self.target_comm_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))


	def save_model(self, train_step):
		num = str(train_step // self.args.save_cycle)
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
		torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')



