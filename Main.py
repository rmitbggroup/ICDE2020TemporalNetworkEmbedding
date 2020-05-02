import argparse
import math
import progressbar
import logging
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import more_itertools as mit
import itertools as it
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances as ed
from decimal import Decimal
import time as Time
logging.disable(logging.WARNING)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


def parse_args():
	parser = argparse.ArgumentParser(description="Run random walks.")

	parser.add_argument('--input1', nargs='?', default='shufData/digg_truncate.txt',
						help='Input training graph')

	parser.add_argument('--input2', nargs='?', default='graphwalk/digg_shuffletimewalk.txt',
						help='Ranom walks of the input training graph')

	parser.add_argument('--learning_rate', type=float, default=0.000001,
						help='learning rate.')

	parser.add_argument('--neg_sample_size', type=int, default=5,
						help='negative sample size.')

	parser.add_argument('--batchread_size', type=int, default=512,
						help='batch read size for training. The actual training size can be smaller.')

	parser.add_argument('--dimension', type=int, default=64,
						help='dimension of output. Final dimension is 2x of this due to CONCAT, and embedding dimension is also 2x of this.')

	parser.add_argument('--hiddendim', type=int, default=128,
						help='dimension of lstm model.')

	parser.add_argument('--sample_portion', type=float, default=0.6,
						help='portion of first/second layer samples for negtive samples.')

	parser.add_argument('--weight_decay', type=float, default=0.0,
						help='weight_decay to prevent overfitting. ')

	parser.add_argument('--margin', type=float, default=5.0,
						help='the safty margin size.')

	parser.add_argument('--epoch', type=int, default=10,
						help='the number of epochs for training.')

	parser.add_argument('--walk_length', type=int, default=5,
						help='walk_length of random walks. This value cannot be greater than the walk_length in WalkGenerator.py.')

	parser.add_argument('--divide', type=int, default=10,
						help='the number of walks, which should be the same as num-walks in WalkGenerator.py')

	parser.add_argument('--base_dir', nargs='?', default='embeddings/attention/digg',
						help='out put directory.')

	parser.add_argument('--clip', type=float, default=5.0,
						help='out put directory.')
	parser.add_argument('--epsilon', type=float, default=1e-08,
						help='epsilon in the AdamOptimizer.')
	return parser.parse_args()

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

class model(object):
	def __init__(self,args):
		self.epsilon=args.epsilon
		self.dimension=2*args.dimension
		self.hiddendim=args.hiddendim
		self.negsample=args.neg_sample_size
		self.sample_portion=args.sample_portion
		self.weight_decay=args.weight_decay
		self.margin=args.margin
		self.batch_num=0
		self.G=nx.MultiGraph()
		self.edges=0
		self.nodes=0
		self.history_size=0
		self.batchread_size=args.batchread_size
		self.recentTime= []
		self.recentHistory = []
		self.recentTimeInfo = []
		self.training_size=0
		self.val_size=0
		self.loss=0
		self.train_iteration=0
		self.val_iteration=0
		self.degreelist=[]
		self.neg_stsampleslist=[]
		self.neg_ensampleslist=[]
		self.divide=args.divide
		self.vars={}
		self.walk_length=args.walk_length
		self.batch_size= tf.placeholder(tf.int32, shape=(None), name='batch_size')
		self.batch_st = tf.placeholder(tf.int32, shape=(None), name='batch_st')
		self.batch_st_dup = tf.placeholder(tf.int32, shape=(None), name='batch_st_dup')
		self.batch_st_history=tf.placeholder(tf.int32, shape=(None), name='batch_st_history')
		self.batch_en=tf.placeholder(tf.int32, shape=(None), name='batch_en')
		self.batch_en_dup=tf.placeholder(tf.int32, shape=(None), name='batch_en_dup')
		self.batch_en_history=tf.placeholder(tf.int32, shape=(None), name='batch_en_history')
		self.neg_st_neigh=tf.placeholder(tf.int32, shape=(None), name='neg_st_neigh')
		self.neg_en_neigh=tf.placeholder(tf.int32, shape=(None), name='neg_en_neigh')
		self.time_st=tf.placeholder(tf.float32, shape=(None), name='time_St')
		self.time_en=tf.placeholder(tf.float32, shape=(None), name='time_en')
		self.op=None

		self.vars['hist_weights'] = self.glorot([args.hiddendim, args.dimension], name='hist_weights')
		self.vars['self_weights'] = self.glorot([self.dimension, args.dimension], name='self_weights')
		self.vars['bias'] = self.zeros([2*args.dimension], name='bias')
		self.cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(args) for _ in range(2)], state_is_tuple=True)


		self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,epsilon=self.epsilon)
		self.readGraphInfo(args.input1,args.input2,args.batchread_size,args.divide,args.walk_length)
		self.degreelist=self.getDegree()
		self.embed = tf.get_variable("node_embeddings", [self.nodes, self.dimension])
		self.build()

	def readGraphInfo(self,graph,walk,batch_size,divide,walk_length):
		with open(graph) as f:
			content=f.readlines()
			for line in content:
				strlist = line.split()
				n1 = int(strlist[0])
				n2 = int(strlist[1])
				time = float(strlist[2])+0.001
				self.G.add_edge(n1, n2)
				

		self.history_size=walk_length*divide
		self.nodes=len(self.G)
		self.edges=self.G.number_of_edges()
		
		
		self.recentTime = [0.0 for i in range(self.nodes)]
		self.recentHistory = [[] for i in range(self.nodes)]
		self.recentTimeInfo=[[] for i in range(self.nodes)]
		self.train_iteration=int(self.edges/batch_size)
		if (self.edges%batch_size>0):
			self.train_iteration+=1

	def get_a_cell(self,args):
		return tf.nn.rnn_cell.LSTMCell(args.hiddendim,state_is_tuple=True)
	def getDegree(self):
		degreelist=[]
		for i in self.G.nodes():
			degreelist.append(self.G.degree(i))
		return degreelist

	def glorot(self,shape, name=None):
		"""Glorot & Bengio (AISTATS 2010) init."""
		init_range = np.sqrt(6.0/(shape[0]+shape[1]))
		initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
		return tf.Variable(initial, name=name)

	def zeros(self,shape, name=None):
		"""All zeros."""
		initial = tf.zeros(shape, dtype=tf.float32)
		return tf.Variable(initial, name=name)
	
	# aggregation for target nodes
	def aggregate(self,batch,batch_history,hidden_dim,dim,timeinfo,act=lambda x : x):
		self_vecs = tf.nn.embedding_lookup(self.embed, batch)
		neigh_vecs = tf.nn.embedding_lookup(self.embed, batch_history)
		self_vecs_dup=tf.nn.embedding_lookup(self.embed, self.batch_st_dup)
		coefficient=(self_vecs_dup-neigh_vecs)
		coefficient=tf.square(coefficient)
		timeinfo=tf.reshape(timeinfo,[-1,1])
		coefficient=timeinfo*coefficient
		coefficient=tf.exp(coefficient)
		neigh_vecs=tf.multiply(neigh_vecs,coefficient)
		
		neigh_dims = [tf.size(batch)*self.divide,int(self.history_size/self.divide),dim]
		neigh_vecs = tf.reshape(neigh_vecs, neigh_dims)		
		dims = tf.shape(neigh_vecs)
		batch_size = dims[0]
		initial_state = self.cell.zero_state(batch_size, tf.float32)
		used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
		length = tf.reduce_sum(used, axis=1)
		length = tf.maximum(length, tf.constant(1.))
		length = tf.cast(length, tf.int32)

		rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
		self.cell, neigh_vecs,
		initial_state=initial_state, dtype=tf.float32, time_major=False,
		sequence_length=length)

		x_norm = tf.layers.batch_normalization(rnn_states[-1][1], training=True)
		x_norm = tf.nn.relu(x_norm)
		neigh_dims = [tf.size(batch),self.divide,args.hiddendim]
		neigh_vecs = tf.reshape(x_norm,neigh_dims)
		dims = tf.shape(neigh_vecs)
		batch_size = dims[0]
		initial_state = self.cell.zero_state(batch_size, tf.float32)
		used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
		length = tf.reduce_sum(used, axis=1)
		length = tf.maximum(length, tf.constant(1.))
		length = tf.cast(length, tf.int32)

		rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
		self.cell, neigh_vecs,
		initial_state=initial_state, dtype=tf.float32, time_major=False,
		sequence_length=length)

		y_norm = tf.layers.batch_normalization(rnn_states[-1][1], training=True)
		#y_norm = tf.nn.leaky_relu(y_norm,alpha=0.01)
		from_neighs = tf.matmul(y_norm, self.vars['hist_weights'])
		from_self = tf.matmul(self_vecs, self.vars["self_weights"])

		output = tf.concat([from_self, from_neighs], axis=1)
		output += self.vars['bias']
		
		return act(output)

	# aggregation for negative samples.
	def aggregate_neg(self,batch,batch_history,hidden_dim,dim,act=lambda x : x):
		self_vecs = tf.nn.embedding_lookup(self.embed, batch)
		
		neigh_vecs = tf.nn.embedding_lookup(self.embed, batch_history)

		neigh_dims = [tf.size(batch)*self.divide,int(self.history_size/self.divide),dim]
		neigh_vecs = tf.reshape(neigh_vecs, neigh_dims)		
		dims = tf.shape(neigh_vecs)
		batch_size = dims[0]
		initial_state = self.cell.zero_state(batch_size, tf.float32)
		used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
		length = tf.reduce_sum(used, axis=1)
		length = tf.maximum(length, tf.constant(1.))
		length = tf.cast(length, tf.int32)

		rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
		self.cell, neigh_vecs,
		initial_state=initial_state, dtype=tf.float32, time_major=False,
		sequence_length=length)

		x_norm = tf.layers.batch_normalization(rnn_states[-1][1], training=True)
		x_norm = tf.nn.relu(x_norm)
		neigh_dims = [tf.size(batch),self.divide,args.hiddendim]
		neigh_vecs = tf.reshape(x_norm,neigh_dims)
		dims = tf.shape(neigh_vecs)
		batch_size = dims[0]
		initial_state = self.cell.zero_state(batch_size, tf.float32)
		used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
		length = tf.reduce_sum(used, axis=1)
		length = tf.maximum(length, tf.constant(1.))
		length = tf.cast(length, tf.int32)

		rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
		self.cell, neigh_vecs,
		initial_state=initial_state, dtype=tf.float32, time_major=False,
		sequence_length=length)

		y_norm = tf.layers.batch_normalization(rnn_states[-1][1], training=True)
		#y_norm = tf.nn.leaky_relu(y_norm,alpha=0.01)
		from_neighs = tf.matmul(y_norm, self.vars['hist_weights'])
		from_self = tf.matmul(self_vecs, self.vars["self_weights"])

		output = tf.concat([from_self, from_neighs], axis=1)
		output += self.vars['bias']
		
		return act(output)

	def getNeighborhood(self,nodes):
		neighborhood=[]

		for nbr1 in nodes:
			lay1_nbrs = list(self.G.neighbors(nbr1))
			lay2_nbrs = []
			for nbr2 in lay1_nbrs:
				lay2_nbrs += list(self.G.neighbors(nbr2))
			layer1 = np.random.choice(lay1_nbrs, int(self.history_size * self.sample_portion), replace=True)
			layer2 = np.random.choice(lay2_nbrs, self.history_size-
				int(self.history_size * self.sample_portion), replace=True)
			neighborhood.extend(layer1)
			neighborhood.extend(layer2)
		return neighborhood

	def _build(self):
		labels1 = tf.reshape(tf.cast(self.batch_st, dtype=tf.int64),[self.batch_size, 1])
		labels2 = tf.reshape(tf.cast(self.batch_en, dtype=tf.int64),[self.batch_size, 1])
		self.neg_stsamples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
			true_classes=labels2,
			num_true=1,
			num_sampled=self.negsample,
			unique=False,
			range_max=len(self.degreelist),
			distortion=0.75,
			unigrams=self.degreelist))
		self.neg_ensamples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
			true_classes=labels1,
			num_true=1,
			num_sampled=self.negsample,
			unique=False,
			range_max=len(self.degreelist),
			distortion=0.75,
			unigrams=self.degreelist))


		self.stoutput=self.aggregate(self.batch_st,self.batch_st_history,self.hiddendim,self.dimension,self.time_st)
		self.enoutput=self.aggregate(self.batch_en,self.batch_en_history,self.hiddendim,self.dimension,self.time_en)

		self.neg_stoutput=self.aggregate_neg(self.neg_stsamples,self.neg_st_neigh,self.hiddendim,self.dimension)
		self.neg_enoutput=self.aggregate_neg(self.neg_ensamples,self.neg_en_neigh,self.hiddendim,self.dimension)

		self.stoutput = tf.nn.l2_normalize(self.stoutput, 1)
		self.enoutput = tf.nn.l2_normalize(self.enoutput, 1)
		self.neg_stoutput = tf.nn.l2_normalize(self.neg_stoutput, 1)
		self.neg_enoutput = tf.nn.l2_normalize(self.neg_enoutput, 1)

	def build(self):
		self._build()
		self._loss()
		#return tf.reduce_sum(tf.pow(Y1 - Y2, 2))

	def _loss(self):
		self.loss=0
		for var in self.vars.values():
			self.loss += self.weight_decay * tf.nn.l2_loss(var)

		st_negPairwise=self.pairwiseDistance(name='st')
		en_negPairwise=self.pairwiseDistance(name='en')
		st_enPairwise=tf.reduce_sum(tf.pow(self.stoutput - self.enoutput, 2),axis = -1,keepdims=True)
	
		distance1= tf.reduce_sum( tf.maximum(self.margin + (st_enPairwise- st_negPairwise),0) )
		distance2= tf.reduce_sum( tf.maximum(self.margin + (st_enPairwise- en_negPairwise),0) )

		self.loss += distance1+distance2
		self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.op=self.optimizer.minimize(self.loss)
	def pairwiseDistance(self,name):
		if name == 'st':
			x1 = self.stoutput[:,tf.newaxis,:]
			y1 = self.neg_stoutput[tf.newaxis,:,:]
		else:
			x1 = self.enoutput[:,tf.newaxis,:]
			y1 = self.neg_enoutput[tf.newaxis,:,:]
		delta =y1-x1
		distance = tf.reduce_sum(delta**2,axis = -1)
		return distance


def base_dir(args):

		base = args.base_dir
		base += "/walk_{names[0]}_lr{names[1]}_margin{names[2]}".format(names=[args.walk_length,args.learning_rate,args.margin])

		return base


def main(args):

	x=model(args)
	x.recentTime = [-0.1 for i in range(x.nodes)]
	x.recentHistory= [ [] for i in range(x.nodes)]
	x.recentTimeInfo=[ [] for i in range(x.nodes)]

	base_dirr=base_dir(args)
	
	config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
	config.gpu_options.allow_growth = True
	
	sess = tf.Session(config=config)
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())

	#the first epoch look through the whole file to load history for every node.
	#note that the running time can be greatly reduced if we store all random walks in the prepocessing step.
	for epoch in range(args.epoch):
		end=False
		f=open(args.input2)
		print ('epoch:',str(epoch+1))
		progress=progressbar.ProgressBar(maxval=x.train_iteration,widgets=['Training: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()])
		progress.start()
		iter_count=0
		#f.readline()
		train_loss=0.0
		val_loss=0.0
		#iteratively training batches
		train_starttime=Time.time()
		incre_time=0
		batch_time=0
		for train_iter in range(x.train_iteration): #
			b_time=Time.time()
			st=[]
			st_dup=[]
			st_history=[]
			time_st=[]
			en=[]
			en_dup=[]
			en_history=[]
			time_en=[]
			batch_s=Time.time()
			for iter2 in range(args.batchread_size):
				line=f.readline()
				if ('' == line):
					end=True
					break
				#print line
				edge,history = line.split(' - ')
				edge=edge.split()
				node1=int(edge[0])
				node2=int(edge[1])
				st.append(node1)
				en.append(node2)

				for i in range(args.walk_length*args.divide):
					st_dup.append(node1)
					en_dup.append(node2)

				s_history,e_history=history.split('$')
				rule=r'\d+\.\d+|\d+'
				s_history=list(map(float, re.findall(rule, s_history)))
				e_history=list(map(float, re.findall(rule, e_history)))

				tempst=[]
				tempen=[]
				tempst_time=[]
				tempen_time=[]
				s_time=s_history[0::2]
				s_history=s_history[1::2]
				s_history=[int(i) for i in s_history]

				e_time=e_history[0::1]
				e_history=e_history[1::1]
				e_history=[int(i) for i in e_history]				

				
				for walk in [list(c)[:args.walk_length] for c in mit.divide(args.divide, s_history)]:
					st_history.extend(walk)
					#print (walk)
					tempst.extend(walk)
				for walk in [list(c)[:args.walk_length] for c in mit.divide(args.divide, e_history)]:
					en_history.extend(walk)
					tempen.extend(walk)

				for walk in [list(c)[:args.walk_length] for c in mit.divide(args.divide, s_time)]:
					time_st.extend(walk)
					tempst_time.extend(walk)
				for walk in [list(c)[:args.walk_length] for c in mit.divide(args.divide, e_time)]:
					time_en.extend(walk)
					tempen_time.extend(walk)
					

				
				if epoch == 0 :
					time = float(edge[3])
					if x.recentTime[node1]<= time:
						x.recentTime[node1] = time
						x.recentHistory[node1] = tempst
						x.recentTimeInfo[node1] =tempst_time

					if x.recentTime[node2]<= time:
						x.recentTime[node2] = time
						x.recentHistory[node2] = tempen
						x.recentTimeInfo[node2] =tempen_time

			feed_dict = dict()		

			feed_dict.update({x.time_st : time_st})
			feed_dict.update({x.time_en : time_en})
			feed_dict.update({x.batch_size : len(st)})
			feed_dict.update({x.batch_st : st})
			feed_dict.update({x.batch_st_dup : st_dup})
			feed_dict.update({x.batch_en : en})
			feed_dict.update({x.batch_en_dup : en_dup})
			feed_dict.update({x.batch_st_history : st_history})
			feed_dict.update({x.batch_en_history : en_history})
			a,b = sess.run([x.neg_stsamples,x.neg_ensamples],feed_dict=feed_dict)

			neg_stsampleslist=x.getNeighborhood(a)
			neg_ensampleslist=x.getNeighborhood(b)

			#feed_dict = dict()
			feed_dict.update({x.neg_st_neigh : neg_stsampleslist})
			feed_dict.update({x.neg_en_neigh : neg_ensampleslist})

			out=sess.run([x.loss,x.op,x.stoutput],feed_dict=feed_dict)
			train_loss+=out[0]
			batch_e=Time.time()

			iter_count+=1
			progress.update(iter_count)

			if end:
				break
		progress.finish()
		train_endtime=Time.time()
		train_loss =train_loss/x.train_iteration

		log_dir = base_dirr+'.log'
		train_loss= round(Decimal(train_loss),4)
		train_time=train_endtime-train_starttime
		with open(log_dir, 'a') as ff:
			content = 'epoch: ' + str(epoch+1) +', avgTrainLoss: '+ str(train_loss)
			time = ' trainTime (hr/sec): '+ str(int(train_time/3600))+'/'+str(train_time) + 'perbatch (sec):'+str(float(train_time/(x.train_iteration+1))) +'\n'
			ff.write(content+time)

		print (content)

		#print ('saving model for epoch ',(epoch+1))
		embeddings=[]
		nodeList=sorted(list(x.G.nodes()))
		batch= [list(c) for c in mit.divide(100, nodeList)]
		for sub in batch:
			st_dup=[]
			for node in sub:
				for i in range(args.walk_length*args.divide):
					st_dup.append(node)

			
			st=sub[0]
			en=sub[-1]
			st_history=list(it.chain.from_iterable(x.recentHistory[st:(en+1)]))
			time_st=list(it.chain.from_iterable(x.recentTimeInfo[st:(en+1)]))
			feed_dict = dict()
			feed_dict.update({x.time_st : time_st})
			feed_dict.update({x.time_en : time_st})
			feed_dict.update({x.batch_size : len(sub)})
			feed_dict.update({x.batch_st : sub})
			feed_dict.update({x.batch_st_dup : st_dup})
			feed_dict.update({x.batch_en : sub})
			feed_dict.update({x.batch_en_dup : st_dup})
			feed_dict.update({x.batch_st_history : st_history})
			feed_dict.update({x.batch_en_history : st_history})
			out=sess.run([x.stoutput], feed_dict=feed_dict)
			batch_embed=out[0]
			for i in range(len(out[0])):
				embeddings.append(batch_embed[i,:])

		embeddings = np.vstack(embeddings)
		embed_dir=base_dirr+'_epoch:'+str(epoch)
		np.save(embed_dir + ".npy",  embeddings)


if __name__ == "__main__":
	args = parse_args()
	main(args)
