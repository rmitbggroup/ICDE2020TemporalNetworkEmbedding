import numpy as np
import networkx as nx
import random
import argparse
import math
import multiprocessing as mp
import more_itertools as mit
from functools import partial
import gc

def parse_args():
	parser = argparse.ArgumentParser(description="Run random walks.")

	parser.add_argument('--input', nargs='?', default='shufData/digg_truncate.txt',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='graphwalk/digg_timewalk.txt',
	                    help='output graph path')

	parser.add_argument('--walk-length', type=int, default=5,
	                    help='Length of walk per source.')

	parser.add_argument('--batch_num', type=int, default=20,
	                    help='The number of batches for loading the input graph. Please set it as a big value such as 20 to improve efficiency and avoid memory overflow.')

	parser.add_argument('--cpu', type=int, default=50,
	                    help='The number of cpus for parallel processing.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source.')

	parser.add_argument('--p', type=float, default=0.5,
	                    help='Return hyperparameter.')

	parser.add_argument('--q', type=float, default=1.0,
	                    help='Inout hyperparameter.')

	parser.add_argument('--decay', type=float, default=1.5,
	                    help='weight decay constant.')

	parser.add_argument('--portion', type=float, default=0.7,
	                    help='portion of first/second layer samples for hungry nodes which do not have neighbors based on the time constraint.')

	return parser.parse_args()

class Walker:
	def __init__(self,num_walks,walk_length,decay,q,p,batch_num):
		'''
		Reads the input network in networkx.
		'''
		self.G = None
		self.batch_num=batch_num
		self.size= 0
		self.walks=args.num_walks
		self.walk_length=args.walk_length
		self.decay = args.decay
		self.contextsize=self.walks*self.walk_length
		self.visnode=np.zeros((self.size,),dtype=np.bool)
		self.visprob=[[] for i in range(self.size)]
		self.visneigh=[[] for i in range(self.size)]
		self.visweight=[[] for i in range(self.size)]
		self.lay2_nbrs=[[] for i in range(self.size)]	
		self.q=args.q
		self.p=args.p
		self.record=[]

		#self.getLayer2neighbor()	
	def reinitialize(self):
		self.visnode=np.zeros((self.size,),dtype=np.bool)
		self.visprob=[[] for i in range(self.size)]
		self.visneigh=[[] for i in range(self.size)]
		self.lay2_nbrs=[[] for i in range(self.size)]
		self.visweight=[[] for i in range(self.size)]

	def getLayer2neighbor(self):
		for node in self.G.nodes():

			stlay1_nbrs = list(self.G.neighbors(node))
			stlay2_nbrs = []
			for nbr in stlay1_nbrs:
				stlay2_nbrs += list(self.G.neighbors(nbr))
			self.lay2_nbrs[node]=stlay2_nbrs
	def returnGraph(self):
		return self.G

	def getWeight(self,cur,nbr,time):
		culWeight=0;
		for i in range(len(self.G[cur][nbr])):
			cur_time=self.G[cur][nbr][i]['time']
			if cur_time<time:
				culWeight+= self.G[cur][nbr][i]['weight'] * math.exp(cur_time-time)
		if culWeight==0:
			return -1
		else:
			return culWeight * self.decay

	def get_walk(self,start_node,time):
		walk = [start_node]
		validwalk = []

		while len(validwalk) < args.walk_length:

				cur = walk[-1]
				cur_nbrs = self.G.neighbors(cur)
				if len(walk) == 1:
					for nbr in cur_nbrs:
						val_nbrs = []
						val_weight = []
						
						weight=self.getWeight(cur,nbr,time)
						if weight != -1:
							val_nbrs.append(nbr)
							val_weight.append(weight)
							
						
						val_index=[i for i in range(len(val_nbrs))]	
							#val_weight.append(cur_weight * decay * math.exp(cur_time-time))
					if len(val_nbrs) == 0:
						break
					norm_const = sum(val_weight)
					normalized_probs = [float(u_prob)/norm_const for u_prob in val_weight]
					index = np.random.choice(val_index, 1, normalized_probs)
					next=val_nbrs[index[0]]
					edge_weight=val_weight[index[0]]
					self.record.append(cur)
					self.visnode[cur]=True
					self.visprob[cur]=normalized_probs
					self.visneigh[cur]=val_nbrs
					self.visweight[cur]=val_weight
					walk.append(next)
					if next != start_node:

						validwalk.append(str(edge_weight)+' '+str(next)+' ')


				else:
					val_weight = []
					pre = walk[-2]
					val_nbrs=[]
					normalized_probs=[]
					if self.visnode[cur]:

						normalized_probs = self.visprob[cur]
						val_nbrs=self.visneigh[cur]
						val_weight=self.visweight[cur]
					else:
						for nbr in cur_nbrs:
							weight=self.getWeight(cur,nbr,time)
							if weight == -1:
								continue
							val_nbrs.append(nbr)
							if nbr == pre:
								#nbr is the previous node in the walk
								val_weight.append(1/self.p * weight)
							else:
								if  self.G.has_edge(nbr, pre):
									val_weight.append(weight)

								else:
									val_weight.append(1/self.q * weight)

						#if len(val_nbrs) == 0:
						#		break
						
						norm_const = sum(val_weight)
						normalized_probs = [float(u_prob)/norm_const for u_prob in val_weight]
						self.record.append(cur)
						self.visnode[cur] = True
						self.visprob[cur]=normalized_probs
						self.visneigh[cur]=val_nbrs
						self.visweight[cur]=val_weight
	
					val_index=[i for i in range(len(val_nbrs))]	
					index = np.random.choice(val_index, 1, normalized_probs)
					#print ("index:",str(index[0])," len nbrs:",str(len(val_nbrs))," len weight:",str(len(val_weight)))
					next=val_nbrs[index[0]]
					edge_weight=val_weight[index[0]]

					
					walk.append(next)
					if next != start_node:
	
						validwalk.append(str(edge_weight)+' '+str(next)+' ')
	
						


		if len(validwalk) < args.walk_length:
			diff = args.walk_length - len(validwalk)
			stlay1_nbrs = list(self.G.neighbors(start_node))
			stlay2_nbrs = []
			for nbr in stlay1_nbrs:
				stlay2_nbrs.extend(list(self.G.neighbors(nbr)))
			layer1 = np.random.choice(stlay1_nbrs, int(diff * args.portion), replace=True)
			layer2 = np.random.choice(stlay2_nbrs, diff - int(diff * args.portion), replace=True)
			for node in layer1:

				validwalk.append('1 '+str(node)+' ')
	
			for node in layer2:

				validwalk.append('1 '+str(node)+' ')

			
		return validwalk

	def getHead(self,st,en,time):
		head=[]
		head.append(str(st)+' ')
		head.append(str(en)+' ')
		head.append(str(self.contextsize)+' ')
		head.append(str(time)+' - ')
		return head

	def getSentence(self,st,en):
			content = []
			count=0
			processed=0
			for edge in self.G.edges(data=True):
				count+=1
				if (count >=st) and (count <=en):
					processed+=1
					print (mp.current_process(),'workload left:',en-st+1,'-',processed,'=',en-st+1-processed,' batch_num:',self.batch_num)
					time=edge[2]['time']
					sentence1 = []
					sentence2 = []
					sentence=self.getHead(edge[0],edge[1],time)
					for iter in range(args.num_walks):
						walk1=self.get_walk(edge[0],time)
						walk2=self.get_walk(edge[1],time)
						sentence1.extend(walk1)
						sentence1.append('; ')
						sentence2.extend(walk2)
						sentence2.append('; ')
					sentence1[-1] = '$ '
					sentence2[-1]='\n'
					
					sentence1.extend(sentence2)
					sentence.extend(sentence1)
					content.append(sentence)

					for node in self.record:
						self.visnode[node] = False
						self.visprob[node] = []
						self.visneigh[node] =[]
						self.visweight[node] =[]
					self.record=[]										

			return content
def caller(indexList,file,num_walks,walk_length,decay,q,p,batch_num):
	model=Walker(num_walks,walk_length,decay,q,p,batch_num)
	model.G = nx.MultiGraph()
	model.batch_num=batch_num
	st=indexList[0]
	en=indexList[-1]
	print (mp.current_process(),'start_index :',st,' end_index :',en)
	index=0
	for line in open(file) :
			strlist = line.split()
			n1 = int(strlist[0])
			n2 = int(strlist[1])
			time = float(strlist[2])
			model.G.add_edge(n1, n2, weight=1, time=time)

	model.size=model.G.size()
	model.reinitialize()
	content=model.getSentence(st,en)		
	return content
def main(args):

	index=0
	indexList=[]
	for line in open(args.input) :
		index+=1
		indexList.append(index)
	batch_num=args.batch_num
	batches= [list(c) for c in mit.divide(batch_num, indexList)]
	del indexList
	cpu=args.cpu
	print ('cpu count: ',cpu)
	
	truncBatches=[]
	for batch in batches:
		truncBatch=[]
		for c in mit.divide(cpu, batch):
			temp= list(c)
			truncBatch.append([temp[0],temp[-1]])
		truncBatches.append(truncBatch)
	del batches
	gc.collect()

	count=0

	for truncBatch in truncBatches:
		count+=1
		pool = mp.Pool(processes=cpu)
		prod_x=partial(caller,file=args.input,num_walks=args.num_walks,walk_length=args.walk_length,
		decay=args.decay,q=args.q,p=args.p,batch_num=count)
		res = pool.map(prod_x, truncBatch)
		pool.close()
		pool.join()
		with open(args.output, 'a') as f:
			for content in res:
				for sentence in content:
					if(len(sentence)>0):
						sentence = ''.join(sentence)
						f.write(sentence)
		del res
		gc.collect()


if __name__ == "__main__":
	args = parse_args()
	main(args)
