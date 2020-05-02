For WalkGenerator.py

Format of the input file:

Each line of the input graph file represent a temporal edge which consists of three columns: 
node 1, node 2 and normalized temporal weight in [0,1].


Format of the output file:

Each line contains random walks for each target edge (u,v). The format is like:

u v contextSize w: walk1 of u; walk2 of u; ... $ walk1 of v; walk2 of v; ...

where contextSize=NumberOfWalk*WalkLength and w=the temporal edge weight. 
Walks generated for the same node (e.g., u) are are seperated by ';'. Walks generated for different nodes are seperated by '$'.


For Main.py

The convergence w.r.t. network reconstruction and link prediction with different edge representations is different. Thus, setting a small learning rate and find the best epoch for different tasks individually is recommended. 

Two input files are needed. One is the input file of WalkGenerator.py and another one is the output file of WalkGenerator.py.

The output will contain embedding of every epoch and a log file recording some statistic information for every epoch.

