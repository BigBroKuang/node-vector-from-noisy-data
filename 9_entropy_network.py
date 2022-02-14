import numpy as np
import csv
import sys
from collections import defaultdict
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

def check_completeness(sim_in):#the number of bi-directional edges
    com_cnt=0
    den=0
    for key,val in sim_in.items():
        #den+=len(val)
        for tgt in val:
            if key in sim_in[tgt]:
                com_cnt+=1
    return den,com_cnt


# model = Word2Vec.load("model/gene2vec sythetic_random_1_5000_1_10%.model")
model = Word2Vec.load("model/gene2vec Wang_5_tolerance_10.model")

gene_wv = list(model.wv.index_to_key)

sim_top=defaultdict(list) #values of top similar nodes
nodes_n = int(len(gene_wv)/2)
sim_name=defaultdict(list)#name of top similar nodes

for node_rank in range(len(gene_wv)):
    sim_origin=[(ele[0],ele[1]) for ele in model.wv.most_similar(positive=[gene_wv[node_rank]], topn=nodes_n)]
    sim_top[gene_wv[node_rank]]=[ele[1] for ele in sim_origin] #{1:[0.9,0.8,...]}
    sim_name[gene_wv[node_rank]]=[ele[0] for ele in sim_origin]#{1:[3,5,8,...]}


curla=[nodes_n*len(sim_top)]

den,cnt =check_completeness(sim_name)#return the number of bi-drectional edges
comp=[nodes_n*len(sim_top)-cnt/2]#the number of edges
compl=[1060]#[0,10,30,50,90,120,150,200,250,300]# save the [139]#
# compl=[10,30,50,80,100,120,140,160] #random network

alpha=2

for run in range(1,601):
    hill_cnt=0
    for node_rank in range(len(gene_wv)):
        sim_sum=sum(sim_top[gene_wv[node_rank]])
        normed =[i/sim_sum for i in sim_top[gene_wv[node_rank]]]
        
        Hill=0
        if alpha==1:
            Hill=1
        
        for ele in normed:
            if alpha==1:
                Hill *=ele**(-ele)   
            else:                
                Hill +=ele**(alpha)
                
        if alpha==1:
            d=int(Hill)
        else:
            d=int(Hill**(1.0/(1-alpha)))
        hill_cnt+=d #total # of edges
        sim_top[gene_wv[node_rank]]=sim_top[gene_wv[node_rank]][:d]
        sim_name[gene_wv[node_rank]]=sim_name[gene_wv[node_rank]][:d]
        
    if run in compl:
        den,cnt =check_completeness(sim_name)
        comp.append(hill_cnt-cnt/2)
        print(hill_cnt-cnt/2)#common edges
    curla.append(hill_cnt)
    
    if run in [300,400,500,550,600]:#[300,400,500,550,600,650,700,800,900,1000]:
        edgelist=[]
        for key,val in sim_name.items():
            for item in val:
                #if (item, key) not in edgelist and (key,item) not in edgelist:
                edgelist.append((item, key))
        gi = open('edgelist/edgelist_wang_entropy_'+str(run)+'.csv','w',newline='')
        cw = csv.writer(gi,delimiter=',')
        cw.writerows(edgelist)
        gi.close()      

# plt.figure(2)
# plt.plot(range(len(curla)),curla)

















