import argparse
import numpy as np
import networkx as nx
import gene2vec
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def parse_args():

    parser = argparse.ArgumentParser(description="Run gene2vec.")
    parser.add_argument('--input', type=str, default='data/sythetic_random_1_5000_3.csv')

    parser.add_argument('--threshold', type=int, default=1)    
    parser.add_argument('--use-cols', type=str, default=False)
    parser.add_argument('--cols', type=list, default=range(4))#[1,2,3,4,5,6,7,8,9,10] [1,2,3,4,5]

    parser.add_argument('--dimensions', type=int, default=128)#word2vec dimension, vector dimension
    parser.add_argument('--walk-length', type=int, default=80)
    parser.add_argument('--num-walks', type=int, default=2)

    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--near-node', type=int, default=2)
    
    parser.add_argument('--by-value', type=str, default=True)
    parser.add_argument('--tolerance', type=int, default=15)#percent
    
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)
    
    parser.add_argument('--iter', type=int,default=10)
    parser.add_argument('--workers', type=int, default=1)

    return parser.parse_args()


def main(args):
    G = gene2vec.gene2vec(raw_data=args.input,
                          use_cols=args.use_cols, 
                          cols=args.cols,
                          walk_length=args.walk_length,
                          near_node=args.near_node,
                          threshold=args.threshold,
                          by_value=args.by_value,
                          tolerance=args.tolerance,
                          p=args.p,
                          q=args.q)
    
    walks = G.gene_walk(num_walks=args.num_walks)
    walks = [list(map(str, walk)) for walk in walks]
    #learn_embeddings(walks)
    model = Word2Vec(walks,    #sentences/walkers
                     seed=1,
                     vector_size=args.dimensions,  #vector dimension
                     window=args.window_size,   #window size 
                     min_count=5, #nodes must present at least 3 times
                     sg=1,  #skip-gram
                     workers=args.workers) #1
    model.save("model/gene2vec sythetic_random_1_5000_3_15%.model")
    # print(G.context_dict)
    # #model1 = Word2Vec.load('text8.model')
    # print(model.wv['AGAP013742'])
    # print(model.wv.most_similar(positive=['AGAP013742'], topn=10))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    


















