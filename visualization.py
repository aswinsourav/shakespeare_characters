# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:39:35 2017

@author: aswin
"""

#from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS

def compute_CS(doc_embeddings_combined):
#plot_with_labels(low_dim_embs, labels)
    CS_dist = 1-cosine_similarity(doc_embeddings_combined)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=99)
    text_MDS = mds.fit_transform(CS_dist)
    return text_MDS

def plot_with_labels(low_dim_embs, labels, filename='Combined.png'):
    #assert low_dim_embs.shape[0] > len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')     
    plt.savefig(filename)