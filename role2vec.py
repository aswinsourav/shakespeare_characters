# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:02:41 2017

@author: aswin
"""
import util
import visualization as plot
import math 
import pandas as pd
import numpy as np
import tensorflow as tf

shakesphere_plays=pd.read_csv("shakesphere_plays.csv")

class Role2Vec:
    def __init__(self,play,model,batch_size=128,window_size=8,embedding_size_w=200,embedding_size_d=200,learning_rate=0.01,n_steps=150001,n_neg_samples=30):

        self.ac_speech_list,self.char_list=util.get_speech_by_character(shakesphere_plays,play)
        print(len(self.char_list))
        self.doc_ids, self.word_ids, self.count, self.dictionary, self.reverse_dictionary=util.build_doc_dataset(self.ac_speech_list)
        self.batch_size=128
        self.window_size=8
        self.vocabulary_size=len(self.count)
        self.document_size=len(self.ac_speech_list)
        self.embedding_size_w=200
        self.embedding_size_d=200
        self.learning_rate=0.01
        self.n_steps=150001
        self.n_neg_samples=30
        self.play=play
        self.model=model
        if(self.model=='pvdw'):
            self.init_pvdw()
            self.sess = tf.Session(graph=self.graph_pvdw)
        else:
            self.init_pvbow()
            self.sess = tf.Session(graph=self.graph_pvbow)
        
    
        

    
    
    def init_pvdw(self): 
        
        self.graph_pvdw=tf.Graph()
        
        with self.graph_pvdw.as_default():
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size+1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        
            word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))
        
            # embedding for documents (can be sentences or paragraph), D in paper
            doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))
        
            combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d
        
            weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
        
            biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        
            embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
            for j in range(self.window_size):
                embed_w = tf.nn.embedding_lookup(word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)
        
            embed_d = tf.nn.embedding_lookup(doc_embeddings, self.train_dataset[:, self.window_size])
        
            embed.append(embed_d)
            embed=tf.concat(embed,1)
        
            self.loss = tf.reduce_mean(
                  tf.nn.nce_loss(weights=weights,
                                 biases=biases,
                                 labels=self.train_labels,
                                 inputs=embed,
                                 num_sampled=self.n_neg_samples,
                                 num_classes=self.vocabulary_size))
        
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = word_embeddings / norm_w
        
            norm_d = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = doc_embeddings / norm_d
        
            # init op 
            self.init_op = tf.global_variables_initializer()
    
    def init_pvbow(self):
        self.graph_pvbow=tf.Graph()

        with self.graph_pvbow.as_default():
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        
        
            # embedding for documents (can be sentences or paragraph), D in paper
            doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))
        
            weights = tf.Variable(
                tf.truncated_normal([self.document_size, self.embedding_size_d],
                    stddev=1.0 / math.sqrt(self.embedding_size_d)))
        
            biases = tf.Variable(tf.zeros([self.document_size]))
        
            #embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
            embed_d = tf.nn.embedding_lookup(doc_embeddings, self.train_dataset)
        
            self.loss = tf.reduce_mean(
                  tf.nn.nce_loss(weights=weights,
                                 biases=biases,
                                 labels=self.train_labels,
                                 inputs=embed_d,
                                 num_sampled=self.n_neg_samples,
                                 num_classes=self.document_size))
        
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        
            norm_d = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = doc_embeddings / norm_d
        
            # init op 
            self.init_op = tf.global_variables_initializer()
            
            
    def build_model(self):
        
        session = self.sess
        
        # We must initialize all variables before we use them.
        session.run(self.init_op)
        print('Initialized')
        
        average_loss = 0
        for step in range(self.n_steps):
            
            if(self.model=='pvdw'):
                batch_inputs, batch_labels = util.generate_batch_pvdm(
                    self.doc_ids,self.word_ids,self.batch_size,self.window_size)
            else:
                batch_inputs, batch_labels = util.generate_batch_pvdbow(
                    self.doc_ids,self.word_ids,self.batch_size,self.window_size)                
            feed_dict = {self.train_dataset: batch_inputs, self.train_labels: batch_labels}
            
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += loss_val
            
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
            if(self.model=='pvdw'):
                word_embeddings=session.run(self.normalized_word_embeddings)
            else:
                word_embeddings=0
            doc_embeddings_pvdw=session.run(self.normalized_doc_embeddings)
        return word_embeddings,doc_embeddings_pvdw

    def plot_chars(self,doc_vec_combined):
        text_MDS=plot.compute_CS(doc_vec_combined)                    
        plot.plot_with_labels(text_MDS,self.char_list,self.play)
        
                

PLAY_LIST= ["a midsummer night's dream",'romeo and juliet']           
for i in PLAY_LIST:
    print(i)                    
    hamlet_vec_pvbow=Role2Vec(i,'pvbow',batch_size=400,window_size=5,n_neg_samples=5)
    hamlet_words,doc_embeddings_cbow=hamlet_vec_pvbow.build_model()
    
    hamlet_vec_pvdw=Role2Vec(i,'pvdw',batch_size=128,window_size=8,n_neg_samples=30)
    hamlet_words,doc_embeddings_pvdw=hamlet_vec_pvdw.build_model()
    
    doc_embeddings_combined=np.minimum.reduce([doc_embeddings_pvdw,doc_embeddings_cbow])
    hamlet_vec_pvdw.plot_chars(doc_embeddings_pvdw)


                    
                    
                    
