# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:58:16 2017

@author: aswin
"""
import urllib
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from itertools import compress
import collections
from nltk.corpus import stopwords

StopWords = stopwords.words("english") 

def extract_plays():
    page = urllib.request.urlopen('http://shakespeare.mit.edu').read()
    soup = BeautifulSoup(page,"html.parser")
    
    all_work = soup.find_all("a")
    
    shakesphere_plays = pd.DataFrame(columns=['ACT','SCENE','CHARACTER','SPEECH', 'PLAY'])
    for j in range(2,39):
        play = all_work[j].get_text().strip()
        url = 'http://shakespeare.mit.edu/' + all_work[j]['href'].replace("index", "full") 
        
        play_url = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(play_url,'html.parser')
        
        finds = soup.find_all(["h3","a","blockquote"])
        
        df = []  
        for i in range(2,len(finds)):
            if(finds[i].name == "h3"):
                if(finds[i].get_text()[0:3] == 'ACT'):
                    act = finds[i].get_text()
                elif(finds[i].get_text()[0:5]=='SCENE'):
                    scene = finds[i].get_text()
            elif(finds[i].attrs != {} and finds[i].attrs["name"][0:6] == "speech"):
                character=finds[i].get_text()
            elif(finds[i].name == "a"):
                entry_dict = {}
                entry_dict['ACT'] = act
                entry_dict['SCENE'] = scene
                entry_dict['CHARACTER'] = character
                entry_dict['SPEECH'] = finds[i].get_text()
                df.append(entry_dict)
            
        df = pd.DataFrame(df)
        df['PLAY'] = play
        
        shakesphere_plays = shakesphere_plays.append(df)
    return shakesphere_plays

if(not os.path.exists("shakesphere_plays.csv")):
    shakesphere_plays=extract_plays()  
    shakesphere_plays=shakesphere_plays.apply(lambda x: x.astype(str).str.lower())
    print(shakesphere_plays.shape)
    shakesphere_plays.to_csv("shakesphere_plays.csv") 


def get_speech_by_character(shakesphere_plays,play,filter_stopwords=False):
    ac_speech_list=char_list=[]
    speech_by_char=pd.DataFrame(shakesphere_plays[shakesphere_plays.PLAY==play].groupby(['PLAY','CHARACTER'],as_index=False)['SPEECH'].apply(lambda x:' '.join(x)).reset_index())
    print(speech_by_char.columns)
    speech_by_char['SPEECH']=speech_by_char.SPEECH.str.replace("[^a-zA-Z]", " ").str.strip() 
    
    if(filter_stopwords):
        for speech in list(speech_by_char['SPEECH']):
              ac_speech_list+=[[word for word in speech.split() if word not in StopWords]]
    else:
        for speech in list(speech_by_char['SPEECH']):
              ac_speech_list+=[[word for word in speech.split()]]   
                                
    char_list=list(speech_by_char['CHARACTER'])
    return ac_speech_list,char_list
    



def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary      


def build_doc_dataset(docs, vocabulary_size=50000):
    count = [['UNK', -1]]
    # words = reduce(lambda x,y: x+y, docs)
    words = []
    doc_ids = [] # collect document(sentence) indices
    for i, doc in enumerate(docs):
        doc_ids.extend([i] * len(doc))
        words.extend(doc)

    word_ids, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)

    return doc_ids, word_ids, count, dictionary, reverse_dictionary    



data_index = 0

def generate_batch_pvdm(doc_ids, word_ids, batch_size, window_size):
    global data_index
    assert batch_size % window_size == 0
    batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = window_size + 1
    buffer = collections.deque(maxlen=span) # used for collecting word_ids[data_index] in the sliding window
    buffer_doc = collections.deque(maxlen=span) # collecting id of documents in the sliding window
    # collect the first window of words
    for _ in range(span):
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    mask = [1] * span
    mask[-1] = 0 
    i = 0
    while i < batch_size:
        if len(set(buffer_doc)) == 1:
            doc_id = buffer_doc[-1]
            # all leading words and the doc_id
            batch[i, :] = list(compress(buffer, mask)) + [doc_id]
            labels[i, 0] = buffer[-1] # the last word at end of the sliding window
            i += 1
        # move the sliding window  
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    return batch, labels      
    

data_index = 0

def generate_batch_pvdbow(doc_ids, word_ids, batch_size, window_size):
    global data_index
    assert batch_size % window_size == 0
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    buffer = collections.deque(maxlen=window_size) # used for collecting word_ids[data_index] in the sliding window
    buffer_doc = collections.deque(maxlen=window_size) # collecting id of documents in the sliding window
    # collect the first window of words
    for _ in range(window_size):
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    i = 0
    while i < batch_size:
        if len(set(buffer_doc)) == 1:
                doc_id = buffer_doc[-1]
                batch[i] = doc_id
                labels[i, :] = buffer[window_size//2] # the last word at end of the sliding window
                i += 1
        # move the sliding window  
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    return batch, labels            