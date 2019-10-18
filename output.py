#!/usr/bin/env python
# coding: utf-8

# In[32]:


import nltk as n
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score


# In[33]:


def sentence_to_wordlist(input_sentence):
    
    clean_sentence = re.sub("[^a-zA-Z0-9]"," ", input_sentence)
    tokens = n.word_tokenize(clean_sentence)
    
    return tokens


# In[34]:


#stop word remove
from nltk.corpus import stopwords
def remove_stop_word(word_list):
   
    stop_words = set(stopwords.words('english')) 
    
    word_tokens = n.word_tokenize(word_list) 
  
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence


# In[35]:


# calculating average word length in an essay

def avg_word_len(essay):
    
    clean_essay = re.sub(r'\W', ' ', essay)
   # print()
    words = n.word_tokenize(clean_essay)
    
    return sum(len(word) for word in words) / len(words)


# In[36]:


# calculating number of sentences in an essay

def sent_count(essay):
    
    sentences = n.sent_tokenize(essay)
    
    return len(sentences)


# In[37]:


# calculating number of words in an essay

def word_count(essay):
    
    clean_essay = re .sub(r'\W', ' ', essay)
    words = n.word_tokenize(clean_essay)
    
    return len(words)


# In[38]:


# tokenizing an essay into a list of word lists

def tokenize(essay):
    stripped_essay = essay.strip()
    
    tokenizer = n.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    
    return tokenized_sentences


# In[39]:


# calculating number of nouns, adjectives, verbs and adverbs in an essay

def count_pos(essay):
    
    tokenized_sentences = tokenize(essay)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = n.pos_tag(sentence)
        
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count
    


# In[40]:


# checking number of misspelled words

def count_spell_error(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
    #big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg 
    #         and lists of most frequent words from Wiktionary and the British National Corpus.
    #         It contains about a million words.
    data = open('big.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if not word in word_dict:
            mispell_count += 1
           # print(word)
    
    return mispell_count


# In[41]:


def essayof_development_essay(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
  
    data = open('development_essay.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[42]:


def essayof_environment(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
   
    data = open('environment.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
           # print(word)
    
    return mispell_count


# In[43]:


def essayof_ETHICS(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
    
    data = open('ETHICS.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[44]:


def essayof_Society_essay(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
  
    data = open('Society_essay.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[45]:


def essayof_social(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
  
    data = open('social.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[46]:


def essayof_res_essay(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
   
    data = open('res_essay.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[47]:


def essayof_festival_essay(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
   
    data = open('festival_essay.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[48]:


def essayof_development_essay(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
    
    
    data = open('development_essay.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if  word in word_dict:
            mispell_count += 1
            #print(word)
    
    return mispell_count


# In[49]:


# calculating number of lemmas per essay

def count_lemmas(essay):
    
    tokenized_sentences = tokenize(essay)      
    
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for sentence in tokenized_sentences:
        tagged_tokens = n.pos_tag(sentence) 
        
        for token_tuple in tagged_tokens:
        
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count


# In[50]:


# extracting essay features

def extract_features(data):
    
    features = data.copy()
    
    features['char_count'] = features['essay'].apply(char_count)
    
    features['word_count'] = features['essay'].apply(word_count)
    
    features['sent_count'] = features['essay'].apply(sent_count)
    
    features['avg_word_len'] = features['essay'].apply(avg_word_len)
    
    features['lemma_count'] = features['essay'].apply(count_lemmas)
    
    features['spell_err_count'] = features['essay'].apply(count_spell_error)
    
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))
    
    return features


# In[51]:


def values(eassy):
    features={}
    #features['char_count'] = char_count(eassy)
    
    #features['word_count'] = word_count(essay)
    
    #features['sent_count'] = sent_count(essay)
    
    features['avg_word_len'] =avg_word_len(essay)
    
    features['lemma_count'] = count_lemmas(essay)
    
    features['spell_err_count'] =count_spell_error(essay)
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = count_pos(essay)
    
    
    return features
    


# In[52]:


def char_count(essay):
    
    clean_essay = re.sub(r'\s', '', str(essay).lower())
    
    return len(clean_essay)


# In[53]:


data=pd.read_csv("test_case.csv")


# In[54]:


#print(data.columns.values)
independent_variables=data.columns
independent_variables=independent_variables.delete(0)
independent_variables=independent_variables.delete(0)
independent_variables=independent_variables.delete(0)
independent_variables=independent_variables.delete(0)
#independent_variables=independent_variables.delete(9)
independent_variables=independent_variables.delete(1)
independent_variables=independent_variables.delete(0)


# In[55]:


x=data[independent_variables]
y=data["domain1_score"]


# In[56]:


import sklearn.linear_model as lm
lr=lm.LinearRegression()
lr.fit(x,y)


# In[60]:


y_pre=lr.predict(x)


# In[63]:


import sys
essay=input("enter the essay")
types=int(input("enter the type"))
l=[]


# In[65]:


l.append(essayof_development_essay(essay))
l.append(essayof_environment(essay))
l.append(essayof_ETHICS(essay))
l.append(essayof_festival_essay(essay))
l.append(essayof_res_essay(essay))
l.append(essayof_social(essay))
l.append(essayof_Society_essay(essay))
m=max(l)
i=l.index(m)


if i==types:
    r=values(essay)
    #print(r)
    for i in independent_variables:
        r_df=pd.DataFrame(data=r,index=[0],columns=x.columns)
    #print(r_df)
    y=lr.predict(r_df)
    print(y)
else:
    print("The essay is on worng topic")
    r=values(essay)
    #print(r)
    for i in independent_variables:
        r_df=pd.DataFrame(data=r,index=[0],columns=x.columns)
    #print(r_df)
    y=lr.predict(r_df)
    print(y)
    


# In[ ]:




