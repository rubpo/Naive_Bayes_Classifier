#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
import numpy as np
from collections import Counter
import math


# 1. Divide the data in two groups: training and test examples

# In[2]:


def read_data(filename):
    with open(filename, 'r',encoding="utf8", errors='ignore') as input:
        for line in input:
            yield line
            


# In[3]:


def ham_split(data):
  for i in data:
    i=i.split("\t")
    if i[0]=='ham':
      yield i[1]

def spam_split(data):
  for i in data:
    i=i.split("\t")
    if i[0]=='spam':
      yield i[1]

def get_target(data) : 
  liste=[]
  for i in data: 
    i=i.split("\t")
    if i[0]=='ham' :
      liste.append(0)
    else :
      liste.append(1)
  return liste


# 2. Parse both the training and test examples to generate both the spam and ham data sets.

# In[4]:


data_file="messages.txt"

data = read_data(data_file)
data_list=list(data)


size = len(data_list)
training_size=int(0.7*size)
test_size=int(0.3*size)

training_data=data_list[:training_size]
test_data=data_list[training_size:training_size+test_size]


ham_training=list(ham_split(training_data))
ham_test=list(ham_split(test_data))
spam_training=list(spam_split(training_data))
spam_test=list(spam_split(test_data))

training_target_matrix=get_target(training_data)
test_target_matrix=get_target(test_data)


# 3. Generate a dictionary from the training data.

# In[5]:


def make_Dictionary(data):
    all_words = []
    for line in data:
      words = line.split()
      all_words += words
    for w in all_words: 
        if w.isalpha() == False:
          for i in range(all_words.count(w)):
            all_words.remove(w)
        elif len(w) == 1:
          for i in range(all_words.count(w)):
            all_words.remove(w)
    all_words=[x.lower() for x in all_words]
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(3000)
    return dictionary


# In[6]:


clean_training_data=ham_training+spam_training
clean_test_data=ham_test+spam_test

training_dictionary=make_Dictionary(clean_training_data)


# 4. Extract features from both the training data and test data.

# In[7]:


def extract_features(data,dictionary):
    features_matrix = np.zeros((len(data), 3000))
    lineID = 0
    for line in data:
      words = line.split()
      for word in words:
        wordID = 0
        for i, d in enumerate(dictionary):
          if d[0] == word:
            wordID = i
            features_matrix[lineID, wordID] = 1 
      lineID = lineID + 1
    return features_matrix


# In[8]:


train_matrix = extract_features(clean_training_data,training_dictionary)
test_matrix = extract_features(clean_test_data,training_dictionary)

test_matrix


# 5. Implement the Naive Bayes from scratch, fit the respective models to the training data.

# In[9]:


def fit(matrix_input, matrix_target) :
  true_probability = matrix_target.count(1)/len(matrix_target)
  false_probability = 1 - true_probability
  matrix_prob_positive = []
  matrix_prob_negative = []
  ham_count=0
  spam_count=0
  columns = [l for l in zip(*matrix_input)]
  for i in range(len(columns)): 
    for j in range(len(columns[i])): 
      if matrix_target[j]==1:
        spam_count=spam_count+columns[i][j]
      else :
        ham_count=ham_count+columns[i][j]
    matrix_prob_positive.append((spam_count*true_probability +1)/matrix_target.count(1)+2)
    matrix_prob_negative.append((ham_count*false_probability+1)/matrix_target.count(0)+2)
    ham_count=0
    spam_count=0
  matrix_prob=[]
  matrix_prob.append(matrix_prob_positive)
  matrix_prob.append(matrix_prob_negative)
  return matrix_prob


# 6. Make predictions for the test data.

# In[10]:


def predict(matrix_prob,matrix_test) :
      result=[]
          for i in range(len(matrix_test)):
        pos_prob=1
    neg_prob=1
    for j in range(len(matrix_test[i])):
      if matrix_test[i][j]==1:
        pos_prob=pos_prob*(matrix_test[i][j]*matrix_prob[0][j])
        neg_prob=neg_prob*(matrix_test[i][j]*matrix_prob[1][j])
    decision = np.argmax([neg_prob,pos_prob])
    if decision == 0 :
      result.append(0)
    elif decision == 1:
      result.append(1)
  return result


# In[11]:


result=predict(fit(train_matrix, training_target_matrix),test_matrix)

result


# 7. Measure the spam-filtering performance for each approach through the confusion matrix.

# In[14]:


#confusion_matrix
def confusion_matrix(tested,predicted):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(0,len(predicted)):
        if predicted[i] == 1:
            if tested[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if tested[i] == 1:
                fn += 1
            else:
                tn += 1
    confusion_matrix=np.array([[tp, fp], [fn, tn]])
    return confusion_matrix

conf_matrix=confusion_matrix(test_target_matrix, result)
print(conf_matrix)

#accuracy

accuracy= ( conf_matrix[1,1] + conf_matrix[0,0] ) / (conf_matrix[1,1] + conf_matrix[0,0] + conf_matrix[1,0] + conf_matrix[0,1])
print('Accuracy : ',accuracy)

