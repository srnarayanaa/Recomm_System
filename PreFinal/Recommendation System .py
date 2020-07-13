#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING NEEDED MODULES THAT INCLUDE PANDAS, SCIPY, SKLEARN, FUZZYWUZZY


# In[2]:


import pandas as pd, time, gc, os, argparse
#TIME TO CALCULATE THE TIME TAKEN TO RECOMMEND


# In[3]:


from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# In[4]:


#DEFINE FUNCTIONS


# In[5]:


import numpy as np
def stringdist(s, t,ratio_calc):
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 
            else:
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1                   
            distance[row][col] = min(distance[row-1][col] + 1, distance[row][col-1] + 1,distance[row-1][col-1] + cost)
    Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    return int(Ratio*100)


# In[6]:


def match(H_MAP, product):
    L = []
    for name, index in H_MAP.items():
        R = stringdist(name.lower(), product.lower(),True)
        if R >= 60:
            L.append((name, index, R))
    L = sorted(L, key = lambda x : x[2])[::-1]
    if not L:
        return False
    else:
        return L[0][1]


# In[7]:


def preprocess():
    dataFrameP = pd.read_excel('products.xlsx')
    dataFrameR = pd.read_csv('ratings.csv')
    no_products = pd.DataFrame(dataFrameR.groupby('productId').size(), columns=['count'])
    mostrelatedproduct = list(set(no_products.query('count >= 50').index))
    filter_product = dataFrameR.productId.isin(mostrelatedproduct).values
    no_users = pd.DataFrame(dataFrameR.groupby('userId').size(), columns=['count'])
    trustable_users = list(set(no_users.query('count >= 50').index))
    filter_user = dataFrameR.userId.isin(trustable_users).values
    dataFrameRF = dataFrameR[filter_product & filter_user]
    P_U_MAT = dataFrameRF.pivot(index = 'productId', columns = 'userId', values = 'rating').fillna(0)
    H_MAP = {product: i for i, product in enumerate(list(dataFrameP.set_index('productId').loc[P_U_MAT.index].title))}
    P_U_SP = csr_matrix(P_U_MAT.values)
    del dataFrameP, no_products, no_users
    del dataFrameR, dataFrameRF, P_U_MAT
    gc.collect()
    return P_U_SP, H_MAP


# In[ ]:





# In[8]:


def recSystem(product, model, P_U_SP, H_MAP):
    model.fit(P_U_SP)
    index = match(H_MAP, product)
    print('You have searched for : ', product)
    print('\n')
    if index == False:
        print('Sorry, product not found!')
    else:
        ans=[]
        distances, indices = model.kneighbors(P_U_SP[index], n_neighbors = 11)
        rec = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key = lambda x: x[1])[0:-1]
        REV_H_MAP = {val: key for key, val in H_MAP.items()}
        for i, (index, dist) in enumerate(rec):
            ans.append(REV_H_MAP[index])
            #if index in REV_H_MAP: print('{0}: {1}, with distance of {2}'.format(i+1, REV_H_MAP[index], dist))
        return ans
            


# In[9]:


from time import sleep
import sys
def runprogram(): 
    model=NearestNeighbors()
    model.set_params(**{'n_neighbors': 20, 'algorithm': 'brute', 'metric': 'cosine', 'n_jobs': -1})
    p=str(input())
    while (p)!='-1':
        P_U_SP, H_MAP = preprocess()
        ans=recSystem(p, model, P_U_SP, H_MAP)
        for s in ans:
            print(s) ; sleep(0.6)
        print('\n')
        p=str(input())


# In[ ]:


runprogram()


# In[ ]:





# In[ ]:




