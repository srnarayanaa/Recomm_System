# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:16:11 2020

@author: srnarayanaa
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def combine_features(row):
    return row['keywords'] +" "+row["category"]+" "+row["seller"]

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

df = pd.read_csv("product.csv")
print(df)
features = ['keywords','category','seller']

for feature in features:
    df[feature] = df[feature].fillna('')
    
df["combined_features"] = df.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)
#GIVE YOUR INPUT HERE
product_user_likes = "ROG Keyboard"
product_index = get_index_from_title(product_user_likes)
similar_products =  list(enumerate(cosine_sim[product_index]))
sorted_similar_products = sorted(similar_products,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Products according to similarity : "+product_user_likes+" are:\n")

for element in sorted_similar_products:
    print(get_title_from_index(element[0]))
