#importing modules
import time, gc, argparse, os
import pandas as pd 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

def model_param(n_neighbors, algorithm, metric, n_jobs=None):
	model=NearestNeighbors()
	if n_jobs and (n_jobs > 1 or n_jobs == -1):
		os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
	model.set_params(**{'n_neighbors': n_neighbors,'algorithm': algorithm,'metric': metric,'n_jobs': n_jobs})
	return model

def _fuzzy_matching(hashmap, product):
	match_tuple = []
	for title, idx in hashmap.items():
		ratio = fuzz.ratio(title.lower(), product.lower())
		if ratio >= 60:
			match_tuple.append((title, idx, ratio))
	match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
	if not match_tuple:
		print('Oops! No match is found')
	else:
		return match_tuple[0][1]

def preprocessing():
	df_products = pd.read_csv('products.csv')
	df_ratings = pd.read_csv('ratings.csv')
	product_count = pd.DataFrame(df_ratings.groupby('productId').size(),columns=['count'])
	bestsellingproducts = list(set(product_count.query('count >= 50').index))
	product_filter = df_ratings.productId.isin(bestsellingproducts).values
	user_count = pd.DataFrame(df_ratings.groupby('userId').size(),columns=['count'])
	active_users = list(set(user_count.query('count >= 50').index))
	user_filter = df_ratings.userId.isin(active_users).values
	df_ratings_filtered = df_ratings[product_filter & user_filter]
	prod_user_mat = df_ratings_filtered.pivot(index='productId', columns='userId', values='rating').fillna(0)
	hashmap = {product: i for i, product in enumerate(list(df_products.set_index('productId').loc[prod_user_mat.index].title))}
	prod_user_sparse = csr_matrix(prod_user_mat.values)
	del df_products, product_count, user_count
	del df_ratings, df_ratings_filtered, prod_user_mat
	gc.collect()
	return prod_user_sparse, hashmap

def recommend(model, data, hashmap, product, count):
	model.fit(data)
	idx=_fuzzy_matching(hashmap, product)
	print('Product you have searched is : ',product)
	print('#.........#\n')
	t0 = time.time()
	distances, indices = model.kneighbors(data[idx], n_neighbors = count+1)
	raw_recommends = \
		sorted(
			list(
				zip(
					indices.squeeze().tolist(),
					distances.squeeze().tolist()
				)
			),
			key=lambda x: x[1]
		)[:0:-1]
	print('It took my system {:.2f}s to make inference \n'.format(time.time() - t0))
	return raw_recommends

def make_recommendation(product, count, model, prod_user_sparse, hashmap):
	raw_recommends = recommend(model, prod_user_sparse, hashmap, product, count)
	reverse_hashmap = {v: k for k, v in hashmap.items()}
	print('Recommendations for {}:'.format(product))
	for i, (idx, dist) in enumerate(raw_recommends):
		print('{0}: {1}, with distance of {2}'.format(i+1, reverse_hashmap[idx], dist))

if __name__ == '__main__':
	product=str(input())
	count=10
	while product!='-1':
		model=model_param(20,'brute','cosine',-1)
		prod_user_sparse, hashmap = preprocessing()
		make_recommendation(product, count, model, prod_user_sparse, hashmap)
		print('\n')
		product=str(input())
