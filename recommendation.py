# https://www.analyticsvidhya.com/blog/2020/11/create-your-own-movie-movie-recommendation-system/
# env: recom

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

#========================================
# Load data
#========================================

movies = pd.read_csv("/Users/juliesohn/Desktop/Coding/recommendation/data/movies.csv")
ratings = pd.read_csv("/Users/juliesohn/Desktop/Coding/recommendation/data/ratings.csv")

#movies.head(3)
#ratings.head(3)

# new dataframe where each column would represent each unique
# userId and each row represents each unique movieId.
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)
#final_dataset.head()

#========================================
# Filter
#========================================
"""
To qualify a movie, a minimum of 10 users should have voted a movie.
To qualify a user, a minimum of 50 movies should have voted by the user.
"""

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

"""f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()
"""
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]

"""f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()"""

final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

# The data is high-dimensional, and many of the values are sparse
# Reduce model sparsity:
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)


#========================================
# Movie Recommendation System
#========================================

# Model
knn = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=20,
    n_jobs=-1)
knn.fit(csr_data)

# Function
def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_recommend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_recommend+1))
        return df
    else:
        return "No movies found. Please check your input"


if __name__=="__main__":
    print(get_movie_recommendation('Iron Man'))
    #get_movie_recommendation('Memento')
    #get_movie_recommendation('The Bodyguard')
