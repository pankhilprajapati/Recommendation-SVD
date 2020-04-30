import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments
import Recommender.Recommender_functions as rf
from tqdm import tqdm
class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''



    def fit(self, review_pth,movie_pth, latent_features=12,learning_rate=0.005,iters=200):
        '''
        Function:
             - fit = perform matrix factorization using FunkSVD to 
                     make prediction of rating given by user to item
        Arguments: 
             - review_pth = path for the reviews dataset
             - movie_pth = path for the movies dataset
             - latent_feature = no of the feature apart from user-items
               which are use to predict the ratings
             - learning_rate = rate at which error is reduced using gradiant 
               decient
             - iters = no of time teh iteration to don in the matrix 
        return:
            None

            storing - n_users, n_movies, rmse, num_rating,
                      perc_rated, user_mat, movie_mat  
        '''
        self.review_df = pd.read_csv(review_pth)
        self.movie_df = pd.read_csv(movie_pth)
        user_item = self.review_df[['user_id','movie_id','rating','timestamp']]
        self.user_item_df = user_item.groupby(['user_id','movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_item_df)
        
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters 

        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_rating = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids = np.array(self.user_item_df.index)
        self.movie_ids = np.array(self.user_item_df.columns)


        user_mat = np.random.rand(self.n_users,self.latent_features)
        movie_mat = np.random.rand(self.latent_features,self.n_movies)

        self.sse_accum = 0
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        for iteration in tqdm(range(self.iters)):
            self.o_sse = self.sse_accum
            self.sse_accum = 0

            for i in range(self.n_users):
                for j in range(self.n_movies):

                    if self.user_item_mat[i,j]>0:
                        diff = self.user_item_mat[i,j] - np.dot(user_mat[i,:],movie_mat[:,j])
                        self.sse_accum += diff**2

                        for k in range(self.latent_features):
                            user_mat[i,k] += self.learning_rate* 2*diff*movie_mat[k,j]
                            movie_mat[k,j] += self.learning_rate* 2*diff*user_mat[i,k]
            
            print("%d \t\t %f"%(iteration+1,self.sse_accum/self.num_rating))
        self.user_mat = user_mat
        self.movie_mat = movie_mat
        self.ranked_movies = rf.create_ranked_df(self.movie_df,self.review_df) 

    def predict_rating(self,user_id,movie_id):
        '''
        Function:
             - predict_rating = predict the rating if the non-Nan values 

        Arguments:
             - user_id = id of the user who will rate
             - movie_id = id of the movie which will be rated
        return:
             - pred = matix of user-movie with the predicted values

        '''
        try:

            user_row = np.where(self.user_ids == user_id)[0][0]
            movie_col = np.where(self.movie_ids == movie_id)[0][0]
            
            pred =  np.dot(self.user_mat[user_row,:],self.movie_mat[:,movie_col])
            movie_name = str(self.movie_df[self.movies_df['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred
        except:
            print('something is wront may be user or movie matrix is not proper')

            return None

    def make_recs(self,_id,_id_type='movie',rec_num=10):
        '''
        Arguments: 
            - _id = either a user or movie id (int)
            - _id_type = 'movie' or 'user' (str)
            - rec_num = no. of recommendation to be made
        Return 
            rec_ids - id of movie with recommendation 
            rec_name - name of the movie recommended             
        '''
        rec_ids,rec_names = None,None
        if _id_type == 'user':
            if _id in self.user_ids:
                # if user is present
                idx = np.where(self.user_ids == _id)[0][0]
                preds = np.dot(self.user_mat[idx,:],self.movie_mat[:,idx])

                indx = preds.argsort()[-rec_num:][::-1]
                rec_ids = self.movie_ids[indx]
                rec_names = rf.get_movie_names(rec_ids, self.movie_df)
            else:

                rec_names = rf.popular_recommendations(_id,rec_num,self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")
        else:
            if _id in self.movie_ids:
                rec_names = list(rf.find_similar_movies(_id,self.movie_df))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")
        
        return rec_ids,rec_names

# How to run
# if __name__ == '__main__':
#     import Recommender as r

#     #instantiate recommender
#     rec = r.Recommender()

#     # fit recommender
#     rec.fit(review_pth='data/train_data.csv', movie_pth= 'data/movies_clean.csv', learning_rate=.01, iters=1)

#     # predict
#     rec.predict_rating(user_id=8, movie_id=2844)

#     # make recommendations
#     print(rec.make_recs(8,'user')) # user in the dataset
#     print(rec.make_recs(1,'user')) # user not in dataset
#     print(rec.make_recs(1853728)) # movie in the dataset
#     print(rec.make_recs(1)) # movie not in dataset
#     print(rec.n_users)
#     print(rec.n_movies)
#     print(rec.num_rating)



