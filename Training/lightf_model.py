import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow import pyfunc

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation
from lightfm.evaluation import precision_at_k as lightfm_prec_at_k
from lightfm.evaluation import recall_at_k as lightfm_recall_at_k

from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.models.lightfm.lightfm_utils import (
    track_model_metrics,
    prepare_test_df,
    prepare_all_predictions,
    compare_metric,
    similar_users,
    similar_items,
)
from recommenders.utils.notebook_utils import store_metadata

print("System version: {}".format(sys.version))
print("LightFM version: {}".format(lightfm.__version__))



# default number of recommendations
K = 5
# percentage of data used for testing
TEST_PERCENTAGE = 0.25
# model learning rate
LEARNING_RATE = 0.25
# no of latent factors
NO_COMPONENTS = 20
# no of epochs to fit model
NO_EPOCHS = 20
# no of threads to fit model
NO_THREADS = 4
# regularisation for both user and item features
ITEM_ALPHA = 1e-6
USER_ALPHA = 1e-6

# seed for pseudonumber generations
SEED = 42


class ArticleRetrievalLightFM(pyfunc.PythonModel):
        
        def __init__(self, rating_df, candidate_df,
                  rating_targets=['time_per_word'],
                  user_id_column = 'user_id',
                  candidate_id_column = "article_id",
                  train_test_split_perc = None,
                  factors=64, 
                  model_type='BAY',                  
                  add_embeding_vector = False,
                  use_pca = False,
                  embedding_type = 'ALPHA_BLEND',
                  embeding_alpha=0.7,
                  debug = True,
                  ):
                
            """
            rating_df: DataFrame with columns
            candidate_df: DataFrame indexed by article_id
            factors: latent factor dim
            """

            dataset = Dataset()

            dataset.fit(users=rating_df['user_id'], 
                        items=rating_df['article_id'])
            ratings_target = ['time_per_word']


            num_users, num_articles = dataset.interactions_shape()
            print(f'Num users: {num_users}, num_topics: {num_articles}.')

            (self.interactions, weights) = dataset.build_interactions(
                                             rating_df[['user_id','article_id']+ratings_target].values
                                             )
            
            self.train_interactions, self.test_interactions = cross_validation.random_train_test_split(
                                                        self.interactions, test_percentage=TEST_PERCENTAGE,
                                                        random_state=np.random.RandomState(SEED)
                                                        )

            print(f"Shape of train interactions: {self.train_interactions.shape}")
            print(f"Shape of test interactions: {self.test_interactions.shape}")

            self.model = LightFM(loss='warp', 
                            no_components=NO_COMPONENTS, 
                            learning_rate=LEARNING_RATE,                 
                            random_state=np.random.RandomState(SEED)
                            )
            
            self.model.fit(interactions=self.train_interactions,
                    epochs=NO_EPOCHS)
    
            # Prepare model evaluation data

            uids, iids, interaction_data = cross_validation._shuffle(
                self.interactions.row, self.interactions.col, self.interactions.data, 
                random_state=np.random.RandomState(SEED))

            cutoff = int((1.0 - TEST_PERCENTAGE) * len(uids))
            test_idx = slice(cutoff, None)

            uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()

            with Timer() as test_time:
                test_df = prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights)
            print(f"Took {test_time.interval:.1f} seconds for prepare and predict test data.")  
            time_reco1 = test_time.interval

            

            with Timer() as test_time:
                all_predictions = prepare_all_predictions(rating_df, uid_map, iid_map, 
                                                        interactions=self.train_interactions,
                                                        model=self.model, 
                                                        num_threads=NO_THREADS)
            print(f"Took {test_time.interval:.1f} seconds for prepare and predict all data.")
            
            

            if debug : print(all_predictions.sample(5, random_state=SEED).to_markdown())


            # Model eval
            with Timer() as test_time:
                eval_precision = precision_at_k(rating_true=test_df, 
                                            rating_pred=all_predictions, k=K)
                eval_recall = recall_at_k(test_df, all_predictions, k=K)
            time_reco3 = test_time.interval

            with Timer() as test_time:
                eval_precision_lfm = lightfm_prec_at_k(self.model, self.test_interactions, 
                                                    self.train_interactions, k=K).mean()
                eval_recall_lfm = lightfm_recall_at_k(self.model, self.test_interactions, 
                                                    self.train_interactions, k=K).mean()
            time_lfm = test_time.interval
                
            print(
                "------ Using Repo's evaluation methods ------",
                f"Precision@K:\t{eval_precision:.6f}",
                f"Recall@K:\t{eval_recall:.6f}",
                "\n------ Using LightFM evaluation methods ------",
                f"Precision@K:\t{eval_precision_lfm:.6f}",
                f"Recall@K:\t{eval_recall_lfm:.6f}", 
                sep='\n')






            
        
            
            


if __name__ == '__main__':
    import pickle
    RATING_PREPROC_DF_PATH="Training/Datas/rating_preprocess_df.pkl"
    CANDIDATE_PREPROC_DF_PATH='Training/Datas/candidate_preprocess_df.pkl'

    with open(RATING_PREPROC_DF_PATH , 'rb') as f:
        rating_df = pickle.load(f)
    with open(CANDIDATE_PREPROC_DF_PATH , 'rb') as f:
        candidate_df = pickle.load(f)

    pipe = ArticleRetrievalLightFM(rating_df, candidate_df)

    

