import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from collections.abc import Iterable
import faiss
from mlflow import pyfunc


class ContentBasedRecommender(pyfunc.PythonModel):
    
    name = 'Content-Based'
    
    def __init__(self,            
                 rating_df, candidate_df,
                user_id_column = 'user_id',
                candidate_id_column = "article_id",
                rating_target = 'time_per_word',
                train_test_split_perc = None,
                use_pca = False,
                **kwargs):
        self.candidate_id_column = candidate_id_column
        self.user_id_column = user_id_column
        if len(rating_target)>0:
            self.weight_column = rating_target[0]

        # self.item_ids = rating_df[candidate_id_column].unique()
        self.item_ids = candidate_df.index

        # self.items_df = items_df
        embeding_cols = [c for c in candidate_df.columns if c!=candidate_id_column]
        self.article_emmbeding_df = candidate_df[embeding_cols]
        self.article_embeddings_sparse = scipy.sparse.csr_matrix( self.article_emmbeding_df.values)
        self.scaler = StandardScaler()

        self.user_profiles = {}
        # self.user_interactions =  rating_df.groupby(user_id_column)[candidate_id_column].apply(
        #         lambda x: [self.item_map[a] for a in x if a in self.item_map]
        #     )

        def smooth_user_preference(x):
            return math.log(1+x, 2)
        self.user_interactions = rating_df \
                    .groupby([self.user_id_column, self.candidate_id_column])[self.weight_column].sum() \
                    .apply(smooth_user_preference).reset_index()
        
        self.interactions_train_df, self.interactions_test_df = train_test_split(self.user_interactions,
                                   stratify=self.user_interactions[user_id_column], 
                                   train_size=train_test_split_perc,
                                   random_state=42)
        
    def train(self):
        self.even_faster_build_users_profiles()
        self.build_faiss_index()
        self.faiss_batch_recommend()
        pass
        
        
    def get_model_name(self):
        return self.name
    
    def faiss_get_similar_items_to_user_profile(self, user_id, topn=1000):
        u = self.user_profiles[user_id].toarray().astype('float32')
        # user already normalized earlier; if not:
        u /= (np.linalg.norm(u) + 1e-9)

        sims, idx = self.faiss_index.search(u, topn)  # < 1 ms

        item_ids = self.item_ids[idx[0]]
        scores = sims[0]
        return np.column_stack((item_ids, scores))
    
    # def faster_get_similar_items_to_user_profile(self, user_id, topn=1000):
    #     #Get normalized user profile 
    #     u = self.user_profiles[user_id]
    #     u = u.toarray()          # (1, d)
        
    #     #Pre-normalized item matrix 
    #     if not hasattr(self, "_normed_items"):
    #         A = self.article_emmbeding_df.values.astype(np.float32)
    #         norms = np.linalg.norm(A, axis=1, keepdims=True)
    #         self._normed_items = A / (norms + 1e-9)   # (n_items, d)

    #     # Cosine similarity = dot product
    #     sims = (u @ self._normed_items.T).ravel()     # (n_items,)

    #     #Partial top-k selection instead of full sort
    #     if topn >= len(sims):
    #         top_idx = np.argsort(sims)[::-1]
    #     else:
    #         top_idx = np.argpartition(sims, -topn)[-topn:]
    #         top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    #     top_item_ids = self.item_ids[top_idx]
    #     top_scores   = sims[top_idx]

    #     #Build output list (item_id, score)*
    #     return np.column_stack((top_item_ids, top_scores))
    #     # return [(self.item_ids[i], sims[i]) for i in top_idx]
        
    def _get_similar_items_to_user_profile(self, user_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[user_id], self.article_emmbeding_df)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    # def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
    #     similar_items = self.faiss_get_similar_items_to_user_profile(user_id)
    #     #Ignores items the user has already interacted
    #     # SLOW similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

    #     ignore = set(items_to_ignore)
    #     mask = ~np.isin([i for i, _ in similar_items], list(ignore))
    #     similar_items_filtered = np.array(similar_items)[mask]
        
    #     recommendations_df = pd.DataFrame(similar_items_filtered, columns=[self.candidate_id_column, 'recStrength']) \
    #                                 .head(topn)

    #     if verbose:
    #         # if self.items_df is None:
    #         #     raise Exception('"items_df" is required in verbose mode')

    #         # recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
    #         #                                               left_on = 'contentId', 
    #         #                                               right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]
    #         pass

    #     return recommendations_df
    
    def faiss_recommend_items(self, user_id, items_to_ignore=[], topn=10):
        uidx = np.where(self._user_list == user_id)[0][0]

        rec_ids = self.item_ids[self._faiss_idx[uidx]]
        rec_sims = self._faiss_sims[uidx]

        # fast ignore filtering
        ignore = set(items_to_ignore)
        mask = ~np.isin(rec_ids, list(ignore))
        rec_ids = rec_ids[mask]
        rec_sims = rec_sims[mask]

        return rec_ids[:topn], rec_sims[:topn]
    
    def predict(self, context, model_input):
        user_ids = model_input["user_id"].tolist()
        N = model_input.get("N", 5) if "N" in model_input else 5
        results = [self.faiss_recommend_items(uid, N) for uid in user_ids]
        return results

    def get_item_profile(self,item_id):
        idx = int(item_id)#item_ids.index(item_id)
        item_profile = self.article_embeddings_sparse[idx:idx+1]

        return item_profile

    def get_item_profiles(self,ids):
        if not isinstance(ids, Iterable):
            ids = [ids]
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    # def build_user_profile(self,user_id, interactions_indexed_df):
    #     interactions_person_df = interactions_indexed_df.loc[user_id]
    #     user_item_profiles = self.get_item_profiles(interactions_person_df[self.candidate_id_column].tolist())
        
    #     user_item_strengths = np.array(interactions_person_df[self.weight_column]).reshape(-1,1)

    #     #Weighted average of item profiles by the interactions strength
    #     user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    #     user_profile_norm = user_item_strengths_weighted_avg #normalize?

    #     return user_profile_norm

    # def build_users_profiles(self,): 
    #     interactions_indexed_df = self.interactions_train_df[self.interactions_train_df[self.candidate_id_column] \
    #                                                 .isin(self.item_ids)].set_index(self.user_id_column)#articles_df['contentId']
        
    #     #SLOW ITER
    #     for user_id in interactions_indexed_df.index.unique():
    #         self.user_profiles[user_id] = self.build_user_profile(user_id, interactions_indexed_df)
    #     return self.user_profiles
    
    # def fast_build_users_profiles(self):
    #     df = (
    #     self.interactions_train_df[
    #         self.interactions_train_df[self.candidate_id_column].isin(self.item_ids)
    #     ].set_index(self.user_id_column)
    # )

    #     # item_id already IS the row index – honor your design
    #     df["idx"] = df[self.candidate_id_column].astype(int)

    #     # Group once
    #     grouped = df.groupby(self.user_id_column)

    #     for user_id, grp in grouped:
    #         rows = grp["idx"].values
    #         weights = grp[self.weight_column].values.reshape(-1, 1)

    #         # Vectorized sparse slice (real batch, no Python loop):
    #         M = self.article_embeddings_sparse[rows]

    #         # Weighted sum (sparse optimized)
    #         weighted = M.multiply(weights).sum(axis=0)

    #         weighted = np.asarray(weighted).ravel()
    #         s = weights.sum()

    #         if s != 0:
    #             profile = weighted / s
    #         else:
    #             profile = weighted

    #         # normalize
    #         norm = np.linalg.norm(profile)
    #         if norm > 0:
    #             profile = profile / norm

    #         # store as (1 × D) sparse row like before
    #         self.user_profiles[user_id] = scipy.sparse.csr_matrix(profile)

    #     return self.user_profiles
    
    def even_faster_build_users_profiles(self):
        df = (
        self.interactions_train_df[
            self.interactions_train_df[self.candidate_id_column].isin(self.item_ids)
        ].set_index(self.user_id_column)
        )

        # Create item index column 
        item_id_to_row = {id: idx for idx, id in enumerate(self.item_ids)}  # <-- ADD THIS LINE
        df["item_idx"] = df[self.candidate_id_column].map(item_id_to_row)
        # df["item_idx"] = df[self.candidate_id_column].astype(int)

        # Factorize user_id index to get dense 0..n_users-1 integers
        # df["user_idx"] = df.index.factorize()[0]
        codes, uniques = pd.factorize(df.index)
        df["user_idx"] = codes

        # Extract arrays
        user_index = df["user_idx"].values        # integer row index
        item_index = df["item_idx"].values        # integer column index
        weights    = df[self.weight_column].values

        n_users = df["user_idx"].max() + 1
        n_items = self.article_embeddings_sparse.shape[0]

        # Construct sparse user-item matrix 
        UIM = csr_matrix((weights, (user_index, item_index)),
                        shape=(n_users, n_items))

        # Matrix multiply → all user profiles at once
        user_profiles_dense = UIM @ self.article_embeddings_sparse

        # Normalize rows
        # user_profiles_dense = normalize(user_profiles_dense, norm="l2", axis=1)

        user_profiles_dense = user_profiles_dense.toarray()  # bring to dense
        norms = np.linalg.norm(user_profiles_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1
        user_profiles_dense /= norms

        # Store in dict format
        user_ids = uniques
        # user_ids = df.index.unique()

        self.user_profiles = {
            user_id: scipy.sparse.csr_matrix(user_profiles_dense[i])
            for i, user_id in enumerate(user_ids)
        }

        return self.user_profiles

    def build_faiss_index(self):
        # normalized item embeddings
        A = self.article_emmbeding_df.values.astype('float32')
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / (norms + 1e-9)

        d = A.shape[1]

        # cosine similarity = inner product on normalized vectors
        index = faiss.IndexFlatIP(d)
        index.add(A)

        self.faiss_index = index
        self._normed_items = A

    def faiss_batch_recommend(self, topn=105):
        # Stack all user profiles into (n_users, d)
        U = np.vstack([self.user_profiles[uid].toarray() for uid in self.user_profiles.keys()]).astype('float32')

        # Normalize if not normalized
        U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)

        # Batch search
        sims, idx = self.faiss_index.search(U, topn)

        # Map FAISS row index → user_id
        self._user_list = np.array(list(self.user_profiles.keys()))
        # self._user_to_faiss_idx = {uid: i for i, uid in enumerate(self._user_list)}

        # Store results for evaluation
        self._faiss_idx   = idx
        self._faiss_sims  = sims

if __name__ == '__main__':
    import model_evaluator
    import pickle
    RATING_PREPROC_DF_PATH="Training/Datas/rating_preprocess_df.pkl"
    CANDIDATE_PREPROC_DF_PATH='Training/Datas/candidate_preprocess_df.pkl'

    with open(RATING_PREPROC_DF_PATH , 'rb') as f:
        rating_df = pickle.load(f)
    with open(CANDIDATE_PREPROC_DF_PATH , 'rb') as f:
        candidate_df = pickle.load(f)


    model = ContentBasedRecommender(rating_df,candidate_df)
    model.train()
    
    rec = model.faiss_recommend_items(42)
    print(rec)

    # print('Evaluating Content-Based Filtering model...')
    # cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(model)
    # print('\nGlobal metrics:\n%s' % cb_global_metrics)
    # cb_detailed_results_df.head(10)


    pass
