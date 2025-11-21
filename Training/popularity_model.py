
from mlflow import pyfunc
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import csr_matrix

class PopularityArticleRecommender(pyfunc.PythonModel):
    
    name = 'PopularityArticleRecommender'
    
    def __init__(self,            
                 rating_df, candidate_df,
                user_id_column = 'user_id',
                candidate_id_column = "article_id",
                rating_target = None,
                train_test_split_perc = None,
                **kwargs):
        
        self.candidate_id_column = candidate_id_column
        self.user_id_column = user_id_column
        if type(rating_target)==type(None):
            self.rating_target = 'count'
            rating_df['count'] = 1
        else:
            self.rating_target = rating_target
        self.item_ids = candidate_df.index
        
        
        self.user_interactions = rating_df.groupby([self.user_id_column, self.candidate_id_column]).first().reset_index()
        # self.user_interactions = rating_df.groupby([self.user_id_column, self.candidate_id_column]).reset_index()
        
        self.interactions_train_df, self.interactions_test_df = train_test_split(self.user_interactions,
                                   stratify=self.user_interactions[user_id_column], 
                                   train_size=train_test_split_perc,
                                   random_state=42)
        
        #  Buid train popularity
        item_popularity_df = self.interactions_train_df.groupby(candidate_id_column)[self.rating_target]\
                                .sum().sort_values(ascending=False).reset_index()
        self.user_exclusion_df = self.interactions_train_df.groupby(user_id_column)[candidate_id_column]\
                                 .agg(list).reset_index(name='seen_items')
        
        self.popularity_df = item_popularity_df
        pass

    def train(self):
        """Precompute FAISS recommendations for all users"""
        self.build_faiss_popularity_recommendations()
        
    def recommend_items(self, user_id, 
                        ignore_seen = True,
                        topn=5
                        ):
        if ignore_seen:
            items_to_ignore =  self.user_exclusion_df.loc[self.user_exclusion_df[self.user_id_column]==user_id,
                                                        'seen_items'].values
        else:
            items_to_ignore = []
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df[self.candidate_id_column].isin(items_to_ignore)] \
                               .sort_values(self.rating_target, ascending = False) \
                               .head(topn)

        return recommendations_df
    
    def build_faiss_popularity_recommendations(self, topn=1000):


        # ----- 1) Map items → indices -----
        item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        # popularity ranking → item indices
        ranked_items = self.popularity_df[self.candidate_id_column].map(item_to_idx).to_numpy()
        n_items = len(self.item_ids)

        # ----- 2) User list (from test-set! evaluation uses this) -----
        self._user_list = np.array(self.interactions_test_df[self.user_id_column].unique())
        n_users = len(self._user_list)
        self._user_to_idx = {uid: i for i, uid in enumerate(self._user_list)}

        # ----- 3) Build USER × ITEM sparse mask of "seen items" -----
        rows = self.interactions_train_df[self.user_id_column].map(self._user_to_idx, na_action='ignore')
        mask = rows.notna()
        rows = rows[mask].astype(int).values
        cols = (
            self.interactions_train_df.loc[mask, self.candidate_id_column]
            .map(item_to_idx)
            .values
        )

        seen_matrix = csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            shape=(n_users, n_items)
        )



        # Preallocate final matrix
        topn = min(topn, len(ranked_items))
        recs = np.full((n_users, topn), -1, dtype=np.int32)

        # Convert popularity list to sparse columns
        ranked_sparse_cols = ranked_items

        # ----- 5) Vectorized filtering -----
        # Strategy:
        #   For each user, we look up seen items via sparse matrix rows.
        #   We remove those indices from ranked_items.
        # This is done using fast CSR row iteration, not dense loops.

        for u in range(n_users):
            seen_cols = seen_matrix.indices[
                seen_matrix.indptr[u]:seen_matrix.indptr[u+1]
            ]

            if len(seen_cols) == 0:
                # user has seen nothing → just take ranking directly
                recs[u] = ranked_sparse_cols[:topn]
                continue

            # Fast set difference using boolean mask
            mask = np.ones(len(ranked_sparse_cols), dtype=bool)
            mask[np.isin(ranked_sparse_cols, seen_cols)] = False

            filtered = ranked_sparse_cols[mask][:topn]

            # pad if needed
            if len(filtered) < topn:
                padded = np.full(topn, -1, dtype=np.int32)
                padded[:len(filtered)] = filtered
                filtered = padded

            recs[u] = filtered

        # Store final matrix where evaluator expects it
        self._faiss_idx = recs
    
    def slow_build_faiss_popularity_recommendations(self, topn=5):
        """
        Precompute popularity recommendations in the same structure expected 
        by faiss_evaluate_model()
        """
        # Sorted popular items (already sorted in popularity_df)
        ranked_items = self.popularity_df[self.candidate_id_column].tolist()

        # Map item_id → index in model.item_ids
        item_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}

        # Convert popularity ranking to index ranking
        ranked_item_indices = np.array([item_to_idx[i] for i in ranked_items], dtype=np.int32)

        # Users from test set (the evaluation uses these)
        self._user_list = np.array(self.interactions_test_df[self.user_id_column].unique())
        self._user_to_idx = {uid: i for i, uid in enumerate(self._user_list)}

        # Exclusion map: items seen in TRAIN set
        train_seen = dict(zip(
            self.user_exclusion_df[self.user_id_column],
            self.user_exclusion_df["seen_items"]
        ))

        # Precompute recommendation list (same for all users except exclusions)
        self._faiss_idx = np.zeros((len(self._user_list), min(topn, len(ranked_item_indices))), dtype=np.int32)

        for u, uid in enumerate(self._user_list):

            seen_items = set(train_seen.get(uid, []))

            # Filter popular items
            filtered_items = [i for i in ranked_items if i not in seen_items]

            # Convert to indices
            filtered_idx = [item_to_idx[i] for i in filtered_items[:topn]]

            # If fewer than topn, pad with -1 (evaluation will ignore unseen indices)
            if len(filtered_idx) < topn:
                filtered_idx += [-1] * (topn - len(filtered_idx))

            self._faiss_idx[u] = np.array(filtered_idx, dtype=np.int32)
    
    def faiss_recommend_items(self, user_id, items_to_ignore=[], topn=5):
        uidx = self._user_to_idx[user_id]#np.where(self._user_list == user_id)[0][0]

        rec_ids = self.item_ids[self._faiss_idx[uidx]]
        # rec_sims = self._faiss_sims[uidx]

        # fast ignore filtering
        ignore = set(items_to_ignore)
        mask = ~np.isin(rec_ids, list(ignore))
        rec_ids = rec_ids[mask]
        # rec_sims = rec_sims[mask]

        return rec_ids[:topn]#, rec_sims[:topn]
    
if __name__ == '__main__':
    import model_evaluator
    import pickle
    RATING_PREPROC_DF_PATH="Training/Datas/rating_preprocess_df.pkl"
    CANDIDATE_PREPROC_DF_PATH='Training/Datas/candidate_preprocess_df.pkl'

    with open(RATING_PREPROC_DF_PATH , 'rb') as f:
        rating_df = pickle.load(f)
    with open(CANDIDATE_PREPROC_DF_PATH , 'rb') as f:
        candidate_df = pickle.load(f)


    model = PopularityArticleRecommender(rating_df,candidate_df)
    # model.train()
    
    rec = model.recommend_items(42)
    print(rec)
    model.train()
    rec = model.faiss_recommend_items(42)
    print(rec)

    POP_MODEL_PATH = 'Training/model_assets/pop_model.pkl'
    with open(POP_MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)