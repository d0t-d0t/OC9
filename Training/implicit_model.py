import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from sklearn.preprocessing import normalize
from implicit.evaluation import precision_at_k,mean_average_precision_at_k,AUC_at_k,ndcg_at_k,train_test_split
from mlflow import pyfunc
from sklearn.decomposition import PCA

class ArticleRetrievalImplicit(pyfunc.PythonModel):
    """
    Implicit retrieval model
    """

    def __init__(self, rating_df, candidate_df,
                  rating_target=['time_per_word'],
                  user_id_column = 'user_id',
                  candidate_id_column = "article_id",
                  train_test_split_perc = None,
                  factors=64, 
                  model_type='BAY',                  
                  add_embeding_vector = False,
                  use_pca = False,
                  embedding_type = 'ALPHA_BLEND',
                  embeding_alpha=0.7,
                  **kargs
                  ):
        """
        rating_df: DataFrame with columns
        candidate_df: DataFrame indexed by article_id
        factors: latent factor dim
        """

        self.name = f'ArticleRetrievalImplicit_{model_type}'
        self.rating_df = rating_df.copy()
        self.candidate_df = candidate_df.copy()
        self.factors = factors
        self.alpha = embeding_alpha
        self.model_type = model_type
        self.use_pca = use_pca

        self.user_id_column = user_id_column
        self.candidate_id_column = candidate_id_column

        self.add_embeding_vector = add_embeding_vector
        self.embedding_type = embedding_type

        # Deduplicate ratings for single article by summing rating
        # if type(rating_target)!=type(None):
        #     self.rating_df = (
        #         self.rating_df.groupby([user_id_column, candidate_id_column], as_index=False)
        #         .agg({rating_target: "sum"})
        #     )

        # Create user-item interaction matrix
        user_ids = rating_df[user_id_column].unique()
        article_ids = rating_df[candidate_id_column].unique()

        if add_embeding_vector and self.embedding_type=='USER_MEAN':
            print('adding embeding')

            embeding_cols = [c for c in self.candidate_df.columns if c!=self.candidate_id_column]
            # rating_target += embeding_cols
            # self.rating_df = self.rating_df.merge(
            #     candidate_df[[candidate_id_column] + embeding_cols],
            #     on=candidate_id_column,
            #     how="left"
            # )##.set_index("article_id")
            item_features = candidate_df[embeding_cols].loc[article_ids]
            self.candidate_item = csr_matrix(item_features.values)




        #Build mappings 
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {article_id: idx for idx, article_id in enumerate(article_ids)}
        self.rev_user_map = {v: k for k, v in self.user_map.items()}
        self.rev_item_map = {v: k for k, v in self.item_map.items()}

        # Map matrix column index back to article_id
        self.matrix_idx_to_article_id = {idx: article_id for article_id, idx in self.item_map.items()}

        # Build interaction matrix
        self.rating_df["user_idx"] = self.rating_df[user_id_column].map(self.user_map)
        self.rating_df["item_idx"] = self.rating_df[candidate_id_column].map(self.item_map)

        rows = self.rating_df["user_idx"].values
        cols = self.rating_df["item_idx"].values
        if type(rating_target)!=type(None):
            if model_type == 'ALS':
                vals = self.rating_df.get(rating_target, pd.Series(1, index=self.rating_df.index)).values

                # Ensure positive non-zero confidence
                vals = np.maximum(vals, 0)

                #linear scaling for confidence weights
                alpha = 40  # try between 10–100
                vals = 1 + alpha * vals  # implicit feedback confidence formula

                # Ensure finite
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                vals = self.rating_df.get(rating_target, pd.Series(1, index=self.rating_df.index)).values

                # ensure all values are positive
                vals = np.maximum(vals, 0)

                # rescale for als
                vals = vals / (np.max(vals) + 1e-6)

            if np.isnan(vals).any():
                print("Warning: NaNs found in rating values")
            if np.isinf(vals).any():
                print("Warning: Infinite values found in rating values")
            if np.all(vals == 0):
                print("Warning: All-zero rating vector — ALS may diverge")
        else:
            # When no rating_target is provided, use binary interactions (all 1s)
            vals = np.ones(len(rows))
    
        # combined_vals = self.build_interaction_matrix( rating_target, weights=None)

        if type(train_test_split_perc)!=type(None):
            match model_type:
                case 'COS':
                    self.user_interactions = (
                            rating_df.groupby(user_id_column)[candidate_id_column].apply(
                                lambda x: [self.item_map[a] for a in x if a in self.item_map]
                            ).to_dict()
                        )
                    self.candidate_item,self.test_candidate_item = train_test_split(self.candidate_item,train_test_split_perc)
                    
                    # extract test item indices (column indices)
                    test_item_idx = set(self.test_candidate_item.nonzero()[1])

                    # filter interactions: keep only test items
                    self.test_interactions = {
                        user: [i for i in items if i in test_item_idx]
                        for user, items in self.user_interactions.items()
                    }

                case _ :
                    vals = vals.flatten()
                    
                    self.user_item = csr_matrix(
                        (vals, (rows, cols)),
                        shape=(len(self.user_map), len(self.item_map))
                    )
                    self.user_item,self.test_user_item = train_test_split(self.user_item,train_test_split_perc)





        # Initialize model
        match model_type:
            case 'ALS':
                self.model = implicit.als.AlternatingLeastSquares(
                    factors=factors,
                    regularization=1,
                    iterations=10,
                    num_threads=0,  
                    use_gpu=False
            )
            case 'BAY':
                self.model = implicit.bpr.BayesianPersonalizedRanking(
                    factors=self.factors,
                    learning_rate=0.05,
                    regularization=0.002,
                    iterations=50
                )
            case 'COS':
                self.model = implicit.nearest_neighbours.CosineRecommender(
                    K=5
                )
            



    def train(self):
        """Train the ALS model on implicit feedback."""
        
        # print("Matrix shape:", self.user_item.shape)
        # print("Model expects items:", self.model.item_factors.shape if hasattr(self.model, "item_factors") else "not trained yet")

        print(f"Training {self.model_type} model...")

        match self.model_type:
            case 'ALS':
                self.model.fit(self.user_item.T)

            case 'BAY':
                self.model.fit(self.user_item)
            
            case 'COS':
                self.model.fit(self.candidate_item)

        print("After fit:")
        # print(" item_factors:", self.model.item_factors.shape)
        # print(" user_factors:", self.model.user_factors.shape)
        # print("user_item shape:", self.user_item.shape)
        print("Training complete.\n")

        if self.add_embeding_vector and self.embedding_type == 'ALPHA_BLEND':
            # Embedding add
            embedding_cols = [c for c in self.candidate_df.columns if c!=self.candidate_id_column]

            if embedding_cols:
                print(f"Found {len(embedding_cols)} embedding columns, merging collaborative and content embeddings...")

                # Align candidates to model’s item order
                item_order = [self.rev_item_map[i] for i in range(len(self.rev_item_map))]
                content_embs = self.candidate_df.set_index(self.candidate_df[self.candidate_id_column].astype(str))
                content_embs = content_embs.loc[item_order, embedding_cols].values

                # Normalize embeddings
                content_embs = normalize(content_embs)
                cf_embs = normalize(self.model.item_factors)

                
                if self.use_pca:  
                    print('using PCA to match dimentionnality')                  
                    content_embs_pca = PCA(n_components=cf_embs.shape[1]).fit_transform(content_embs)
                    self.model.item_factors = self.alpha * cf_embs + (1 - self.alpha) * content_embs_pca
                else:
                    print('crop match dimentionnality')        
                    # Match dimensionality
                    if content_embs.shape[1] != cf_embs.shape[1]:
                        min_dim = min(content_embs.shape[1], cf_embs.shape[1])
                        content_embs = content_embs[:, :min_dim]
                        cf_embs = cf_embs[:, :min_dim]

                    # Blend collaborative and content embeddings
                    self.model.item_factors = self.alpha * cf_embs + (1 - self.alpha) * content_embs

    def build_interaction_matrix(self, rating_columns, weights=None):
        """Build interaction matrix using multiple rating columns"""
        if weights is None:
            weights = [1.0] * len(rating_columns)
        
        # Combine multiple rating signals
        combined_vals = np.zeros(len(self.rating_df))
        for i, col in enumerate(rating_columns):
            if col in self.rating_df.columns:
                vals = self.rating_df[col].fillna(0).values
                # Normalize and weight each signal
                if np.max(vals) > 0:
                    vals = vals / np.max(vals)
                combined_vals += weights[i] * vals
        
        # Use the combined values instead of just one column
        return combined_vals

    def predict(self, context, model_input):
        user_ids = model_input["user_id"].tolist()
        N = model_input.get("N", 10) if "N" in model_input else 10
        results = [self.recommend(uid, N) for uid in user_ids]
        return results

    def recommend(self, user_id, N=10):
        """USER-ITEM recommendations"""
        if user_id not in self.user_map:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_idx = self.user_map[user_id]
        user_row = self.user_item[user_idx, :]
        ids, scores = self.model.recommend(user_idx, 
                                           user_row, 
                                           N=N)

        results = [
            {"article_id": self.matrix_idx_to_article_id[i], "score": float(s)}
            for i, s in zip(ids, scores)
        ]
        return results
    
    def recommend_cos(self, user_id, N=5
                        ):

        if user_id not in self.user_interactions:
            raise ValueError("User not found in training data")

        seen_items = self.user_interactions[user_id]

        # Combine similar items from all items the user interacted with
        scores = {}

        for item in seen_items:
            similar = self.model.similar_items(item, N + 1)  # includes the item itself

            for other_item, score in similar:
                if other_item not in seen_items:
                    scores[other_item] = max(scores.get(other_item, 0), score)

        # Sort by similarity score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:N]

        # Map back to article IDs
        return [(self.rev_item_map[i], score) for i, score in ranked]
    

    def similar_items(self, article_id, N=10):
        """ITEM-ITEM recommendations"""
        if article_id not in self.item_map:
            raise ValueError(f"Unknown article_id: {article_id}")

        item_idx = self.item_map[article_id]
        similar = self.model.similar_items(item_idx, N=N)

        results = [
            {"article_id": self.rev_item_map[i], "score": float(score)}
            for i, score in similar
        ]
        return results


def precompute_model_results(model, N=5):
    """
    Precompute recommendations for all known users
    Returns: dict {user_id: list of recommended article_ids}
    """
    precomputed_results = {}
    
    for user_id in model.user_map.keys():
        try:
            recommendations = model.recommend(user_id, N=N)
            precomputed_results[user_id] = [rec["article_id"] for rec in recommendations]
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            precomputed_results[user_id] = []
    
    return precomputed_results

def precompute_model_results_cos( model : ArticleRetrievalImplicit,
                                      cible = 'ALL',
                                      N=5,

                                     ):
        recommendations = {} 
        match cible:
            case 'TEST':
                interaction = model.test_interactions
            case _:
                interaction = model.user_interactions
            

        for user,v in interaction.items():
            recs = model.recommend_cos(
                user, N
            )
            recommendations[user] = [item_id for item_id, _ in recs]

        return recommendations

def content_based_evaluation(pipeline:ArticleRetrievalImplicit, k=5 ):
    results = {
        "precision": [],
        "recall": [],
        "map": [],
        "ndcg": [],
    }
    def precision_at_k(recommended, ground_truth, k):
        if len(recommended) == 0:
            return 0
        recommended_k = recommended[:k]
        return len(set(recommended_k) & set(ground_truth)) / k
    
    def recall_at_k(recommended, ground_truth, k):
        recommended_k = recommended[:k]
        if len(ground_truth) == 0:
            return 0
        return len(set(recommended_k) & set(ground_truth)) / len(ground_truth)
    
    def ndcg_at_k(recommended, ground_truth, k):
        recommended_k = recommended[:k]
        def dcg(items):
            return sum(
                1 / np.log2(idx + 2)
                for idx, item in enumerate(items)
                if item in ground_truth
            )
        ideal_items = ground_truth[:k]
        return dcg(recommended_k) / (dcg(ideal_items) or 1)
    
    def average_precision_at_k(recommended, ground_truth, k):
        score = 0.0
        hits = 0

        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                hits += 1
                score += hits / (i + 1)

        if not ground_truth:
            return 0

        return score / min(len(ground_truth), k)
        
    recommendations = precompute_model_results_cos(pipeline,cible = 'TEST',N=k)

    for user, recs in recommendations.items():
        truth = pipeline.test_interactions.get(user, [])

        results["precision"].append(precision_at_k(recs, truth, k))
        results["recall"].append(recall_at_k(recs, truth, k))
        results["map"].append(average_precision_at_k(recs, truth, k))
        results["ndcg"].append(ndcg_at_k(recs, truth, k))

    return {metric: np.mean(values) for metric, values in results.items()}


def implicit_evaluation(pipeline:ArticleRetrievalImplicit, K=5):
    """Evaluate using implicit library metrics"""
    model = pipeline.model
    train_user_item = pipeline.user_item
    test_user_item = pipeline.test_user_item
    
    metrics = {}
    # Align test matrix user count
    n_users = model.user_factors.shape[0]
    test_user_item = test_user_item[:n_users, :]
    
    # 1. Precision@K and Recall@K
    precision = precision_at_k(
        model, train_user_item, test_user_item, K=K,
        show_progress=True
    )
    metrics[f'Precision_at_{K}'] = precision
    
    # 2. Mean Average Precision (MAP)
    map_score = mean_average_precision_at_k(
        model, train_user_item, test_user_item, K=K,
        show_progress=True
    )
    metrics[f'MAP_at_{K}'] = map_score
    
    # 3. NDCG@K - considers ranking position
    ndcg = ndcg_at_k(
        model, train_user_item, test_user_item, K=K,
        show_progress=True
    )
    metrics[f'NDCG_at_{K}'] = ndcg
    
    # 4. AUC@K - area under ROC curve
    auc = AUC_at_k(
        model, train_user_item, test_user_item, K=K,
        show_progress=True
    )
    metrics[f'AUC_at_{K}'] = auc
    
    return metrics

if __name__ == '__main__':
    import pickle
    RATING_PREPROC_DF_PATH="Training/Datas/rating_preprocess_df.pkl"
    CANDIDATE_PREPROC_DF_PATH='Training/Datas/candidate_preprocess_df.pkl'

    with open(RATING_PREPROC_DF_PATH , 'rb') as f:
        rating_df = pickle.load(f)
    with open(CANDIDATE_PREPROC_DF_PATH , 'rb') as f:
        candidate_df = pickle.load(f)

    pipe_params = {
    'rating_df' : rating_df,
    'candidate_df' : candidate_df,
    'ratings_keep' : ["user_id","article_id","time_per_word","category_id"],
    'candidates_keep' : ["article_id"],                      
    'rating_target' : None,
    'train_test_split_perc' : 0.8,
    'add_embeding_vector': False,
    'use_pca' : False,
    'embeding_alpha':0,
    'embedding_type' : 'USER_MEAN',
    'model_type':'ALS'
    
    }


    pipe = ArticleRetrievalImplicit(**pipe_params)
    pipe.train()
    precompute_model_results_cos(pipe)

    pass