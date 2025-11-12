import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from sklearn.preprocessing import normalize
from implicit.evaluation import precision_at_k,mean_average_precision_at_k,AUC_at_k,ndcg_at_k,train_test_split
from mlflow import pyfunc

class ArticleRetrievalImplicit(pyfunc.PythonModel):
    """
    Implicit retrieval model
    """

    def __init__(self, rating_df, candidate_df,
                  rating_target='time_per_word',
                  user_id_column = 'user_id',
                  candidate_id_column = "article_id",
                  train_test_split_perc = None,
                  factors=64, 
                  model_type='BAY',                  
                  add_embeding_vector = False,                  
                  embeding_alpha=0.7,
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

        self.candidate_id_column = candidate_id_column

        self.add_embeding_vector = add_embeding_vector

        # Deduplicate ratings for single article by summing rating
        rating_df = (
            rating_df.groupby([user_id_column, candidate_id_column], as_index=False)
            .agg({rating_target: "sum"})
        )

        #Build mappings 
        self.user_map = {u: i for i, u in enumerate(rating_df[user_id_column].astype(str).unique())}
        self.item_map = {a: i for i, a in enumerate(rating_df[candidate_id_column].astype(str).unique())}
        self.rev_user_map = {v: k for k, v in self.user_map.items()}
        self.rev_item_map = {v: k for k, v in self.item_map.items()}

        # Align candidate_df to those same item IDs
        # self.candidate_df = self.candidate_df.loc[
        #     self.candidate_df.index.astype(str).isin(self.item_map.keys())
        # ].copy()

        # Map matrix column index back to article_id
        self.matrix_idx_to_article_id = {idx: article_id for article_id, idx in self.item_map.items()}


        # Build interaction matrix
        self.rating_df["user_idx"] = self.rating_df[user_id_column].astype(str).map(self.user_map)
        self.rating_df["item_idx"] = self.rating_df[candidate_id_column].astype(str).map(self.item_map)

        rows = self.rating_df["user_idx"].values
        cols = self.rating_df["item_idx"].values
        vals = self.rating_df.get(rating_target, pd.Series(1, index=self.rating_df.index)).values

        # ensure all values are positive
        vals = np.maximum(vals, 0)

        # rescale for als
        vals = vals / (np.max(vals) + 1e-6)

        self.user_item = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_map), len(self.item_map))
        )

        if type(train_test_split_perc)!=type(None):
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

    def train(self):
        """Train the ALS model on implicit feedback."""
        
        print("Matrix shape:", self.user_item.shape)
        # print("Model expects items:", self.model.item_factors.shape if hasattr(self.model, "item_factors") else "not trained yet")

        print("Training ALS model...")

        match self.model_type:
            case 'ALS':
                self.model.fit(self.user_item.T)

            case 'BAY':
                self.model.fit(self.user_item)

        print("After fit:")
        # print(" item_factors:", self.model.item_factors.shape)
        # print(" user_factors:", self.model.user_factors.shape)
        print("user_item shape:", self.user_item.shape)

        if self.add_embeding_vector:
            # Embedding add
            embedding_cols = [c for c in self.candidate_df.columns if c.startswith("emb_")]

            if embedding_cols:
                print(f"Found {len(embedding_cols)} embedding columns, merging collaborative and content embeddings...")

                # Align candidates to model’s item order
                item_order = [self.rev_item_map[i] for i in range(len(self.rev_item_map))]
                content_embs = self.candidate_df.set_index(self.candidate_df[self.candidate_id_column].astype(str))
                content_embs = content_embs.loc[item_order, embedding_cols].values

                # Normalize embeddings
                content_embs = normalize(content_embs)
                cf_embs = normalize(self.model.item_factors)

                # Match dimensionality
                if content_embs.shape[1] != cf_embs.shape[1]:
                    min_dim = min(content_embs.shape[1], cf_embs.shape[1])
                    content_embs = content_embs[:, :min_dim]
                    cf_embs = cf_embs[:, :min_dim]

                # Blend collaborative and content embeddings
                self.model.item_factors = self.alpha * cf_embs + (1 - self.alpha) * content_embs

        print("Training complete.\n")

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

def implicit_evaluation(pipeline:ArticleRetrievalImplicit, K=10):
    """Evaluate using implicit library metrics"""
    model = pipeline.model
    train_user_item = pipeline.user_item
    test_user_item = pipeline.test_user_item
    
    metrics = {}
    
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