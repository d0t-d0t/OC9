import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from recommenders.models.sar import SAR
from recommenders.models.ncf.ncf_singlenode import NCF
# from recommenders.models.deeprec import GraphDR
# from recommenders.datasets import pandas_df
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, ndcg_at_k, map_at_k
from mlflow import pyfunc
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

class ArticleRetrievalRecommenders(pyfunc.PythonModel):
    """
    Recommenders library-based retrieval model
    """

    def __init__(self, rating_df, candidate_df,
                  rating_targets=['time_per_word'],
                  user_id_column='user_id',
                  candidate_id_column="article_id",
                  train_test_split_perc=None,
                  factors=64,
                  model_type='SAR',
                  add_embeding_vector=False,
                  use_pca=False,
                  embedding_type='ALPHA_BLEND',
                  embeding_alpha=0.7,
                  ):
        """
        rating_df: DataFrame with columns
        candidate_df: DataFrame indexed by article_id
        factors: latent factor dim
        """

        self.name = f'ArticleRetrievalRecommenders_{model_type}'
        self.rating_df = rating_df.copy()
        self.candidate_df = candidate_df.copy()
        self.factors = factors
        self.alpha = embeding_alpha
        self.model_type = model_type
        self.use_pca = use_pca
        self.candidate_id_column = candidate_id_column
        self.add_embeding_vector = add_embeding_vector
        self.embedding_type = embedding_type
        self.user_id_column = user_id_column

        if add_embeding_vector and self.embedding_type == 'USER_MEAN':
            print('adding embedding on user side')
            embedding_cols = [c for c in self.candidate_df.columns if c != self.candidate_id_column]
            rating_targets += embedding_cols
            self.rating_df = self.rating_df.merge(
                candidate_df[[candidate_id_column] + embedding_cols],
                on=candidate_id_column,
                how="left"
            )

        # Build mappings
        self.user_map = {u: i for i, u in enumerate(rating_df[user_id_column].astype(str).unique())}
        self.item_map = {a: i for i, a in enumerate(rating_df[candidate_id_column].astype(str).unique())}
        self.rev_user_map = {v: k for k, v in self.user_map.items()}
        self.rev_item_map = {v: k for k, v in self.item_map.items()}

        # Prepare data for recommenders
        self.rating_df["user_idx"] = self.rating_df[user_id_column].astype(str).map(self.user_map)
        self.rating_df["item_idx"] = self.rating_df[candidate_id_column].astype(str).map(self.item_map)

        # Build interaction data
        self.train_data = self.build_interaction_data(rating_targets)

        if train_test_split_perc is not None:
            from sklearn.model_selection import train_test_split
            self.train_data, self.test_data = train_test_split(
                self.train_data, test_size=train_test_split_perc, random_state=42
            )
        else:
            self.test_data = None

        # Initialize model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate recommender model"""
        if self.model_type == 'SAR':
            return SAR(
                col_user="user_idx",
                col_item="item_idx",
                col_rating="rating",
                col_timestamp=None,
                similarity_type="jaccard",
                time_decay_coefficient=30,
                time_now=None,
                timedecay_formula=True,
                normalize=True
            )
        elif self.model_type == 'NCF':
            return NCF(
                n_users=len(self.user_map),
                n_items=len(self.item_map),
                model_type="NeuMF",
                n_factors=self.factors,
                layer_sizes=[64, 32, 16],
                n_epochs=20,
                batch_size=256,
                learning_rate=0.001,
                verbose=1
            )
        elif self.model_type == 'GraphDR':
            return GraphDR(
                n_users=len(self.user_map),
                n_items=len(self.item_map),
                n_factors=self.factors,
                embedding_dim=64,
                hidden_dim=64,
                n_layers=3,
                dropout=0.2,
                n_epochs=50,
                batch_size=512,
                learning_rate=0.001,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def build_interaction_data(self, rating_columns, weights=None):
        """Build interaction data using multiple rating columns"""
        if weights is None:
            weights = [1.0] * len(rating_columns)
        
        # Start with base dataframe
        interaction_data = self.rating_df[['user_idx', 'item_idx']].copy()
        
        # Combine multiple rating signals
        combined_vals = np.zeros(len(self.rating_df))
        for i, col in enumerate(rating_columns):
            if col in self.rating_df.columns:
                vals = self.rating_df[col].fillna(0).values
                # Normalize and weight each signal
                if np.max(vals) > 0:
                    vals = vals / np.max(vals)
                combined_vals += weights[i] * vals
        
        interaction_data['rating'] = combined_vals
        
        # Filter out zero ratings for some models
        if self.model_type in ['SAR']:
            interaction_data = interaction_data[interaction_data['rating'] > 0]
        
        return interaction_data

    def train(self):
        """Train the recommender model"""
        print(f"Training {self.model_type} model...")
        print(f"Training data shape: {self.train_data.shape}")
        
        if self.model_type == 'SAR':
            self.model.fit(self.train_data)
        elif self.model_type in ['NCF', 'GraphDR']:
            self.model.fit(
                self.train_data, 
                eval_data=self.test_data,
                verbose=1
            )
        
        print("Training complete.")
        
        # Handle embedding blending for neural models
        if (self.add_embeding_vector and 
            self.embedding_type == 'ALPHA_BLEND' and 
            self.model_type in ['NCF', 'GraphDR'] and
            hasattr(self.model, 'item_embedding')):
            
            embedding_cols = [c for c in self.candidate_df.columns if c != self.candidate_id_column]
            
            if embedding_cols:
                print(f"Found {len(embedding_cols)} embedding columns, blending collaborative and content embeddings...")
                
                # Get collaborative embeddings
                cf_embs = self.model.item_embedding
                
                # Get content embeddings
                item_order = [self.rev_item_map[i] for i in range(len(self.rev_item_map))]
                content_embs = self.candidate_df.set_index(self.candidate_df[self.candidate_id_column].astype(str))
                content_embs = content_embs.loc[item_order, embedding_cols].values
                content_embs = normalize(content_embs)
                
                # Handle dimensionality matching
                if self.use_pca:
                    print('Using PCA to match dimensionality')
                    content_embs_pca = PCA(n_components=cf_embs.shape[1]).fit_transform(content_embs)
                    blended_embs = self.alpha * cf_embs + (1 - self.alpha) * content_embs_pca
                else:
                    print('Cropping to match dimensionality')
                    min_dim = min(content_embs.shape[1], cf_embs.shape[1])
                    content_embs = content_embs[:, :min_dim]
                    cf_embs_cropped = cf_embs[:, :min_dim]
                    blended_embs = self.alpha * cf_embs_cropped + (1 - self.alpha) * content_embs
                
                # Update model embeddings
                self.model.item_embedding = blended_embs

    def predict(self, context, model_input):
        """Predict recommendations for users"""
        user_ids = model_input["user_id"].tolist()
        N = model_input.get("N", 10) if "N" in model_input else 10
        
        results = []
        for uid in user_ids:
            try:
                recs = self.recommend(uid, N)
                results.append(recs)
            except ValueError as e:
                # Handle cold-start users
                print(f"Warning: {e}, returning popular items")
                results.append(self.get_popular_items(N))
        
        return results

    def recommend(self, user_id, N=10):
        """Generate recommendations for a specific user"""
        if user_id not in self.user_map:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_idx = self.user_map[user_id]
        
        if self.model_type == 'SAR':
            # SAR uses different recommendation interface
            user_data = pd.DataFrame({
                'user_idx': [user_idx],
                'user_id': [user_id]
            })
            recommendations = self.model.recommend_k_items(
                user_data, 
                top_k=N, 
                remove_seen=True
            )
            
            # Convert to our format
            results = []
            for _, row in recommendations.iterrows():
                article_id = self.rev_item_map[row['item_idx']]
                results.append({
                    "article_id": article_id,
                    "score": float(row['score'])
                })
                
        else:  # NCF, GraphDR
            # For neural models, get scores for all items
            user_indices = np.full(len(self.item_map), user_idx)
            item_indices = np.arange(len(self.item_map))
            
            scores = self.model.predict(user_indices, item_indices)
            
            # Get top N items
            top_indices = np.argsort(scores)[::-1][:N]
            
            results = [
                {
                    "article_id": self.rev_item_map[idx],
                    "score": float(scores[idx])
                }
                for idx in top_indices
            ]
        
        return results

    def get_popular_items(self, N=10):
        """Get popular items as fallback for cold-start"""
        item_popularity = self.train_data.groupby('item_idx')['rating'].sum()
        popular_items = item_popularity.sort_values(ascending=False).head(N)
        
        return [
            {
                "article_id": self.rev_item_map[idx],
                "score": float(score)
            }
            for idx, score in popular_items.items()
        ]

    def evaluate(self, metrics=['precision', 'recall', 'ndcg'], k=10):
        """Evaluate model performance"""
        if self.test_data is None:
            print("No test data available for evaluation")
            return None
        
        # For SAR, use built-in evaluation
        if self.model_type == 'SAR':
            from recommenders.evaluation.python_evaluation import ranking_metrics
            
            # Generate recommendations for all test users
            test_users = self.test_data['user_idx'].unique()
            user_data = pd.DataFrame({
                'user_idx': test_users
            })
            
            all_predictions = self.model.recommend_k_items(
                user_data, 
                top_k=k, 
                remove_seen=True
            )
            
            # Calculate metrics
            eval_results = ranking_metrics(
                rating_true=self.test_data,
                rating_pred=all_predictions,
                col_user='user_idx',
                col_item='item_idx',
                col_rating='rating',
                col_prediction='score',
                relevancy_method='top_k',
                k=k
            )
            
        else:
            # For neural models, use appropriate evaluation
            eval_results = self.model.evaluate(
                self.test_data,
                metrics=metrics,
                k=k
            )
        
        return eval_results