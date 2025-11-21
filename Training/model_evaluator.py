
import random
import pandas as pd
import numpy as np

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class RecommendationModelEvaluator:

    def __init__(self,model):#full_interaction_df,test_interaction_df,train_interaction_df,all_items
        self.full_interaction_df = model.user_interactions
        self.test_interaction_df = model.interactions_test_df
        self.train_interaction_df = model.interactions_train_df
        self.all_items = set(model.item_ids) 

        self.items_interacted_cache = {
            uid: set(df[model.candidate_id_column])
            for uid, df in self.full_interaction_df.groupby(model.user_id_column)
        }

        pass

    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted = self.items_interacted_cache[user_id]
        non_interacted = list(self.all_items - interacted)
        random.seed(seed)
        return set(random.sample(non_interacted, sample_size))
        # interacted_items = get_items_interacted(user_id, self.full_interaction_df)
        # #all_items = set(articles_df[model.candidate_id_column])
        # non_interacted_items = sorted(self.all_items - interacted_items)

        # random.seed(seed)
        # non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        # return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn): 
            idx = np.where(recommended_items == item_id)[0]
            if len(idx) == 0:
                return 0, -1
            index = int(idx[0])
            return int(index < topn), index       
            # try:
            #     index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            # except:
            #     index = -1
            # hit = int(index in range(0, topn))
            # return hit, index

    def evaluate_model_for_user(self, model, user_id, k = 5):
        user_id = int(user_id)
        #Getting the items in test set
        interacted_values_testset = self.test_interaction_df.loc[
                                            self.test_interaction_df[model.user_id_column] == user_id
                                            ]

        if type(interacted_values_testset[model.candidate_id_column]) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset[model.candidate_id_column])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset[model.candidate_id_column])])  
            
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        rec_ids = model.recommend_items(user_id, 
                                        topn=100)
        rec_ids_k = rec_ids[:k]


        hits_at_k_count = 0
 
        #NEGATIVE SAMPLING EVALUATION
        for item_id in person_interacted_items_testset:
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            mask = np.isin(rec_ids, list(items_to_filter_recs))
            valid_recs = rec_ids[mask]

            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_k, index_at_k = self._verify_hit_top_n(item_id, valid_recs, k)
            hits_at_k_count += hit_at_k


        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_k = hits_at_k_count / float(interacted_items_count_testset)

        hits_in_top_k = len(set(rec_ids_k) & person_interacted_items_testset)
        precision_at_k = hits_in_top_k / float(k)



        person_metrics = {'hits@k_count':hits_at_k_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@k': recall_at_k,
                          'precision@k': precision_at_k,
                          }
        return person_metrics

    def faiss_evaluate_model_for_user(self, model, user_id):
        user_id = int(user_id)
        
        # Get test items
        test_items = self.test_interaction_df.loc[
            self.test_interaction_df[model.user_id_column] == user_id,
            model.candidate_id_column
        ]
        test_items_set = set(test_items if isinstance(test_items, pd.Series) else [int(test_items)])
        
        # Find precomputed FAISS row
        # user_idx = np.where(model._faiss_user_list == user_id)[0][0]
        rec_ids = model._faiss_idx[user_id]
        rec_sims = model._faiss_sims[user_id]
        
        hits_at_5 = 0
        hits_at_10 = 0
        
        for item_id in test_items_set:
            # Sample non-interacted items
            non_interacted_sample = self.get_not_interacted_items_sample(user_id, 
                                                                        EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                        seed=item_id % (2**32))
            items_to_check = non_interacted_sample | {item_id}
            
            # Filter precomputed recommendations to items_to_check
            mask = np.isin(rec_ids, list(items_to_check))
            valid_recs = rec_ids[mask]
            
            # Compute hits
            hit5, _ = self._verify_hit_top_n(item_id, valid_recs, 5)
            hit10, _ = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_5 += hit5
            hits_at_10 += hit10
        
        n_test_items = len(test_items_set)
        return {
            'hits@5_count': hits_at_5,
            'hits@10_count': hits_at_10,
            'interacted_count': n_test_items,
            'recall@5': hits_at_5 / n_test_items,
            'recall@10': hits_at_10 / n_test_items
        }

    def evaluate_model(self, model, topn=100,batch_size = 20000,k=5):
        if hasattr(model, '_faiss_idx'):
            func =  self.faiss_evaluate_model
        else:
            func = self.slow_evaluate_model

        result = func(model, batch_size = batch_size, k=k)
        return result

    def faiss_evaluate_model(self, model, topn=100,batch_size = 20000,k=5):
        """
        Vectorized evaluation using precomputed FAISS recommendations.
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        import pandas as pd

        # Prepare users
        users_list = np.array(self.test_interaction_df[model.user_id_column].unique())
        user_to_idx = {uid: i for i, uid in enumerate(users_list)}
        item_id_to_idx = {id: idx for idx, id in enumerate(model.item_ids)}
        n_users = len(users_list)
    
        # Map test interactions to row/col indices for sparse matrix
        rows = self.test_interaction_df[model.user_id_column].map(user_to_idx).to_numpy()
        # cols = self.test_interaction_df[model.candidate_id_column].to_numpy() OLD ASSUMING ROW ID alignement

        cols = (
                self.test_interaction_df[model.candidate_id_column]
                .map(item_id_to_idx)
                .to_numpy()
            )
        data = np.ones_like(rows)
        test_interaction_mask = csr_matrix((data, (rows, cols)), shape=(n_users, len(model.item_ids)))

        hits_at_k_list,recall_at_k_list, precision_at_k_list, ndcg_at_k_list, auc_at_k_list, mpa_at_k_list = [], [], [], [], [],[]
        # hits_at_10_list = []
        n_test_items_list = []

        for start in range(0, n_users, batch_size):
            print(f'start new batch batch at {start} on {n_users}' )
            end = min(start + batch_size, n_users)
            batch_users = users_list[start:end]
            batch_idx = np.arange(start, end)

            # Build FAISS recommendation matrix for this batch
            #uidx = np.where(self._user_list == user_id)[0][0]
            batch_rec = np.vstack([model._faiss_idx[model._user_to_idx[uid]] for uid in batch_users])

            # Compute hits in sparse way
            hits_batch = np.array([
                test_interaction_mask[i, batch_rec[i]].toarray().ravel()
                for i in range(len(batch_idx))
            ])

            
            # hits_at_10_list.append(hits_batch[:, :10].sum(axis=1))
            n_relevant = test_interaction_mask[batch_idx].sum(axis=1).A1  # shape: (batch_size,)
            n_relevant[n_relevant == 0] = 1  # avoid division by zero

            # Recall@k
            hits_k = hits_batch[:, :k]
            recall_batch = hits_k.sum(axis=1) / n_relevant
            recall_at_k_list.append(recall_batch)

            # Precision@k
            precision_batch = hits_k.sum(axis=1) / k
            precision_at_k_list.append(precision_batch)

            # NDCG@k
            discounts = 1.0 / np.log2(np.arange(2, k + 2))
            dcg_batch = (hits_k * discounts).sum(axis=1)
            ideal_hits = np.minimum(n_relevant, k)
            idcg_batch = np.array([np.sum(1.0 / np.log2(np.arange(2, ih + 2))) for ih in ideal_hits])
            ndcg_batch = dcg_batch / (idcg_batch + 1e-8)
            ndcg_at_k_list.append(ndcg_batch)

            # AUC@k (pairwise fraction among top-k)
            # In top-k ranking, each hit is ranked above remaining non-hits
            hits =  hits_k.astype(float)
            n_pos = hits.sum(axis=1)
            n_neg = k - n_pos
            rank_positions = np.arange(1, k + 1)
            cum_neg = np.cumsum(1 - hits, axis=1)
            # each positive contributes (#neg below it)
            correct_pairs = ((hits * (n_neg[:, None] - cum_neg + (1 - hits))) ).sum(axis=1)
            auc_at_k_batch = correct_pairs / np.clip(n_pos * n_neg, 1, None)  # avoid div0
            auc_at_k_list.append(auc_at_k_batch)

            # MPA@k (Mean percentile of hits)
            ranks = np.tile(np.arange(1, k + 1), (len(batch_idx), 1))
            mpa_batch = ((hits_k * (1 - (ranks - 1) / k)).sum(axis=1)) / np.clip(hits_k.sum(axis=1), 1, None)
            mpa_at_k_list.append(mpa_batch)

        # Concatenate all batches
        recall_at_k_all = np.concatenate(recall_at_k_list)
        precision_at_k_all = np.concatenate(precision_at_k_list)
        ndcg_at_k_all = np.concatenate(ndcg_at_k_list)
        auc_at_k_all = np.concatenate(auc_at_k_list)
        mpa_at_k_all = np.concatenate(mpa_at_k_list)

        # Build detailed DataFrame
        # detailed_results_df = pd.DataFrame({
        #     '_user_id': users_list,
        #     'interacted_count': test_interaction_mask.sum(axis=1).A1,
        #     'recall_at_k': recall_at_k_all,
        #     'precision_at_k': precision_at_k_all,
        #     'ndcg_at_k': ndcg_at_k_all,
        #     'auc_at_k': auc_at_k_all,
        #     'mpa_at_k': mpa_at_k_all
        # })

        # Global metrics
        global_metrics = {

            f'recall_at_{k}': recall_at_k_all.mean(),
            f'precision_at_{k}': precision_at_k_all.mean(),
            f'NDCG_at_{k}': ndcg_at_k_all.mean(),
            # f'AUC_at_{k}': auc_at_k_all.mean(),
            f'MAP_at_{k}': mpa_at_k_all.mean()
        }

        return global_metrics#, detailed_results_df

    def slow_evaluate_model(self, model,batch_size = 20000, k = 5):
        #print('Running evaluation for users')
        people_metrics = []
        users_list = list(self.test_interaction_df[model.user_id_column].unique())
        for idx, user_id in enumerate(users_list):
            if idx % 100 == 0 and idx > 0:
               print(f'{idx} users processed on {len(users_list)}')
            person_metrics = self.evaluate_model_for_user(model, user_id)  
            person_metrics['_user_id'] = user_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_k = detailed_results_df['hits@k_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_precision_at_k =  detailed_results_df['precision@k'].mean()
        
        global_metrics = {
                          f'recall_at_{k}': global_recall_at_k,
                          f'precision_at_{k}' : global_precision_at_k
                          }    
        return global_metrics
    
    def get_items_interacted(self,user_id,):
        return self.items_interacted_cache.get(user_id, set())

#  
if __name__ == '__main__':
    import pickle
    # CB_MODEL_PATH = 'Training/model_assets/cb_model.pkl'
    # with open(CB_MODEL_PATH, 'rb') as f:
    #     model = pickle.load(f)
    from popularity_model import PopularityArticleRecommender
    POP_MODEL_PATH = 'Training/model_assets/pop_model.pkl'
    with open(POP_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    model_evaluator = RecommendationModelEvaluator(model)   
    print('Evaluating Content-Based Filtering model...')
    cb_global_metrics = model_evaluator.faiss_evaluate_model(model)
    print('\nGlobal metrics:\n%s' % cb_global_metrics)
