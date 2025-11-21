import mlflow
import time
import subprocess

def start_local_experiment( host='127.0.0.1',
                            port='8080',
                            uri=r'/mlruns',
                            experiment_name="ArticleRecommendator"
                            ):
    command = f'''mlflow server --host {host}  --port {port} \n
                mlflow ui --backend-store-uri {uri}'''
    print(command)

    result = subprocess.Popen(command, shell=True)

    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

    mlflow.set_experiment(experiment_name)

from model_evaluator import RecommendationModelEvaluator
from implicit_model import ArticleRetrievalImplicit
from implicit_model import implicit_evaluation

def mlflow_experiment(rating_df ,
                      candidate_df,
                      model_class = ArticleRetrievalImplicit,
                      ratings_keep= ["user_id","article_id","time_per_word"],
                      candidates_keep = ["article_id"],                      
                      rating_target = ['time_per_word'],
                      train_test_split_perc = 0.8,                      
                      evaluation_type = 'IMPLICIT' ,
                      log_model = False,
                      
                      **kwargs):
    start_time = time.time()
    # convert id to str
    candidate_df = candidate_df.reset_index()

    rating_df = rating_df[ratings_keep]
    # candidate_df = candidate_df[candidates_keep]

    if kwargs.get('sample',False):
        rating_df = rating_df[rating_df['user_id'].isin(rating_df['user_id'].sample(n=kwargs['sample']))]

    if kwargs.get('seen_candidate_only',False):
        seen_candidate = rating_df['article_id'].unique()
        candidate_df = candidate_df[candidate_df['article_id'].isin(seen_candidate)]

    model = model_class(rating_df=rating_df,
                                     candidate_df=candidate_df,
                                     rating_target=rating_target,
                                     train_test_split_perc = train_test_split_perc,
                                     embeding_alpha= kwargs.get('embeding_alpha',0.8),
                                     factors=kwargs.get('factors',64),
                                     add_embeding_vector=kwargs.get('add_embeding_vector',False),
                                     model_type=kwargs.get('model_type','BAY'),
                                     use_pca=kwargs.get('use_pca',False),
                                     embedding_type=kwargs.get('embedding_type','ALPHA_BLEND'),

                                     )  
    
    model.train()

    match evaluation_type:
        case 'IMPLICIT':
            evaluation_dic = implicit_evaluation(model)
        case 'CB':
            model_evaluator = RecommendationModelEvaluator(model)       
            evaluation_dic  = model_evaluator.evaluate_model(model)


    process_time = time.time() - start_time


    with mlflow.start_run() as run:

        for k,v in evaluation_dic.items():
            mlflow.log_metric(k, v)


        signature = None
        params_dic = {}
        params_dic["Process_Time"] = process_time
        params_dic["train_test_split_perc"] = train_test_split_perc
        params_dic["ratings_keep"] = ratings_keep
        params_dic["candidates_keep"] = candidates_keep
        params_dic["rating_target"] = rating_target
        params_dic['rating_length'] = rating_df.shape[0] 
        params_dic['candidate_length'] = candidate_df.shape[0] 
        
        for k,v in kwargs.items():
            params_dic[k] = v

        if log_model:
            model_info  = mlflow.pyfunc.log_model(
                                        python_model=model,        
                                        name=model.name,
                                        signature=signature,
                                        input_example=None,
                                        # registered_model_name=f"{model.name}",
                                        )

        # Log other information about the model
        mlflow.log_params(params_dic)
    return model


if __name__ == '__main__':
    start_local_experiment()