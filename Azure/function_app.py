import azure.functions as func
import logging
import os
import pickle
# import implicit
from typing import List, Dict
import json


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)#

class ArticleRetrievalService:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.precomp_dic = None
        self.is_model_loaded = False
        self.is_precomp_loaded = False
    
    def load_model(self):
        """Load the trained model and data"""
        try:
            # Get the path to model files
            model_path = os.path.join(os.path.dirname(__file__), 'model_assets', 'model.pkl')

            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)          

            self.is_model_loaded = True
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def load_precomp(self):
            """Load the trained model and data"""
            try:
                # Get the path to model files
                precomp_path = os.path.join(os.path.dirname(__file__), 'model_assets', 'precompute_dic.pkl')

                # Load model
                with open(precomp_path, 'rb') as f:
                    self.precomp = pickle.load(f)          

                self.is_precomp_loaded = True
                logging.info("Precomp loaded successfully")
                
            except Exception as e:
                logging.error(f"Error loading precomp: {str(e)}")
                raise
        
    
    def get_recommendations_by_model(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations for a user"""
        if not self.is_model_loaded:
            self.load_model()
        
        try:
            recommendations = self.model.recommend(user_id, n_recommendations)
            return recommendations
        except ValueError as e:
            if "Unknown user_id" in str(e):
                # Return popular items for new users
                # return self.get_popular_items(n_recommendations)
                return None
            raise
    
    def get_recommendations_by_precomp(self, user_id: str, n_recommendations: int = 5)-> List[Dict]:
        """Get recommendations for a user using precomp"""

        if not self.is_precomp_loaded:
            self.load_precomp()
        
        recommendations = self.precomp.get(user_id,['unk'])

        results = [
            {"article_id": i}
            for i in recommendations[:n_recommendations+1]
        ]
        return results

    
    # def get_similar_items(self, article_id: str, n_similar: int = 10) -> List[Dict]:
    #     """Get similar items"""
    #     if not self.is_model_loaded:
    #         self.load_model()
        
    #     return self.model.similar_items(article_id, n_similar)

    # def get_popular_items(self, n_items: int = 10) -> List[Dict]:
    #     """Get precompute popular items as fallback"""
    #     popular_items = self.candidate_df.head(n_items)
    #     return [
    #         {"article_id": str(row[self.model.candidate_id_column]), "score": 1.0}
    #         for _, row in popular_items.iterrows()
    #     ]

    
# Initialize the service

# article_service = None
article_service = ArticleRetrievalService()

# @app.function_name(name="RecommendArticles")
@app.route(route="recommend_articles")#, methods=["POST"], auth_level=func.AuthLevel.FUNCTION
def recommend_articles(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for article recommendations"""
    logging.info('Python HTTP trigger function processed a request.')
    
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON",
            status_code=400
        )
    
    user_id = req_body.get('user_id')
    n_recommendations = req_body.get('n_recommendations', 5)
    
    if not user_id:
        return func.HttpResponse(
            "Please provide a user_id in the request body",
            status_code=400
        )
    
    try:
        if type(article_service)!=type(None):
            recommendations = article_service.get_recommendations_by_precomp(user_id, n_recommendations)
        else:
            recommendations = 'failed'
        
        return func.HttpResponse(
            json.dumps({
                "user_id": user_id,
                "recommendations": recommendations,
                "count": len(recommendations)
            }),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="WTFiSGOinG10_Almost")
def WTFiSGOinG10_Almost(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )