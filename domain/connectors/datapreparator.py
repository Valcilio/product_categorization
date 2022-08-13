import json
import joblib
import pandas as pd

from domain.use_cases.featurederivator import FeatureDerivator
from domain.use_cases.rescaler import Rescaler

class DataPreparator():

    def __init__(self, json_df: json):

        self.json_df = json_df

        self.category_scaler         = joblib.load(open('scalers/label-encoder_category_scaler.pkl', 'rb'))
        self.express_delivery_scaler = joblib.load(open('scalers/min-max_express_delivery_scaler.pkl', 'rb'))
        self.minimum_quantity_scaler = joblib.load(open('scalers/min-max_minimum_quantity_scaler.pkl', 'rb'))
        self.page_position_scaler    = joblib.load(open('scalers/min-max_page_position_scaler.pkl', 'rb'))
        self.position_scaler         = joblib.load(open('scalers/min-max_position_scaler.pkl', 'rb'))
        self.price_per_weight_scaler = joblib.load(open('scalers/min-max_price_per_weight_scaler.pkl', 'rb'))
        self.price_scaler            = joblib.load(open('scalers/min-max_price_scaler.pkl', 'rb'))
        self.view_counts_scaler      = joblib.load(open('scalers/min-max_view_counts_scaler.pkl', 'rb'))
        self.week_of_year_scaler     = joblib.load(open('scalers/min-max_week_of_year_scaler.pkl', 'rb'))
        self.weight_scaler           = joblib.load(open('scalers/min-max_weight_scaler.pkl', 'rb'))

    def preparate_data(self):
        '''Run all data preparation methods'''

        self._structure_json()
        self._run_feature_derivator()
        self._run_rescaler()

        return self.df_pred

    def return_product_id(self):
        '''Return product_id'''

        self._structure_json()

        return self.df_pred['product_id']

    def reverse_category(self, df_pred: pd.DataFrame):
        '''Reverse category in numeric format to strings'''

        self._structure_json()
        df_pred = pd.DataFrame(df_pred)
        df_pred['category'] = self._rescaler().inverse_category(df_pred)

        return df_pred.drop('label-encoder_category', axis=1)

    def _run_rescaler(self):
        '''Rescale all variables for modeling'''

        self.df_pred = self._rescaler().run()

    def _rescaler(self):
        '''Definy rescaler'''

        rescaler = Rescaler(df=self.df_pred,
                            position_scaler=self.position_scaler,
                            price_scaler=self.price_scaler,
                            weight_scaler=self.weight_scaler,
                            minimum_quantity_scaler=self.minimum_quantity_scaler,
                            view_counts_scaler=self.view_counts_scaler,
                            express_delivery_scaler=self.express_delivery_scaler,
                            page_position_scaler=self.page_position_scaler,
                            price_per_weight_scaler=self.price_per_weight_scaler,
                            week_of_year_scaler=self.week_of_year_scaler,
                            category_scaler=self.category_scaler)

        return rescaler      

    def _run_feature_derivator(self):
        '''Derivate the ipva_dono and best_offer variables'''

        feature_derivator = FeatureDerivator(df=self.df_pred)
        self.df_pred = feature_derivator.get_features()

    def _structure_json(self):
        '''Structure JSON for prediction'''

        json_df = self.json_df['data']
        self.df_pred = pd.read_json(json_df)