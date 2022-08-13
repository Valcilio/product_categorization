import joblib
import pandas as pd

class ModelRunner():

    def __init__(self, df_prepared: pd.DataFrame):
        
        self.df_prepared = df_prepared
        self.class_model = joblib.load(open('models/xgb_class_model.pkl', 'rb'))

    def run_class_model(self):
        '''Run classification model'''

        yhat_class = pd.Series(self.class_model.predict(self.df_prepared))
        yhat_class = yhat_class.astype(int)
        yhat_class.name = 'label-encoder_category'

        return yhat_class