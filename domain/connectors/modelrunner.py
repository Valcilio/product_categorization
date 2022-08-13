import numpy as np
import joblib
import pandas as pd

class ModelsRunner():

    def __init__(self, df_prepared: pd.DataFrame):
        
        self.df_prepared = df_prepared.copy()
        self.class_model = joblib.load(open('models/xgb_class_model.pkl', 'rb'))
        self.reg_model = joblib.load(open('models/rf_reg_model.pkl', 'rb'))

    def run_class_model(self):
        '''Run classification model'''

        yhat_class = pd.DataFrame(self.class_model.predict_proba(self.df_prepared))[1]
        pred_class = yhat_class.apply(lambda x: 'no' if x <= 0.74 else 'yes')
        pred_class = pd.DataFrame(pred_class).rename(columns={1:'will_have_leads'})

        return pred_class

    def run_reg_model(self):
        '''Run regression model'''

        yhat_reg = np.expm1(self.reg_model.predict(self.df_prepared))
        pred_reg = pd.DataFrame(yhat_reg)
        pred_reg = pd.DataFrame(pred_reg).rename(columns={0:'number_of_leads'})
        pred_reg['number_of_leads'] = pred_reg['number_of_leads'].round()

        return pred_reg
