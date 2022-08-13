import numpy as np
import pandas as pd
import joblib
from   sklearn import preprocessing as pp

from .context import Context as c

class DataTransformer():

    def __init__(self, df: pd.DataFrame, **kwargs):

        self.df = df.copy()

    def derivate_int_float_columns(self, **kwargs):
        '''Filter dataset to just contain int64 and float64 columns'''

        df = self.df.copy()
        
        num_attributes = df.select_dtypes(include=['int64', 'float64'])

        return num_attributes

    def rescaling(self, y: str, df: pd.DataFrame, method: str = 'yeo-johnson', **kwargs):
        '''Rescale the column passed as "y" to a scale who was passed in "method" attribute
        and check if this transformation is correct with the method "test_rescale"'''

        df = df.copy()

        if method in ['box-cox', 'yeo-johnson']:
            df[f'{method}_{y}'], scaler = self._transfomer(df=df, scaler=pp.PowerTransformer(method=method), y=y)
        elif method == 'min-max':
            df[f'{method}_{y}'], scaler = self._transfomer(df=df, scaler=pp.MinMaxScaler(), y=y)
        elif method == 'robust-scaler':
            df[f'{method}_{y}'], scaler = self._transfomer(df=df, scaler=pp.RobustScaler(), y=y)
        elif method == 'label-encoder':
            df[f'{method}_{y}'], scaler = self._transfomer(df=df, scaler=pp.LabelEncoder(), y=y)
        elif method == 'log1p':
            scaler = 'log1p'
            df[f'{method}_{y}'], scaler = np.log1p(df[y])

        self._save_scaler(scaler=scaler, scaler_name= f'{method}_{y}' + '_scaler.pkl')
        df = df.drop(y, axis=1)

        return df, scaler

    def inverse_transformation(self, df: pd.DataFrame, col_orig_name: str, y_nt: str, scaler, **kwargs):
        '''Undo the rescaling based in the scaler passed'''

        df = pd.DataFrame(df).copy()
        df[f'{col_orig_name}'] = df[y_nt]
        df = df.drop(y_nt, axis=1)

        if scaler == 'log1p':
            df[f'{y_nt}_reversed'] = np.expm1(df[f'{col_orig_name}'])
        else:
            df[f'{y_nt}_reversed'] = scaler.inverse_transform(df[[f'{col_orig_name}']])

        return df.drop(col_orig_name, axis=1)    

    def reverse_concat_y(self, scaler, col_orig_name: str, y_nt: str, y_val: pd.Series, yhat: pd.Series, **kwargs):

        yhat_xgb_df = self.reverse_y(scaler=scaler, col_orig_name=col_orig_name, y_nt=0, y_val=yhat, new_name='yhat')
        y_val_df = self.reverse_y(scaler=scaler, col_orig_name=col_orig_name, y_nt=y_nt, y_val=y_val, new_name='y')
        y_df = pd.concat([y_val_df, yhat_xgb_df], axis=1)

        return y_df

    def reverse_y(self, scaler, col_orig_name: str, y_nt: str, y_val: pd.Series, new_name: str, **kwargs):
        '''Inverse transform y to original state'''

        y_val_df = self.inverse_transformation(df=y_val, scaler=scaler, col_orig_name=col_orig_name, y_nt=y_nt).reset_index(drop=True)
        y_val_df = y_val_df.rename(columns={f'{y_nt}_reversed':new_name})
        y_val_df = y_val_df.reset_index(drop=True)

        return y_val_df

    def _transfomer(self, df: pd.DataFrame, y: str, scaler):
        scaler = scaler.fit(df[[y]])
        return scaler.transform(df[[y]]), scaler

    def _save_scaler(self, scaler, scaler_name: str):

        joblib.dump(scaler, open((c().scalers_fld() / scaler_name), 'wb'))