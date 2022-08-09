from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn import metrics as m
from sklearn.model_selection import train_test_split

from resources.datatransformer import DataTransformer

class ModelValidator():

    def __init__(self, 
                model_name: str, 
                model: any, 
                X: pd.DataFrame = None, 
                y: pd.Series = None, 
                **kwargs):

        self.model_name = model_name
        self.model = model
        self.X = X
        self.y = y

    def regression_metrics(self, df_val: pd.DataFrame, n_yval: str = 'y', n_yhat: str = 'yhat', **kwargs):
        '''Calculate and return regression metrics'''

        df_val = df_val[df_val['y'] > 0]
        yval = df_val[n_yval]
        yhat = df_val[n_yhat]
        mae = m.mean_absolute_error(yval, yhat)
        mape = m.mean_absolute_percentage_error(yval, yhat)
        rmse = np.sqrt(m.mean_squared_error(yval, yhat))
        
        return pd.DataFrame({'Model Name': self.model_name,
                            'MAE': mae,
                            'MAPE': mape,
                            'RMSE': rmse}, index=[0])

    def classification_metrics(self, X: pd.DataFrame, y: pd.Series, 
                            threshold: float, **kwargs):
        '''Calculate metrics for evalueting classification models'''

        prob = pd.DataFrame(self.model.predict_proba(X))[1]
        yhat = prob.apply(lambda x: 1 if x >= threshold else 0)

        accuracy = m.accuracy_score(y, yhat)
        kappa = m.cohen_kappa_score(y, yhat)
        auc_roc = m.roc_auc_score(y, prob)
        precision = m.precision_score(y, yhat)
        recall = m.recall_score(y, yhat)
        f1_sc = m.f1_score(y, yhat)
        
        return pd.DataFrame([{'Model Name': self.model_name,
                            'Accuracy': accuracy,
                            'Auc-Roc': auc_roc,
                            'Kappa Score': kappa,
                            'Precision Score': precision,
                            'Recall Score': recall,
                            'F1-Score': f1_sc}])

    def kfolds_cross_val_class(self, cv: int,
                            threshold: float,
                            verbose: bool = False, 
                            test_size: float = 0.2, 
                            **kwargs):
        '''Kfolds Cross-Validation for validate with a reality simulation 
        the models' performance'''

        accuracy_list = []
        aucroc_list = []
        kappa_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for k in range(cv):
            if verbose:
                print( '\nKFold Number: {}'.format( k ) )

            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=test_size)

            self.model.fit(X_train, y_train) 
            m_result = self.classification_metrics(X=X_val, y=y_val, threshold=threshold)

            # store performance of each kfold iteration
            accuracy_list.append(m_result['Accuracy'])
            aucroc_list.append(m_result['Auc-Roc'])
            kappa_list.append(m_result['Kappa Score'])
            precision_list.append(m_result['Precision Score'])
            recall_list.append(m_result['Recall Score'])
            f1_list.append(m_result['F1-Score'])

        return pd.DataFrame({
                            'Model Name': self.model_name,
                            'Accuracy CV': np.mean(accuracy_list),
                            'AUC-ROC CV': np.mean(aucroc_list),
                            'Kappa CV': np.mean(kappa_list),
                            'Precision CV': np.mean(precision_list),
                            'Recall CV': np.mean(recall_list),
                            'F1-Score CV': np.mean(f1_list)
                            }, index=[0])

    def kfolds_cross_val_reg(self, y_scaler,
                            col_orig_name: str, y_nt: str,
                            cv: int, verbose: bool = False, 
                            test_size: float = 0.2, **kwargs):
        '''Kfolds Cross-Validation for validate with a reality simulation 
        the models' performance'''

        mae_list = []
        mape_list = []
        rmse_list = []

        for k in range(cv):
            if verbose:
                print( '\nKFold Number: {}'.format( k ) )

            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=test_size)

            self.model.fit(X_train, y_train) 
            yhat = self.model.predict(X_val)

            data_trans = DataTransformer(df=X_train)
            y_df = data_trans.reverse_concat_y(scaler=y_scaler, col_orig_name=col_orig_name, y_nt=y_nt, y_val=y_val, yhat=yhat)
            m_result = self.regression_metrics(y_df.round())

            # store performance of each kfold iteration
            mae_list.append(m_result['MAE'])
            mape_list.append(m_result['MAPE'])
            rmse_list.append(m_result['RMSE'])

        return pd.DataFrame({'Model Name': self.model_name,
                            'MAE CV': np.mean(mae_list),
                            'MAPE CV': np.mean(mape_list),
                            'RMSE CV': np.mean(rmse_list)}, 
                            index=[0])

    def shap_importance(self, **kwargs):
        '''Output a plot with shap values'''

        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X)
        shap.plots.beeswarm(shap_values)

    def plot_feature_importance(self, **kwargs):
        '''Plot feature importance values'''

        feat_imp = self._feat_imp_values()
        plt.subplots(figsize=(20,6))
        sns.barplot(x='feature_importance', y='feature', data=feat_imp, orient='h', color='royalblue')\
                    .set_title('Feature Importance');

    def _feat_imp_values(self, **kwargs):
        '''Feature importance values'''

        feat_imp = pd.DataFrame({'feature': self.X.columns,
                                'feature_importance': self.model.feature_importances_})\
                                .sort_values('feature_importance', ascending=False)\
                                .reset_index(drop=True)

        return feat_imp