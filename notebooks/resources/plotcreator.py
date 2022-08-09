import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
from .datatransformer import DataTransformer

class  PlotCreator():

    def __init__(self, df: pd.DataFrame, 
                individual_figsize = (18,8), titlesize=20, 
                axes_size = 13, ticks_size = 10, **kwargs):

                self.individual_figsize = individual_figsize
                self.titlesize = titlesize
                self.axes_size = axes_size
                self.ticks_size = ticks_size
                self.df = df
                self.transformer = DataTransformer(df)

    def save_fig(self, saving_figloc: str, df: pd.DataFrame = None, **kwargs):
        '''Save figure and dataframes in the given path'''

        if df is not None:
            fig_name = os.path.basename(saving_figloc)
            dfi.export(df, fig_name);
            shutil.move(fig_name, saving_figloc);
        else:
            plt.savefig(saving_figloc)

    def statistical_description(self, saving_figloc: str = False, **kwargs):
        '''Calculate and show statistical descriptions'''

        df = self.transformer.derivate_int_float_columns()

        ct1 = pd.DataFrame(df.apply(np.mean)).T
        ct2 = pd.DataFrame(df.apply(np.median)).T

        d1 = pd.DataFrame(df.apply(np.std)).T
        d2 = pd.DataFrame(df.apply(min)).T
        d3 = pd.DataFrame(df.apply(max)).T
        d4 = pd.DataFrame(df.apply(lambda x: x.max() - x.min())).T
        d5 = pd.DataFrame(df.apply(lambda x: x.skew())).T
        d6 = pd.DataFrame(df.apply(lambda x: x.kurtosis())).T

        m = pd.concat([ct1, ct2, d1, d2, d3, d4, d5, d6]).T.reset_index()
        m.columns = ['attributes', 'mean', 'median', 'std', 
                     'min', 'max', 'range', 'skew', 'kurtosis']

        if saving_figloc:
            self.save_fig(df = m, saving_figloc=saving_figloc)

        return m

    def plot_cont_analy(self, expl_var: str, res_var: str, saving_figloc: str = False, **kwargs):
        '''Method for analyze the continuous variables'''

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14));
        
        self._heatmap(ax_=ax1, res_var=res_var, expl_var=expl_var)
        self._distribution(ax_=ax2, expl_var=expl_var)
        self._scatter(ax_=ax3, res_var=res_var, expl_var=expl_var)
        self._regplot(ax_=ax4, res_var=res_var, expl_var=expl_var)

        if saving_figloc:
            self.save_fig(saving_figloc=saving_figloc)

    def plot_cat_analy(self, res_var: str, expl_var: str, saving_figloc: str = False, **kwargs):
        '''Method for analyze the categorical variable'''

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14));

        self._median_cat(expl_var=expl_var, res_var=res_var, ax_=ax1)
        self._volumetry_cat(expl_var=expl_var, res_var=res_var, ax_=ax2)
        self._sum_cat(expl_var=expl_var, res_var=res_var, ax_=ax3)
        self._mean_cat(expl_var=expl_var, res_var=res_var, ax_=ax4)

        if saving_figloc:
            self.save_fig(saving_figloc=saving_figloc)

    def outlier_detector_boxplot(self, x: str, saving_figloc: str = False, **kwargs):
        '''Plot boxplot, it's idealized to detect outliers in temporal situations'''
 
        df = self.df.copy()

        plt.title(f"{x}'s Boxplot to Outlier Detection", fontsize=self.titlesize)

        sns.boxplot(data=df, x=x, **kwargs)

        if saving_figloc:
            self.save_fig(saving_figloc=saving_figloc)

        return None

    def distribution_check(self, all_figsize: tuple =(22,15), saving_figloc: str = False, **kwargs):
        '''Plot distribution to check the type of from all the numerical variables in dataset'''

        df = self.df.copy()

        df.hist(figsize=all_figsize, **kwargs);

        if saving_figloc:
            self.save_fig(saving_figloc=saving_figloc)

        return None

    def _heatmap(self, ax_, res_var: str, expl_var: str, method: str = 'pearson', **kwargs):
        '''Plot heatmap to analysis correlation'''

        sns.heatmap(data=self.df[[res_var, expl_var]].corr(method=method), annot=True, ax=ax_) 
        ax_.title.set_text(f"Correlation Heatmap ({res_var} x {expl_var})");

    def _distribution(self, ax_, expl_var: str, **kwargs):
        '''Plot distribution'''

        self.df['views'].hist(ax=ax_);
        ax_.title.set_text(f"Distribution ({expl_var})");

    def _scatter(self, ax_, res_var: str, expl_var: str, **kwargs):
        '''Plot a scatter'''

        sns.scatterplot(x=expl_var, y=res_var, data=self.df, ax=ax_);
        ax_.title.set_text(f"Scatterplot ({res_var} x {expl_var})");

    def _regplot(self, ax_, res_var: str, expl_var: str, **kwargs):
        '''Plot a regplot'''

        sns.regplot(x=expl_var, y=res_var, data=self.df, ax=ax_);
        ax_.title.set_text(f"Regplot({res_var} x {expl_var})");

    def _median_cat(self, ax_, res_var: str, expl_var: str, **kwargs):
        '''Plot a categorical with median from the response
        variable and '''

        df_group = self.df[[res_var, expl_var]].groupby(expl_var).median().reset_index()
        df_group.plot.bar(x=expl_var, y=res_var, ax=ax_, **kwargs);
        ax_.title.set_text(f"Median By Category ({res_var} x {expl_var})");

    def _sum_cat(self, ax_, res_var: str, expl_var: str, **kwargs):
        '''Plot a categorical with median from the response
        variable and '''

        df_group = self.df[[res_var, expl_var]].groupby(expl_var).sum().reset_index()
        df_group.plot.bar(x=expl_var, y=res_var, ax=ax_, **kwargs);
        ax_.title.set_text(f"Sum By Category ({res_var} x {expl_var})");

    def _mean_cat(self, ax_, res_var: str, expl_var: str, **kwargs):
        '''Plot a categorical with mean from the response
        variable and '''

        df_group = self.df[[res_var, expl_var]].groupby(expl_var).mean().reset_index()
        df_group.plot.bar(x=expl_var, y=res_var, ax=ax_, **kwargs);
        ax_.title.set_text(f"Mean By Category ({res_var} x {expl_var})");

    def _volumetry_cat(self, ax_, expl_var: str, **kwargs):
        '''Plot volumetry bars'''

        pd.DataFrame(self.df[expl_var].value_counts()).plot.bar(ax=ax_)
        ax_.title.set_text(f"Volumetry Explanable Variable ({expl_var})");