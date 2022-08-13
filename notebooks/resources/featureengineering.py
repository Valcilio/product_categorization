import datetime as dt
import numpy as np
import pandas as pd

class FeatureEngineering():

    def __init__(self, time_var: str, df: pd.DataFrame):
        '''This class is specific for this problem'''

        self.time_var = time_var
        self.df = df.copy()

    def page_position(self):
        '''Indicates the position on the search page'''

        return self.df['position'] / self.df['search_page']

    def price_per_weight(self):
        '''Indicates the price per weight'''

        self.df['price_per_weight'] = self.df['price'] / self.df['weight']
        self.df['price_per_weight'] = np.where(self.df['weight'] <= 0, 0, self.df['price_per_weight'])

        return self.df['price_per_weight']

    def day_of_week(self):
        '''Indicates the day of week'''

        return self.df[self.time_var].dt.day_of_week

    def day_of_month(self):
        '''Indicates the day of month'''

        return self.df[self.time_var].dt.day

    def week_of_year(self):
        '''Indicates the week of year'''

        return self.df[self.time_var].dt.isocalendar().week

    def month(self):
        '''Indicates the month year'''

        return self.df[self.time_var].dt.month

    def year(self):
        '''Indicate the year'''

        return self.df[self.time_var].dt.year