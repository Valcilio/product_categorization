import numpy as np
import pandas as pd

from domain.entities.product import Product

class FeatureDerivator():

    def __init__(self, df: pd.DataFrame):

        self.df = Product(df).suit_product()

    def get_features(self):
        '''Derivate all new features'''

        self.df['week_of_year'] = self._week_of_year()
        self.df['page_position'] = self._page_position()
        self.df['price_per_weight'] = self._price_per_weight()

        return self.df

    def _week_of_year(self):
        '''Indicates the week of year'''

        return self.df['creation_date'].dt.isocalendar().week.astype(int)

    def _page_position(self):
        '''Indicates the position on the search page'''

        return self.df['position'] / self.df['search_page']

    def _price_per_weight(self):
        '''Indicates the price per weight'''

        self.df['price_per_weight'] = self.df['price'] / self.df['weight']
        self.df['price_per_weight'] = np.where(self.df['weight'] <= 0, 0, self.df['price_per_weight'])

        return self.df['price_per_weight']