import pandas as pd
import warnings

class Product():

    def __init__(self, df: pd.DataFrame):

        self.df = df

    def suit_product(self):
        '''Suit product's features'''

        self._filter_data()
        self._definy_dtypes()

        return self.df

    def _filter_data(self):
        '''Filter data'''

        cols = self._product_cols()
        self.df = self.df.filter(cols, axis=1)

    def _definy_dtypes(self):
        '''Definy data types'''

        warnings.filterwarnings('ignore')
        self.df['product_id']       = self.df['product_id'].astype(str)
        self.df['position']         = self.df['position'].astype(float)
        self.df['price']            = self.df['price'].astype(float)
        self.df['weight']           = self.df['weight'].astype(float)
        self.df['minimum_quantity'] = self.df['minimum_quantity'].astype(float)
        self.df['view_counts']      = self.df['view_counts'].astype(float)
        self.df['express_delivery'] = self.df['express_delivery'].astype(float)
        self.df['search_page']      = self.df['search_page'].astype(int)
        self.df['creation_date']    = pd.to_datetime(self.df['creation_date'])

        return self.df

    def _product_cols(self):
        '''List of features for modeling'''

        return ['product_id', 'position', 'price',
                'weight', 'minimum_quantity', 'view_counts',
                'express_delivery', 'search_page', 'creation_date']