import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from domain.entities.product import Product

class Rescaler():

    def __init__(self, df: pd.DataFrame,
                position_scaler: MinMaxScaler,
                price_scaler: MinMaxScaler,
                weight_scaler: MinMaxScaler,
                minimum_quantity_scaler: MinMaxScaler,
                view_counts_scaler: MinMaxScaler,
                express_delivery_scaler: MinMaxScaler,
                page_position_scaler: MinMaxScaler,
                price_per_weight_scaler: MinMaxScaler,
                week_of_year_scaler: MinMaxScaler,
                category_scaler: MinMaxScaler):

        self.df = df
        self.position_scaler = position_scaler
        self.price_scaler = price_scaler
        self.weight_scaler = weight_scaler
        self.minimum_quantity_scaler = minimum_quantity_scaler
        self.view_counts_scaler = view_counts_scaler
        self.express_delivery_scaler = express_delivery_scaler
        self.page_position_scaler = page_position_scaler
        self.price_per_weight_scaler = price_per_weight_scaler
        self.week_of_year_scaler = week_of_year_scaler
        self.category_scaler = category_scaler

    def inverse_category(self, df_pred: pd.DataFrame):
        '''Inverse category to the normal scale'''

        df_pred['category'] = self.category_scaler.inverse_transform(df_pred[['label-encoder_category']])

        return df_pred['category']

    def run(self):
        '''Return dataset with model features rescaled'''

        self._position()
        self._price()
        self._weight()
        self._minimum_quantity()
        self._view_counts()
        self._express_delivery()
        self._page_position()
        self._price_per_weight()
        self._week_of_year()
        self._filter_cols()

        return self.df

    def _position(self):
        '''Rescale position for 0 to 1 with MinMaxScaler'''

        self.df['min-max_position'] = self.position_scaler.transform(self.df[['position']])

    def _price(self):
        '''Rescale price for 0 to 1 with MinMaxScaler'''

        self.df['min-max_price'] = self.price_scaler.transform(self.df[['price']])

    def _weight(self):
        '''Rescale weight for 0 to 1 with MinMaxScaler'''

        self.df['min-max_weight'] = self.weight_scaler.transform(self.df[['weight']])

    def _minimum_quantity(self):
        '''Rescale minimum_quantity for 0 to 1 with MinMaxScaler'''

        self.df['min-max_minimum_quantity'] = self.minimum_quantity_scaler.transform(self.df[['minimum_quantity']])

    def _view_counts(self):
        '''Rescale view_counts for 0 to 1 with MinMaxScaler'''

        self.df['min-max_view_counts'] = self.view_counts_scaler.transform(self.df[['view_counts']])

    def _express_delivery(self):
        '''Rescale express_delivery for 0 to 1 with MinMaxScaler'''

        self.df['min-max_express_delivery'] = self.express_delivery_scaler.transform(self.df[['express_delivery']])

    def _page_position(self):
        '''Rescale page_position for 0 to 1 with MinMaxScaler'''

        self.df['min-max_page_position'] = self.page_position_scaler.transform(self.df[['page_position']])

    def _price_per_weight(self):
        '''Rescale price_per_weight for 0 to 1 with MinMaxScaler'''

        self.df['min-max_price_per_weight'] = self.price_per_weight_scaler.transform(self.df[['price_per_weight']])

    def _week_of_year(self):
        '''Rescale weight for 0 to 1 with MinMaxScaler'''

        self.df['min-max_week_of_year'] = self.week_of_year_scaler.transform(self.df[['week_of_year']])

    def _filter_cols(self):
        '''Filtering to just return models' cols'''

        self.df = self.df[self._model_cols()]

    def _model_cols(self):
        '''List with all columns to model'''

        return ['min-max_position', 'min-max_price', 'min-max_weight',
                'min-max_minimum_quantity', 'min-max_view_counts',
                'min-max_express_delivery', 'min-max_page_position',
                'min-max_price_per_weight', 'min-max_week_of_year']

    