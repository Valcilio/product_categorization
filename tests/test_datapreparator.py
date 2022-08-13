import pandas as pd
import pytest

from domain.connectors.datapreparator import DataPreparator

@pytest.fixture
def load_data():
    '''Load data for testing'''

    df_json = pd.read_csv('tests/test_data/test_df.csv').to_json()

    return {'data': df_json}

@pytest.fixture
def rescaled_features():
    '''List of rescaled features for testing'''

    return ['min-max_position', 'min-max_price', 'min-max_weight',
            'min-max_minimum_quantity', 'min-max_view_counts',
            'min-max_express_delivery', 'min-max_page_position',
            'min-max_price_per_weight', 'min-max_week_of_year']

@pytest.fixture
def model_features():
    '''List of features for model'''

    return {'min-max_express_delivery':'float64',
            'min-max_minimum_quantity':'float64',
            'min-max_page_position':'float64',
            'min-max_position':'float64',
            'min-max_price_per_weight':'float64',
            'min-max_price':'float64',
            'min-max_view_counts':'float64',
            'min-max_week_of_year':'float64',
            'min-max_weight':'float64'}

@pytest.fixture
def run_preparate_data(load_data):
    '''Run process to rescale all data for testing'''

    df_pred = DataPreparator(json_df=load_data).preparate_data()

    return df_pred

def test_datapreparator_dtypes(run_preparate_data, model_features):
    '''Test if the data types are returning correct after transformation'''

    cols_dtype = dict(run_preparate_data.dtypes)

    assert cols_dtype == model_features

def test_datapreparator_scales(run_preparate_data, rescaled_features):
    '''Test if scales are returning correct'''

    for feature in rescaled_features:
        assert ((run_preparate_data[feature].max() <= 1) & (run_preparate_data[feature].min() >= 0))
