import joblib
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from domain.connectors.datapreparator import DataPreparator

@pytest.fixture
def load_data():
    '''Load data for testing'''

    df_json = pd.read_csv('tests/test_data/test_df.csv').to_json()

    return {'data': df_json}

@pytest.fixture
def load_category_scaler():
    '''Load category scaler'''

    return joblib.load(open('scalers/label-encoder_category_scaler.pkl', 'rb'))

@pytest.fixture
def structure_json(load_data):
    '''Structure JSON for testing'''

    json_df = load_data['data']
    df_test = pd.read_json(json_df)

    return df_test

@pytest.fixture
def encoding_category(structure_json, load_category_scaler):
    '''Encoding category to testing'''

    other_categories = ['Papel e Cia', 'Bijuterias e Jóias']
    structure_json['category'] = structure_json['category'].apply(lambda x: 'Outros' if x in other_categories else x)
    structure_json['label-encoder_category'] = load_category_scaler.transform(structure_json['category'].ravel())

    return structure_json

@pytest.fixture
def categories():
    '''List of categories to exist after reverse transformation'''

    return ['Decoração', 'Lembrancinhas', 'Outros', 'Bebê']

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
def call_datapreparator(load_data):
    '''Run process to rescale all data for testing'''

    data_preparator = DataPreparator(json_df=load_data)

    return data_preparator

@pytest.fixture
def get_product_id(load_data):
    '''Return product_id'''

    df_test = pd.read_json(load_data['data'])

    return df_test['product_id']

@pytest.fixture
def run_preparate_data(call_datapreparator):
    '''Run process to rescale all data for testing'''

    df_pred = call_datapreparator.preparate_data()

    return df_pred

def test_datapreparator_dtypes(run_preparate_data, model_features):
    '''Test if the data types are returning correct after transformation'''

    cols_dtype = dict(run_preparate_data.dtypes)

    assert cols_dtype == model_features

def test_datapreparator_scales(run_preparate_data, rescaled_features):
    '''Test if scales are returning correct'''

    for feature in rescaled_features:
        assert ((run_preparate_data[feature].max() <= 1) & (run_preparate_data[feature].min() >= 0))

def test_reverse_encoding(call_datapreparator, encoding_category, categories):
    '''Test if is reversing values of categories correctly'''

    encoding_category = call_datapreparator.reverse_category(encoding_category)
    categories_df = list(encoding_category['category'].unique())

    assert categories == categories_df

def test_return_product_id(get_product_id, call_datapreparator):
    '''Test if is returning the product_id correctly'''

    product_id = call_datapreparator.return_product_id()

    assert_series_equal(product_id, get_product_id) 