import joblib
import pandas as pd
import pytest
import warnings

from domain.use_cases.featurederivator import FeatureDerivator
from domain.use_cases.rescaler import Rescaler

@pytest.fixture
def load_data():
    '''Load data for testing'''

    return pd.read_csv('tests/test_data/test_df.csv')

@pytest.fixture
def add_cols(load_data):
    '''Derivating new cols for the models'''

    df = load_data
    feature_derivator = FeatureDerivator(df=load_data)
    df = feature_derivator.get_features()

    return df

@pytest.fixture
def load_scalers():
    '''Load scalers for transform data'''

    warnings.filterwarnings(action='ignore')
    category_scaler         = joblib.load(open('scalers/label-encoder_category_scaler.pkl', 'rb'))
    express_delivery_scaler = joblib.load(open('scalers/min-max_express_delivery_scaler.pkl', 'rb'))
    minimum_quantity_scaler = joblib.load(open('scalers/min-max_minimum_quantity_scaler.pkl', 'rb'))
    page_position_scaler    = joblib.load(open('scalers/min-max_page_position_scaler.pkl', 'rb'))
    position_scaler         = joblib.load(open('scalers/min-max_position_scaler.pkl', 'rb'))
    price_per_weight_scaler = joblib.load(open('scalers/min-max_price_per_weight_scaler.pkl', 'rb'))
    price_scaler            = joblib.load(open('scalers/min-max_price_scaler.pkl', 'rb'))
    view_counts_scaler      = joblib.load(open('scalers/min-max_view_counts_scaler.pkl', 'rb'))
    week_of_year_scaler     = joblib.load(open('scalers/min-max_week_of_year_scaler.pkl', 'rb'))
    weight_scaler           = joblib.load(open('scalers/min-max_weight_scaler.pkl', 'rb'))
    
    return [position_scaler, price_scaler, weight_scaler,
            minimum_quantity_scaler, view_counts_scaler, 
            express_delivery_scaler, page_position_scaler, 
            price_per_weight_scaler, week_of_year_scaler,
            category_scaler]

@pytest.fixture
def rescaled_features():
    '''List of rescaled features for testing'''

    return ['min-max_position', 'min-max_price', 'min-max_weight',
            'min-max_minimum_quantity', 'min-max_view_counts',
            'min-max_express_delivery', 'min-max_page_position',
            'min-max_price_per_weight', 'min-max_week_of_year']

@pytest.fixture
def encoding_category(load_data, load_scalers):
    '''Encoding category to testing'''

    other_categories = ['Papel e Cia', 'Bijuterias e Jóias']
    load_data['category'] = load_data['category'].apply(lambda x: 'Outros' if x in other_categories else x)
    load_data['label-encoder_category'] = load_scalers[9].transform(load_data[['category']])

    return load_data

@pytest.fixture
def categories():
    '''List of categories to exist after reverse transformation'''

    return ['Decoração', 'Lembrancinhas', 'Outros', 'Bebê']

@pytest.fixture
def rescaler(add_cols, load_scalers):

    warnings.filterwarnings(action='ignore')
    rescaler = Rescaler(df=add_cols, 
                        position_scaler=load_scalers[0],
                        price_scaler=load_scalers[1],
                        weight_scaler=load_scalers[2],
                        minimum_quantity_scaler=load_scalers[3],
                        view_counts_scaler=load_scalers[4],
                        express_delivery_scaler=load_scalers[5],
                        page_position_scaler=load_scalers[6],
                        price_per_weight_scaler=load_scalers[7],
                        week_of_year_scaler=load_scalers[8],
                        category_scaler=load_scalers[9])

    return rescaler

@pytest.fixture
def run_rescaler(rescaler):
    '''Run rescaler for get data for testing'''

    df = rescaler.run()

    return df

def test_run(run_rescaler, rescaled_features):
    '''Test if the data types are returning correct after transformation'''

    for feature in rescaled_features:
        assert ((run_rescaler[feature].max() <= 1) & (run_rescaler[feature].min() >= 0))

def test_inverse_category(categories, rescaler, encoding_category):
    '''Test if the data types are returning correct after transformation'''

    encoding_category['label-encoder_category'] = rescaler.inverse_category(encoding_category)
    categories_df = list(encoding_category['label-encoder_category'].unique())

    assert categories == categories_df
