import numpy as np
import pandas as pd
import pytest

from domain.use_cases.featurederivator import FeatureDerivator

@pytest.fixture
def load_data():
    '''Load data for testing'''

    return pd.read_csv('tests/test_data/test_df.csv')

@pytest.fixture
def correct_dtypes():
    '''Return the correct data types expected'''

    return {
            'week_of_year': np.dtype('int64'),
            'page_position': np.dtype('float64'),
            'price_per_weight': np.dtype('float64')
            }

def test_dtypes(load_data, correct_dtypes):
    '''Test if the data types are returning correct after transformation'''

    feature_derivator = FeatureDerivator(df=load_data)
    df = feature_derivator.get_features()
    df = df[['week_of_year', 'page_position', 'price_per_weight']]
    dtypes_df = dict(df.dtypes)

    assert correct_dtypes == dtypes_df
