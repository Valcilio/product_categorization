import numpy as np
import pandas as pd
import pytest

from domain.entities.product import Product

@pytest.fixture
def load_data():
    '''Load data for testing'''

    return pd.read_csv('tests/test_data/test_df.csv')

@pytest.fixture
def correct_dtypes():
    '''Return the correct data types expected'''

    return {'product_id':       np.dtype('O'),
            'position':         np.dtype('float64'),
            'price':            np.dtype('float64'),
            'weight':           np.dtype('float64'),
            'minimum_quantity': np.dtype('float64'),
            'view_counts':      np.dtype('float64'),
            'express_delivery': np.dtype('float64'),
            'search_page':      np.dtype('int64'),
            'creation_date':    np.dtype('<M8[ns]')}

def test_suit_product(load_data, correct_dtypes):
    '''Test if the data types and columns filtering are returning correct after transformation'''

    df = Product(df=load_data).suit_product()
    dtypes_df = dict(df.dtypes)

    assert correct_dtypes == dtypes_df