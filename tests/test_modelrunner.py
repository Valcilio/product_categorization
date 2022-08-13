import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from domain.connectors.datapreparator import DataPreparator
from domain.connectors.modelrunner import ModelRunner

@pytest.fixture
def load_data():
    '''Load data for testing'''

    df_json = pd.read_csv('tests/test_data/test_df.csv').to_json()
    json_df = {'data':df_json}
    df_preparated = DataPreparator(json_df).preparate_data()

    return df_preparated

@pytest.fixture
def load_predictions():
    '''Load data for testing'''

    pred_test = pd.read_csv('tests/test_data/pred_test.csv')['category']
    pred_test.name = 'label-encoder_category'

    return pred_test

@pytest.fixture
def pred_model(load_data):
    '''Running model for prediction'''

    pred = ModelRunner(load_data).run_class_model()

    return pred

def test_prediction(load_predictions, pred_model):
    '''Test if predictions from notebook and model are equals'''

    assert_series_equal(pred_model, load_predictions)