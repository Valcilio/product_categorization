import os
from flask import Flask, request, Response
import pandas as pd

from domain.connectors.modelrunner import ModelRunner
from domain.connectors.datapreparator import DataPreparator

app = Flask(__name__)

@app.route('/framework_data', methods=['GET', 'POST'])
def product_category():
    '''Get json with instructions for get data, run pipeline and 
    reponse with json containing all data'''

    framework_json = request.get_json()

    if framework_json: 
        if isinstance(framework_json, dict): 
            pass

        pipeline = DataPreparator(framework_json)
        df_prepared = pipeline.preparate_data()
        class_pred = ModelRunner(df_prepared).run_class_model()
        
        class_pred = pipeline.reverse_category(class_pred)
        product_id = pipeline.return_product_id()

        df_response = pd.concat([product_id, class_pred], axis=1)

        return df_response.to_json(orient='records', date_format='iso')
    else:
        return Response('{}', status='200', mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port, debug=True)