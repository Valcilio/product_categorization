import os
from flask import Flask, request, Response
import pandas as pd

from domain.connectors.modelsrunner import ModelsRunner
from domain.connectors.datapreparator import DataPreparator

app = Flask(__name__)

@app.route('/framework_data', methods=['GET', 'POST'])
def crypto_forecast():
    '''Get json with instructions for get data, run pipeline and 
    reponse with json containing all data'''

    framework_json = request.get_json()

    if framework_json: 
        if isinstance(framework_json, dict): 
            pass

        pipeline = DataPreparator(json_df=framework_json)
        df_prepared = pipeline.preparate_data()

        model_run = ModelsRunner(df_prepared=df_prepared)
        class_pred = model_run.run_class_model()
        reg_pred = model_run.run_reg_model()

        if class_pred.iloc[0,0] == 'yes':
            df_response = pd.concat([class_pred, reg_pred], axis=1)
        else:
            df_response = class_pred

        return df_response.to_json(orient='records', date_format='iso')
    else:
        return Response('{}', status='200', mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port, debug=True)