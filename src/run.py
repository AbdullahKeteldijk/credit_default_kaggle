from flask import Flask
from flask import request
from flask import jsonify

import json
import pandas as pd
from src.model_pipeline import ModelPipeline


app = Flask(__name__)

from flask import Flask, escape, request

app = Flask(__name__)


@app.route('/')
def hello():

    with open('../models/output.json', 'r') as fp:
        output = json.load(fp)

    return jsonify(output)

    # name = request.args.get("name", "World")
    # return 'Hello, {}!'.format(escape(name))


if __name__ == "__main__":
    flaskapp_args = {"host": "127.0.0.1", "port": 5000}
    app.run(debug=True)


# @app.route("/", methods=['GET', 'POST'])
# def run():
#
#     df = pd.read_csv('../data/hmeq.csv')
#     pipe = ModelPipeline(df)
#     pipe.fit_transform()
#     pipe.print_model_performance()
#
#     return jsonify({'Accuracy': pipe.accuracy,
#                     'F1 score': pipe.f1,
#                     'Precision': pipe.precision,
#                     'Recall': pipe.recall})
#
# if __name__ == '__main__':
#     flaskapp_args = {"host": "0.0.0.0", "port": 5000}
#     app.run(**flaskapp_args)

