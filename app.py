import json
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, Response
from model import download_data, format_data, train_model
from config import model_file_path, data_base_path

app = Flask(__name__)


def update_data():
    """Download price data, format data and train model."""
    download_data()
    format_data()
    train_model()

def get_inference(token):
    """Load model and predict current price based on token."""
    if token not in ["ETH", "SOL", "BTC", "ARB", "BNB"]:
        raise ValueError("Token not supported")

    model_file_path_token = os.path.join(data_base_path, f"model_{token}USDT.pkl")
    with open(model_file_path_token, "rb") as f:
        loaded_model = pickle.load(f)

    now_timestamp = pd.Timestamp(datetime.now()).timestamp()
    X_new = np.array([now_timestamp]).reshape(-1, 1)
    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0][0]

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token:
        error_msg = "Token is required"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_inference(token)
        return Response(str(inference), status=200)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
