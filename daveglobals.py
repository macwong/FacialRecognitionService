import os
from pathlib import Path

class Globals():
    app_path = str(Path(os.getcwd()).parent)
    data_path = os.path.join(app_path, "data")
    model_path = os.path.join(app_path, "models")
    temp_path = os.path.join(data_path, "temp")
    facenet_path = os.path.join(app_path, "data\\facenet_models\\20180402-114759.pb")
    current_prediction_id = 0
