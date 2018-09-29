from flask import Flask, jsonify, abort, make_response, request
import trainer, predictor
import os
from daveglobals import Globals
import addnewface
import time

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/daveface/train', methods=['POST'])
def train():
    print("Training started...")
    returnValue = "Fail..."

    requestData = request.get_json()
    
    if (
        not requestData 
        or not 'input_folder_path' in requestData 
        or not 'model_folder_name' in requestData
    ):
        print("Invalid JSON request data...", returnValue)
        abort(400)
        
    model_type = "knn"
    
    if 'model_type' in requestData:
        model_type = requestData['model_type']

    input_folder_path = requestData['input_folder_path']
    model_folder_name = requestData['model_folder_name']
    
    success, error = trainer.train(input_folder_path, model_folder_name, model_type)
    code = 400
    
    if success:
        returnValue = "Success!"
        code = 201
    
        
    print("Training ended...", returnValue)
        
    return jsonify({
            'success': success,
            'error': error
            }), code

    
@app.route('/daveface/retrain', methods=['POST'])
def retrain():
    print("Retraining started...")
    returnValue = "Fail..."
    requestData = request.get_json()
    
    if (not requestData or not 'model_folder_name' in requestData):
        print("Invalid JSON request data...", returnValue)
        abort(400)
        
    model_folder_name = requestData['model_folder_name']
    
    model_type = "knn"
    
    if 'model_type' in requestData:
        model_type = requestData['model_type']
        
    success, error = trainer.retrain(model_folder_name, model_type)
    code = 400
    
    if success:
        returnValue = "Success!"
        code = 201
    
        
    print("Training ended...", returnValue)
        
    return jsonify({
            'success': success,
            'error': error
            }), code        

    

@app.route('/daveface/addface', methods=['POST'])
def addface():
    print("Add face started...")
    returnValue = "Fail..."
    
    requestData = request.get_json()
    
    if (
        not requestData 
        or not 'image' in requestData
        or not 'model' in requestData
        or not 'name' in requestData
    ):
        print("Invalid JSON request data...", returnValue)
        abort(400)
        
    image = requestData["image"]
    model = requestData["model"]
    name = requestData["name"]
    
    success, error = addnewface.add(image, model, name)
    code = 400
    
    if success:
        returnValue = "Success!"
        code = 201

    print("Add face ended...", returnValue)
    
    return jsonify({
        'success': success,
        'error': error
    }), code



@app.route('/daveface/predict', methods=['POST'])
def predict():
    print("==================")
    print("Predicting started")
    print("==================")
    print("")
    start = time.time()

    returnValue = "Fail..."
    
    requestData = request.get_json()
    
    if (
        not requestData 
        or not 'image' in requestData
        or not 'model' in requestData
    ):
        print("Invalid JSON request data...", returnValue)
        abort(400)

    verbose = True

    if 'verbose' in requestData:
        verbose = requestData["verbose"]
        
    image = requestData["image"]
    model = requestData["model"]
    
    predict_response = predictor.predict(image, model, verbose)
    code = 400
    
    if predict_response.success:
        returnValue = "Success!"
        code = 201

    end = time.time()
    print("Time taken (Total):", end - start)

    print("Predicting ended...", returnValue, "\n")
    
    return jsonify({
            'success': predict_response.success,
            'predictions': predict_response.predictions,
            'error': predict_response.error
            }), code

@app.route('/daveface/getmodels', methods=['GET'])
def getmodels():
    if os.path.isdir(Globals.model_path) == False:
        return jsonify({
            'success': True,
            'models': []
        })

    directories = [
        o
        for o in os.listdir(Globals.model_path)
        if os.path.isdir(os.path.join(Globals.model_path, o))
    ]
    
    return jsonify({
        'success': True,
        'models': directories
    })


if __name__ == '__main__':
    app.run(debug=True)