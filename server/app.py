from flask import Flask, request
import pandas as pd
import numpy as np
from datetime import datetime
import pickle, os, json

# Initialize Flask app
app = Flask(__name__)

class RandomForrest:
    def __init__(self):
        self.model = None
        self.model_dir_path = None

    def load(self):
        models_path = './model'
        latest_model_folder = None
        max_date = None

        for folder in os.listdir(models_path):
            if os.path.isdir(os.path.join(models_path, folder)):
                
                date = datetime.strptime(folder, "%Y-%m-%dT%H.%M.%S.%fZ")

                if (max_date is None) or (date > max_date):
                    max_date = date
                    latest_model_folder = folder
        
        if latest_model_folder is None:
            raise("No model folder was found, please train a model first!")
        
        self.model_dir_path = os.path.join(models_path, latest_model_folder)
        pickle_file_path = os.path.join(models_path, latest_model_folder, 'model.pkl')
        with open(pickle_file_path, 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)

        print("Latest Model Loaded : ", pickle_file_path)


    def log_data(self, data):
        log_file_path = os.path.join(self.model_dir_path, 'log.json')
        if os.path.exists(log_file_path):
            # If the file exists, append to it
            with open(log_file_path, 'r+') as f:
                logs = json.load(f)
                logs.append(data)
                f.seek(0)
                json.dump(logs, f, indent=4)
        else:
            # If the file doesn't exist, create a new file and write the data
            with open(log_file_path, 'w') as f:
                json.dump([data], f, indent=4)


randomForrest = RandomForrest()

@app.route('/test', methods=['POST'])
def recommend():

    payload = request.get_json()
    data = payload['data']
    features = payload['features']

    data = np.array([data])
    data_df = pd.DataFrame(data, columns=features)
    prediction = int(randomForrest.model.predict(data_df)[0])
    res = "Diabetic" if prediction else "Not Diabetic"

    randomForrest.log_data({
        "request": payload,
        "response": res,
        "time" : datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    })
    
    return "Patient is " + res

if __name__ == '__main__':
    try:
        randomForrest.load()
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        print(e)