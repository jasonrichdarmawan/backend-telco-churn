import joblib
from keras.models import load_model
from tensorflow_addons.metrics import F1Score
import pandas as pd

from flask import Flask, request
import numpy as np

# Load the sequential improved model
with open('./models/preprocessor.pkl', 'rb') as file_1:
    prep = joblib.load(file_1)
    
model = load_model(
    filepath='./models/seq_imp.h5',
    custom_objects={
        'Addons>F1Score': F1Score(num_classes=1, average='macro', threshold=0.3)
    }
)

# Database simulation
db = pd.read_csv('./db/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
# App Initialization
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    args = request.json
    
    customerID = args.get('customerID')
    
    fetch_customer_data = args.get('fetch_customer_data')
    
    print("hello")
    
    # guard clause
    if type(customerID) != list:
        return "Bad Request", 400
    
    data = db.loc[db['customerID'].isin(customerID)]
    
    X_final = prep.transform(X=data)
    
    y_pred_proba = model.predict(x=X_final)
    
    y_pred = np.where(y_pred_proba >= 0.3, 'Yes', 'No')
    
    data['Churn'] = y_pred
    
    # guard clause
    if fetch_customer_data == True:
        return data.to_json()
    else:
        return data[['customerID', 'Churn']].to_json()
    
@app.route('/query', methods=['GET'])
def query():
    """
    simulation for Exploratory Data Analysis.
    """
    return db.to_json()
    
# when the appy is deploy, the __name__ is app instead of __main__
if __name__ == '__main__':
    # host='0.0.0.0' to have have the server available externally as well.
    # make sure to remove parameter port when deploying.
    app.run(host='0.0.0.0', port=5002)