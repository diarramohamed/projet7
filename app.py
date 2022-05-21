from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

# load trained classifier

X_test = pd.read_csv('X_test.csv', index_col='SK_ID_CURR', encoding ='utf-8')
clf_path = 'model.pkl'
with open(clf_path, 'rb') as f:
    model = pickle.load(f) 
    

#{"SK_ID_CURR": 12333}
@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    SK_ID_CURR=data['SK_ID_CURR']  
    client = X_test[X_test.index == int(SK_ID_CURR)]
    
    prediction = model.predict(client)
    probability= model.predict_proba(client)
    output={'prediction': str(prediction[0]),'probability': probability.max()}
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5000, debug=True)