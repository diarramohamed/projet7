import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import lime
from lime import lime_tabular
import streamlit.components.v1 as cp
import pickle

clf_inter = 'local_I.pkl'
with open(clf_inter, 'rb') as f:
    local_inter = pickle.load(f) 


url='http://mohamed22222.pythonanywhere.com/api'

@st.cache
def load_data():
    df = pd.read_csv('X_test1.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    #df.drop(columns=['Unnamed: 0'],inplace=True)
    return df

data1=load_data()

#Title
st.title('Dashboard Credit Scoring')
#Infos personnel du client
#st.header("**Information du client**")
#revenu= st.dataframe(data1[data1.index == int(id)])
#age
#sexe
#Family status
#Nb_enfant
#Montant du crédit


  
#interpretor = lime_tabular.LimeTabularExplainer(
 #   training_data=np.array(data1),
 #   feature_names=data1.columns,
 #   mode='classification'
#)

list_ind=list(data1.index)

#sidebar
id=st.sidebar.selectbox('choisir un client',list_ind)

st.dataframe(data1[data1.index == int(id)])

# json
data_json= {'SK_ID_CURR':id}
response=requests.post(url=url,json=data_json).json()
#st.write(response)

# proba
if int(response['prediction'])==0:
    st.write('Le client est solvable avec une probabilité de '+str(response['probability']))
else :
    st.write('Le client est non solvable avec une probabilité de '+str(response['probability']))

# exp = interpretor.explain_instance(
#    data_row=data1[data1.index == int(id)], ##new data
#   predict_fn=model.predict_proba
#)


#cp.html(exp.as_html(),height=400)

exp= local_inter[str(float(id))]
cp.html(exp.as_html(),height=400)

