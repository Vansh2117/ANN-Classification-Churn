import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# load the trained model
model = tf.keras.models.load_model('model.h5')
# load the scaler and encoder pickle files
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('ohe_geography.pkl','rb') as file:
    ohe_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')
# User inputs
geography =  st.selectbox('Geography',ohe_geo.categories_[0])
Gender =  st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_prdts = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([Gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_prdts],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})
# applying the ohe on geo
geo_encoded=ohe_geo.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
# Combining the columns of ohe in main data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
# scalling the data
input_scaled=scaler.transform(input_data)
# prediction
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba>0.5:
    print('The customer is likely to churn')
else:
    print('The customer is not likely to churn')