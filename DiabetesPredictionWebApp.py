# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:49:48 2022

@author: rituraj
"""

import numpy as np
#pickle is used for open saved file
import pickle

#streamlit is used for deployment
import streamlit as st

# loading the saved model
diabetes_model = pickle.load(open('D:/Projects/ML Projects/Diabetes Prediction/diabetes_model.sav', 'rb'))


#creating diabetes prediction function
def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = diabetes_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    #giving a title
    st.title("Diabetes Prediction Web App")
    
    #getting the input data from the user
    Pregnancies = st.text_input('Enter the number of pregnancies')
    Glucose = st.text_input('Enter Glucose Level')
    BloodPressure = st.text_input('Enter Blood Pressure value')
    SkinThickness = st.text_input('Enter SkinThickness')
    Insulin = st.text_input('Enter insulin level')
    BMI = st.text_input('Enter BMI Value')
    DiabetesPedigreeFunction = st.text_input('Enter Diabetes Pedigree Function')
    Age = st.text_input('Enter Age')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                         Insulin, BMI, Age, DiabetesPedigreeFunction])
        
        
        #gives the output
        st.success(diagnosis)
    

if __name__ == '__main__':
    main()

  