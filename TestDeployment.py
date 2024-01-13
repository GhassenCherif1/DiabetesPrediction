import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import streamlit as st

def diabetes_prediction(input):
    model=pickle.load(open("D:\model.sav","rb"))
    input_array = np.asarray(input)
    input_reshaped = input_array.reshape(1,-1)
    prediction = model.predict(input_reshaped)
    if(prediction[0]==0):
        return "This person is not diabetic"
    else:
        return "This person is diabetic"

def main():
    st.title("Predicting Diabetes for a Brighter Tomorrow")
    Pregnancies = st.text_input("Pregnancies")
    Gluscose = st.text_input("Glucose")
    bp = st.text_input("Blood Pressure")
    sth = st.text_input("Skin Thickness")
    ins = st.text_input("Insulin")
    bmi = st.text_input("Body Mass Index")
    dpf = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")
    
    diagnosis = ""
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Gluscose,bp,sth, ins,bmi,dpf,age])
    
    st.success(diagnosis)    

if __name__ == "__main__":
    main()