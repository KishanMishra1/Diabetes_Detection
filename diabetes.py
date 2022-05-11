
#Import the libraries

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


st.write("""
# Diabetes Detection : Detect if someone has diabetes or not !
""")

#Open and display Image
image=Image.open("imagex.png")
st.image(image,caption='ML',use_column_width=True)

#Data

df=pd.read_csv("diabetes.csv")

#Set a subheader

st.subheader("Data Information:")

# Data as a table

st.dataframe(df)

# Data's statistics

st.write(df.describe())

#Show wthe data as chart

chart=st.bar_chart(df)

#Split independent (x) and dependent variable (Y)
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values

#Split the dataset into 75% training and 25% testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#Get the feature input from the user

def get_user_input():
    pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin= st.sidebar.slider('Insulin', 0.0, 846.0, 30.5)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf= st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

#Store a dictionary
    user_data={
        'pregnancies':pregnancies,
        'glucose':glucose,
        'blood_pressure':blood_pressure,
        'skin_thickness':skin_thickness,
        'insulin':insulin,
        'BMI':bmi,
        'DPF':dpf,
        'age':age
    }

    features=pd.DataFrame(user_data,index=[0])
    return features

# Store the user input into a variable
user_input=get_user_input()

#Set a subheader and display the users input

st.subheader('User Input')
st.write(user_input)

#Create and train the model

RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Show model matrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))* 100)+'%')

#Store the model prediction in a variable
prediction=RandomForestClassifier.predict(user_input)

# Set a subheader
st.subheader('Classification :')
st.write(prediction)

