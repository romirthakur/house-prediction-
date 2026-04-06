import streamlit as st
import joblib
import numpy  as np

st.title("House Price Prediction")  

st.divider()

st.write("This app uses machine learning for prediction:")

st.divider()

bedroom = st.number_input("Number of bedrooms", min_value=0, value=0)
bathroom = st.number_input("Number of bathrooms", min_value=0, value=0)
living_area = st.number_input("Living area", min_value=0, value=2000)
condtion = st.number_input("Condition of the house", min_value=0, value=3)
numberofschool = st.number_input("Number of schools nearby", min_value=0, value=2)

st.divider()

Model = joblib.load("model.pkl")

x = [[bedroom, bathroom, living_area, condtion, numberofschool]]
predictbutton = st.button("Predict!")

if predictbutton:
    x_array = np.array(x)
    prediction = Model.predict(x_array)
    st.write(f"The predicted price of the house is: {prediction}")

else:
    st.write("Please enter the details of the house and click on Predict to see the predicted price.")








#order of x ["['number of bedrooms', 'number of bathrooms', 'living area',
#  'condition of the house', 'Number of schools nearby'"]
