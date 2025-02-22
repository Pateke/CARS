
import pickle
import streamlit as st       
import pandas as pd

# load the file that contains the model (model.pkl)
with open("model.pkl", "rb") as f:
  model = pickle.load(f)

# give the Streamlit app page a title
st.title("Price Predictor")

# input widget for getting user values for X (feature matrix value)
Price = st.slider("Price", min_value=0, max_value=100, value=20)
Engine_Size = st.slider("Engine_Size", min_value=0, max_value=100, value=20)
Mileage = st.slider("Mileage", min_value=0, max_value=100, value=20)
Owner_Count = st.slider("Owner_Count", min_value=0, max_value=100, value=20)

# After selesting price, the user then submits the price value
if st.button("Predict"):
  # take the price value, and format the value the right way
  prediction = model.predict([[Price, Engine_Size, Mileage, Owner_Count]])[0].round(2)
  st.write("The price of your vehicle is", prediction, "thousand dollars")
