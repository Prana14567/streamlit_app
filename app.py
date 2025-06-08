import streamlit as st
import joblib

model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Startup Profit Classifier")

r_d = st.number_input("R&D Spend")
admin = st.number_input("Administration")
marketing = st.number_input("Marketing Spend")
florida = st.checkbox("Florida")
new_york = st.checkbox("New York")
unknown = st.checkbox("Unknown")


input_data = [[r_d, admin, marketing, int(florida), int(new_york), int(unknown)]]
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Profit Level: {prediction[0]}")
