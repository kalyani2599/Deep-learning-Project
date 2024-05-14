# import the required libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st


# provide header name to browser
st.set_page_config(page_title='Breast Cancer Analysis', layout='wide')

# Add a title in the body of the browser
st.title('Breast Cancer Analysis')

# Sidebar sliders
mean_radius = st.sidebar.slider('Mean Radius', 0.0, 40.0, 15.0)
mean_texture = st.sidebar.slider('Mean Texture', 0.0, 30.0, 20.0)
mean_perimeter = st.sidebar.slider('Mean Perimeter', 0.0, 300.0, 100.0)
mean_area = st.sidebar.slider('Mean Area', 0.0, 2500.0, 500.0)
mean_smoothness = st.sidebar.slider('Mean Smoothness', 0.0, 0.3, 0.1)
mean_compactness = st.sidebar.slider('Mean Compactness', 0.0, 0.5, 0.2)
mean_concavity = st.sidebar.slider('Mean Concavity', 0.0, 0.5, 0.2)
mean_concave_points = st.sidebar.slider('Mean Concave Points', 0.0, 0.3, 0.1)
mean_symmetry = st.sidebar.slider('Mean Symmetry', 0.0, 1.0, 0.5)
mean_fractal_dimension = st.sidebar.slider('Mean Fractal Dimension', 0.0, 0.2, 0.1)
radius_error = st.sidebar.slider('Radius Error', 0.0, 2.0, 0.5)
texture_error = st.sidebar.slider('Texture Error', 0.0, 5.0, 1.0)
perimeter_error = st.sidebar.slider('Perimeter Error', 0.0, 20.0, 5.0)
area_error = st.sidebar.slider('Area Error', 0.0, 200.0, 50.0)
smoothness_error = st.sidebar.slider('Smoothness Error', 0.0, 0.03, 0.01)
compactness_error = st.sidebar.slider('Compactness Error', 0.0, 0.1, 0.05)
concavity_error = st.sidebar.slider('Concavity Error', 0.0, 0.1, 0.05)
concave_points_error = st.sidebar.slider('Concave Points Error', 0.0, 0.05, 0.025)
symmetry_error = st.sidebar.slider('Symmetry Error', 0.0, 0.1, 0.05)
fractal_dimension_error = st.sidebar.slider('Fractal Dimension Error', 0.0, 0.03, 0.015)
worst_radius = st.sidebar.slider('Worst Radius', 0.0, 40.0, 20.0)
worst_texture = st.sidebar.slider('Worst Texture', 0.0, 40.0, 20.0)
worst_perimeter = st.sidebar.slider('Worst Perimeter', 0.0, 300.0, 150.0)
worst_area = st.sidebar.slider('Worst Area', 0.0, 3000.0, 750.0)
worst_smoothness = st.sidebar.slider('Worst Smoothness', 0.0, 0.3, 0.15)
worst_compactness = st.sidebar.slider('Worst Compactness', 0.0, 1.0, 0.5)
worst_concavity = st.sidebar.slider('Worst Concavity', 0.0, 1.0, 0.5)
worst_concave_points = st.sidebar.slider('Worst Concave Points', 0.0, 0.5, 0.25)
worst_symmetry = st.sidebar.slider('Worst Symmetry', 0.0, 1.0, 0.5)
worst_fractal_dimension = st.sidebar.slider('Worst Fractal Dimension', 0.0, 0.3, 0.15)

# Add a button for prediction
submit = st.button('Predict')

st.subheader('Predictions Are : ')

# Create a function to predict
def predict_result(scaler_path, model_path):
    with open(scaler_path, 'rb') as file1:
        scaler = pickle.load(file1)
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    
    dct = {
        'mean radius': mean_radius,
        'mean texture': mean_texture,
        'mean perimeter': mean_perimeter,
        'mean area': mean_area,
        'mean smoothness': mean_smoothness,
        'mean compactness': mean_compactness,
        'mean concavity': mean_concavity,
        'mean concave points': mean_concave_points,
        'mean symmetry': mean_symmetry,
        'mean fractal dimension': mean_fractal_dimension,
        'radius error': radius_error,
        'texture error': texture_error,
        'perimeter error': perimeter_error,
        'area error': area_error,
        'smoothness error': smoothness_error,
        'compactness error': compactness_error,
        'concavity error': concavity_error,
        'concave points error': concave_points_error,
        'symmetry error': symmetry_error,
        'fractal dimension error': fractal_dimension_error,
        'worst radius': worst_radius,
        'worst texture': worst_texture,
        'worst perimeter': worst_perimeter,
        'worst area': worst_area,
        'worst smoothness': worst_smoothness,
        'worst compactness': worst_compactness,
        'worst concavity': worst_concavity,
        'worst concave points': worst_concave_points,
        'worst symmetry': worst_symmetry,
        'worst fractal dimension': worst_fractal_dimension
    }
    
    x_new = pd.DataFrame([dct])  # Wrap the dictionary in a list to create a DataFrame
    # Scale input data
    x_new_pre = scaler.transform(x_new)
    # Predict using the loaded model
    prediction = model.predict(x_new_pre)[0]
    return prediction

# Show the results in Streamlit
if submit:
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'
    prediction = predict_result(scaler_path, model_path)
    result = 'Benign' if prediction.any() == 0 else 'Malignant'
    st.write(f'Predicted Result: {result}')

