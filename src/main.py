import gradio as gr
import joblib
import numpy as np
import pandas as pd
import mlModel
from sklearn.preprocessing import LabelEncoder

def load_resources():
    # Cache model and encoders on first use
    global model, encoder
    if 'model' not in globals():
        model = joblib.load('random_forest_model.pkl')
    if 'encoder' not in globals():
        encoder = joblib.load('encoder.pkl')

def predict_new_data(milage, horsepower, displacement, cylinders, model_year, brand, fuel_type, ext_col, int_col, transmission, accident, clean_title):
    # Create a dictionary to convert input to DataFrame format
    load_resources()
    user_data = {
        'milage': [milage],
        'horsepower': [horsepower],
        'displacement': [displacement],
        'cylinders': [cylinders],
        'model_year': [model_year],
        'brand': [brand],
        'fuel_type': [fuel_type], #mappe til int
        'ext_col': [ext_col],
        'int_col': [int_col],
        'transmission': [transmission],
        'accident': [accident],
        'clean_title': [clean_title]
    }

    user_data_df = pd.DataFrame(user_data)


    user_data_df["accident"] = user_data_df["accident"].replace({
        'At least 1 accident or damage reported': 1,
        'None reported': 0
    })

    # Replace values in the 'clean_title' column
    user_data_df["clean_title"] = user_data_df["clean_title"].replace({
        "Yes": 1,
        "No": 0
    })

    # Replace values in the 'transmission' column
    user_data_df["transmission"] = user_data_df["transmission"].replace({'Automatic':1, 'Dual Clutch':2, 'Manual':3, 'Variator':4, 'Other':5})

    # Encoding categorical variables
    categorical_columns = ['brand', 'fuel_type', 'ext_col', 'int_col']
    lb = LabelEncoder()


        # Encode categorical variables using loaded encoders
    categorical_columns = ['brand', 'fuel_type', 'ext_col', 'int_col']
    for col in categorical_columns:
        if col in user_data_df.columns:
            if col in encoder:
                le = encoder[col]
                user_data_df[col] = le.transform(user_data_df[col])

        print("Current formulation of the user data: \n", user_data_df.head())

        prediction = model.predict(user_data_df)
        return f"{prediction[0]} USD"


    inputs = [
        gr.Number(label="Milage"),
        gr.Number(label="Horsepower"),
        gr.Number(label="Displacement"),
        gr.Number(label="Cylinders"),
        gr.Number(label="Model Year"),
        gr.Dropdown(choices=["Toyota", "Mercedes", "Volvo"], label="Brand"),
        gr.Dropdown(choices=["Gasoline", "Diesel", "Electric", "Hybrid", "Flex Fuel"], label="Fuel Type"),
        gr.Dropdown(choices=["Black", "Gray", "Red"], label="Exterior Color"),
        gr.Dropdown(choices=["Black", "Gray", "White"], label="Interior Color"),
        gr.Dropdown(choices=["Automatic", "Manual", "Variator", "Dual Clutch", "Other"], label="Transmission"),
        gr.Dropdown(choices=["None reported", "At least 1 accident or damage reported"], label="Accident"),
        gr.Dropdown(choices=["Yes", "No"], label="Clean Title")
    ]


    # Specify the inputs as a list of two textboxes
    demo = gr.Interface(
        fn=predict_new_data,
        inputs=inputs,
        # brand, year, milage, fuel_type, transmission, exterior_color, interior_color, accident, clean_title, horsepower, displacement, cylinder
        # drop down, int, int, drop town, drop down, drop down, drop down, drop down, drop down, int, int, int
        title = "Car Price Prediction",
        outputs=["text"],
    )


    if __name__ == "__main__":
        demo.launch()
