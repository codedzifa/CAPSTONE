# IMPORT LIBRARIES
import gradio as gr
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from xgboost import XGBClassifier



# Function to load ML toolkit
def load_ml_toolkit(file_path):
    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit(r"export/App_toolkit.pkl")

encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]

# Import the model
model = XGBClassifier()
model.load_model(r"xgb_model.json")


#Colmuns to work with 
input_cols = ["tenure", "montant", "frequence_rech", "arpu_segment", "frequence", "data_volume", "regularity", "freq_top_pack"]
columns_to_scale = ["montant", "frequence_rech", "arpu_segment", "frequence", "data_volume", "regularity", "freq_top_pack"]
categoricals = ["tenure"]



# Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
   
    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=input_cols)

    # Encode the categorical column
    input_data["tenure"] = encoder.transform(input_data["tenure"])
    
    # Scale the numeric columns
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

    # Making the prediction
    model_output = model.predict(input_data)
    return {"Prediction: CHURN": float(model_output[0]), "Prediction: STAY": 1-float(model_output[0])}


#App Interface
with gr.Blocks() as turn_on_the_gradio:
    gr.Markdown("# 📞 EXPRESSO TELECOM CUSTOMER CHURN ☎️")
    gr.Markdown('''
        
        ## WELCOME CHERISHED USER👋 
        
        ### PLEASE GO AHEAD AND MAKE A PREDICTION 🙂''')
    
    # Receiving Inputs
    
    gr.Markdown("**SECTION ONE**")
    gr.Markdown("**CUSTOMER NETWORK ACTTIVITIES**")
    with gr.Row():
        montant = gr.Slider(label="Top-up amount", minimum=20, step=1, interactive=True, value=1, maximum= 500000)
        data_volume = gr.Slider(label="Number of connections", minimum=0, step=1, interactive=True, value=1, maximum= 2000000)

   
    with gr.Row():
        frequence_rech = gr.Slider(label="Recharge Frequency", minimum=1, step=1, interactive=True, value=1, maximum=220)
        freq_top_pack = gr.Slider(label="Top Package Activation Frequency", minimum=1, step=1, interactive=True, value=1, maximum=1050)
        regularity = gr.Slider(label="Regularity (out of 90 days)", minimum=1, step=1, interactive=True, value=1, maximum=90)        
        tenure = gr.Dropdown(label="Tenure (time on the network)", choices=["D 3-6 month", "E 6-9 month", "F 9-12 month", "G 12-15 month", "H 15-18 month", "I 18-21 month", "J 21-24 month", "K > 24 month"], value="K > 24 month")


    gr.Markdown("**SECTION 2**")
    gr.Markdown("**CUSTOMER INCOME DETAILS**")
    with gr.Row():
        arpu_segment = gr.Slider(label="Income over the last 90 days", step=1, maximum=287000, interactive=True)
        frequence = gr.Slider(label="Number of times the customer has made an income", step=1, minimum=1, maximum=91, interactive=True)

    # Output Prediction
    output = gr.Label("...")
    submit_button = gr.Button("Submit")
    
   
    
    submit_button.click(fn = process_and_predict,
                        outputs = output,
                        inputs=[tenure, montant, frequence_rech, arpu_segment, frequence, data_volume, regularity, freq_top_pack])

turn_on_the_gradio.launch(inbrowser= True)