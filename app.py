import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

expected_inputs = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                   'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK', 'TENURE_NUMERIC']

# Function to load machine learning components
def load_components_func(fp):
    # To load the machine learning components saved to re-use in the app
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object

# Loading the machine learning components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "ML Components", "ML_Model.pkl")
ml_components_dict = load_components_func(fp=ml_core_fp)

# Defining the variables for each component
scaler = ml_components_dict['scaler']
model = ml_components_dict['model']

def predict_churn(MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, DATA_VOLUME,
                  ON_NET, ORANGE, TIGO, ZONE1, ZONE2, REGULARITY, FREQ_TOP_PACK, TENURE,
                  scaler=scaler, model=model):

    # Function to map the tenure choices to numeric values
    def map_tenure_to_numeric(tenure_choice):
        return {
            'K > 24 month': 25,
            'I 18-21 month': 19.5,
            'H 15-18 month': 16.5,
            'G 12-15 month': 13.5,
            'J 21-24 month': 22.5,
            'F 9-12 month': 10.5,
            'E 6-9 month': 8,
            'D 3-6 month': 4.5
        }[tenure_choice]

    # Map the tenure choices to numeric values
    tenure_numeric = map_tenure_to_numeric(TENURE)

    input_data = pd.DataFrame([[MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, DATA_VOLUME,
                                ON_NET, ORANGE, TIGO, ZONE1, ZONE2, REGULARITY, FREQ_TOP_PACK, tenure_numeric]],
                              columns=expected_inputs)

    # Encode the data
    scaled_df = scaler.transform(input_data)

    # Prediction
    model_output = model.predict_proba(scaled_df)
    # Probability of Churn (Positive class)
    prob_Churn = float(model_output[0][1])
    # Probability of staying (Negative Class)
    prob_Stay = 1 - prob_Churn
    return {"Prediction Churn": prob_Churn,
            "Prediction Not Churn": prob_Stay}

# Replace the existing Tenure encoding with the provided list
tenure_choices = ['K > 24 month', 'I 18-21 month', 'H 15-18 month', 'G 12-15 month',
                  'J 21-24 month', 'F 9-12 month', 'E 6-9 month', 'D 3-6 month']

with gr.Blocks() as demo:
    gr.Markdown('''
    # 📞 Telecom Customer Churn Prediction App ☎️ ''')
    gr.Markdown('''
    
    ## Welcome Cherished User 👋 
    
    ### Please Predict Customer Churn 🙂''')

    with gr.Row():
        Data_Volume = gr.Number(label='Data Volume', step=1)
        On_Net = gr.Number(label='On Net', step=1)
        Orange = gr.Number(label='Orange', step=1)
        Tigo = gr.Number(label='Tigo', step=1)
        Zone1 = gr.Number(label='Zone1', step=1)
        Zone2 = gr.Number(label='Zone2', step=1)
        MONTANT = gr.Number(label='Top-up Amount', step=1)
        FREQUENCE_RECH = gr.Number(label='Recharge Frequency', step=1)
        REVENUE = gr.Number(label='Monthly Revenue', step=1)
        ARPU_SEGMENT = gr.Number(label='ARPU Segment', step=1)
        FREQUENCE = gr.Number(label='Frequency of Transactions', step=1)
        REGULARITY = gr.Number(label='Regularity', step=1)
        FREQ_TOP_PACK = gr.Number(label='Frequency of Top Pack Activation', step=1)
        TENURE = gr.Dropdown(label='Tenure', choices=tenure_choices)

    submit_button = gr.Button('Predict')

    with gr.Row():
        with gr.Accordion('Churn Prediction'):
            output1 = gr.Slider(maximum=1, minimum=0, value=1, label='Churn (Yes)')
            output2 = gr.Slider(maximum=1, minimum=0, value=1, label='No Churn (No)')

    submit_button.click(fn=predict_churn, inputs=[Data_Volume, On_Net, Orange, Tigo, Zone1, Zone2, MONTANT,
                                                   FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, REGULARITY,
                                                   FREQ_TOP_PACK, TENURE],
                        outputs=[output1, output2])

demo.launch()
