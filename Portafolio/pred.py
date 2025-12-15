import torch
import pandas as pd
import numpy as np
import joblib
from torch import nn
import datetime


def one_hot_encode(value, categories):
    vec = [1 if value == cat else 0 for cat in categories]
    return vec

def prediccion(data_dict):
    model = joblib.load("P1/heartdisease.joblib")

    CHESTPAIN_MAP = ["ATA", "NAP", "ASY", "TA"]
    RESTINGECG_MAP = ["Normal", "ST", "LVH"]
    ST_SLOPE_MAP = ["Up", "Flat", "Down"]

    Age = float(data_dict["Age"])
    Sex = 1 if data_dict["Sex"] == "M" else 0
    RestingBP = float(data_dict["RestingBP"])
    Cholesterol = float(data_dict["Cholesterol"])
    FastingBS = int(data_dict["FastingBS"])
    MaxHR = float(data_dict["MaxHR"])
    ExerciseAngina = 1 if data_dict["ExerciseAngina"] == "Y" else 0
    Oldpeak = float(data_dict["Oldpeak"])

    cp_vec = one_hot_encode(data_dict["ChestPainType"], CHESTPAIN_MAP)
    ecg_vec = one_hot_encode(data_dict["RestingECG"], RESTINGECG_MAP)
    slope_vec = one_hot_encode(data_dict["ST_Slope"], ST_SLOPE_MAP)

    features = [
        Age, Sex, RestingBP, Cholesterol, FastingBS, MaxHR,
        ExerciseAngina, Oldpeak,
        *cp_vec,
        *ecg_vec,
        *slope_vec
    ]

    X = np.array([features])

    pred = model.predict(X)[0]
    return int(pred) 

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def stock_pred(datos):
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    state_dict = torch.load("P1/stock.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load("P1/scaler.pkl")
    close_val = float(datos["Close"])

    seq = np.array([close_val for _ in range(50)]).reshape(-1, 1)
    data_scaled = scaler.transform(seq)

    x = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)

    # Predicción
    with torch.no_grad():
        pred_scaled = model(x).numpy()[0][0]

    pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Desescalar
    min_close, max_close = 3.177, 135.58
    pred_real = max(min(pred_real, max_close), min_close)

    return pred_real

def preprocess_input(data):
    
    Petrol = 1 if data['Fuel_Type'] == 'Petrol' else 0
    Diesel = 1 if data['Fuel_Type'] == 'Diesel' else 0
    CNG = 1 if data['Fuel_Type'] == 'CNG' else 0
    
    Dealer = 1 if data['Seller_Type'] == 'Dealer' else 0
    Individual = 1 if data['Seller_Type'] == 'Individual' else 0
    
    Manual = 1 if data['Transmission'] == 'Manual' else 0
    Automatic = 1 if data['Transmission'] == 'Automatic' else 0
    
    # Construir array en el orden que espera el modelo
    features = [  # Selling_Price
        float(data['Present_Price']),
        int(data['Kms_Driven']),
        int(data['Owner']),
        Petrol, Diesel, CNG,
        Dealer, Individual,
        Manual, Automatic,
        int(data['No_of_years'])
    ]
    return features

def predict_car_price(data):
    model = joblib.load("car_price.joblib")
    features = preprocess_input(data)
    prediction = model.predict([features])
    return round(float(prediction[0]), 2)