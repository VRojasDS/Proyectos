from flask import Flask, render_template, request
from pred import prediccion, stock_pred, predict_car_price
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        datos = {
            "Age": request.form["Age"],
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingBP": request.form["RestingBP"],
            "Cholesterol": request.form["Cholesterol"],
            "FastingBS": request.form["FastingBS"],
            "RestingECG": request.form["RestingECG"],
            "MaxHR": request.form["MaxHR"],
            "ExerciseAngina": request.form["ExerciseAngina"],
            "Oldpeak": request.form["Oldpeak"],
            "ST_Slope": request.form["ST_Slope"]
        }

        resultado = prediccion(datos)

        if resultado == 1:
            mensaje = "Tiene enfermedad al corazon"
        else:
            mensaje = "El modelo NO detecta enfermedad cardíaca."

        return render_template("formulario.html", resultado=mensaje)

    return render_template("formulario.html", resultado=None)

@app.route("/stock", methods=["GET", "POST"])
def stock():
    if request.method == "POST":
        datos = {
            "Adj Close": float(request.form["AdjClose"]),
            "Close": float(request.form["Close"]),
            "High": float(request.form["High"]),
            "Low": float(request.form["Low"]),
            "Open": float(request.form["Open"]),
            "Volume": float(request.form["Volume"])
        }

        resultado = stock_pred(datos)

        mensaje = f"La predicción del Close es: {resultado}"
        return render_template("stock_form.html", resultado=mensaje)

    return render_template("stock_form.html", resultado=None)

@app.route('/car', methods=['GET', 'POST'])
def car():
    prediction_text = None
    if request.method == 'POST':
        data = {
            'Present_Price': request.form.get('Present_Price', 0),
            'Kms_Driven': request.form.get('Kms_Driven', 0),
            'Owner': request.form.get('Owner', 0),
            'Fuel_Type': request.form.get('Fuel_Type', 'Petrol'),
            'Seller_Type': request.form.get('Seller_Type', 'Dealer'),
            'Transmission': request.form.get('Transmission', 'Manual'),
            'No_of_years': request.form.get('No_of_years', 0)
        }

        # Predicción en USD
        prediction = predict_car_price(data)
        # Formatear con comas
        prediction_text = f"Precio estimado del auto: ${int(prediction):,} USD"

    return render_template('car.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run()
