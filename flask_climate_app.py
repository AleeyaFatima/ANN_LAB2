"""
ANN Lab Task 02 - Flask Web App for Climate Change Impact Prediction
Author: Aleeya Fatima (24108108)
Department of Robotics & Artificial Intelligence, SZABIST Islamabad
"""

import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset and prepare model
data = pd.read_csv("realistic_climate_change_impacts.csv")
middle = data["EconomicImpact_USD"].median()
data["HighImpact"] = (data["EconomicImpact_USD"] > middle).astype(int)

features = ["TemperatureAnomaly_C", "CO2Level_ppm", "PopulationAffected"]
target = "HighImpact"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

def impact_check(temperature, co2, population):
    new_data = pd.DataFrame([[temperature, co2, population]], columns=features)
    new_data_scaled = scaler.transform(new_data)
    answer = clf.predict(new_data_scaled)[0]
    chance = clf.predict_proba(new_data_scaled)[0][1] * 100
    if answer == 1:
        return f"High Economic Impact ({chance:.1f}% probability of high)"
    else:
        return f"Low Economic Impact ({100 - chance:.1f}% probability of low)"

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Climate Change Impact Predictor - Aleeya Fatima</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body {
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            color: #f8fafc;
        }
        .container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 36px;
            width: 100%;
            max-width: 480px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        }
        .badge {
            display: inline-block;
            background: rgba(52, 211, 153, 0.2);
            color: #34d399;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        h1 {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #ffffff;
        }
        p.subtitle {
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 24px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #cbd5e1;
        }
        input[type="number"], input[type="text"] {
            width: 100%;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            color: #ffffff;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
        }
        input:focus {
            border-color: #34d399;
            box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.25);
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            border: none;
            border-radius: 10px;
            color: #ffffff;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 10px;
            box-shadow: 0 4px 15px rgba(5, 150, 105, 0.4);
            transition: transform 0.15s ease, background 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #047857 0%, #065f46 100%);
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
            font-size: 16px;
            {% if "High Economic Impact" in result %}
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
            {% elif "Low Economic Impact" in result %}
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #86efac;
            {% else %}
            display: none;
            {% endif %}
        }
        .footer {
            margin-top: 24px;
            text-align: center;
            font-size: 12px;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">ANN Lab Task 02 • Logistic Regression</span>
        <h1>Climate Impact Predictor</h1>
        <p class="subtitle">Forecast economic severity (High vs. Low) from environmental metrics</p>

        <form action="/predict" method="post">
            <div class="form-group">
                <label>Temperature Anomaly (°C)</label>
                <input type="number" step="any" name="temperature" value="{{ temperature }}" placeholder="e.g. 1.8" required>
            </div>
            <div class="form-group">
                <label>CO2 Level (ppm)</label>
                <input type="number" step="any" name="co2" value="{{ co2 }}" placeholder="e.g. 415.2" required>
            </div>
            <div class="form-group">
                <label>Population Affected</label>
                <input type="number" step="any" name="population" value="{{ population }}" placeholder="e.g. 750000" required>
            </div>
            <button type="submit">Check Economic Impact</button>
        </form>

        <div class="result-box">
            {{ result }}
        </div>

        <div class="footer">
            Aleeya Fatima (24108108) • SZABIST Islamabad
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, result="", temperature="", co2="", population="")

@app.route("/predict", methods=["POST"])
def predict():
    temperature = float(request.form.get("temperature", 0))
    co2 = float(request.form.get("co2", 0))
    population = float(request.form.get("population", 0))
    result = impact_check(temperature, co2, population)
    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        temperature=temperature,
        co2=co2,
        population=population
    )

if __name__ == "__main__":
    print("Starting Climate Impact Prediction Web App on http://127.0.0.1:5005 ...")
    app.run(port=5005, debug=True)
