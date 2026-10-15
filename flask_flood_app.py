"""
ANN Lab Task 02 - Flask Web App for Flood Prediction
Author: Aleeya Fatima (24108108)
Department of Robotics & Artificial Intelligence, SZABIST Islamabad
"""

import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset and prepare model
data = pd.read_csv("flood_dataset_classification.csv")
features = ["Rainfall", "Elevation", "Slope", "distance"]
target = "occured"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

def flood_check(rainfall, elevation, slope, distance):
    new_data = pd.DataFrame([[rainfall, elevation, slope, distance]], columns=features)
    new_data_scaled = scaler.transform(new_data)
    answer = clf.predict(new_data_scaled)[0]
    chance = clf.predict_proba(new_data_scaled)[0][1] * 100
    if answer == 1:
        return f"Flood Likely ({chance:.1f}% chance)"
    else:
        return f"No Flood Expected ({100 - chance:.1f}% chance)"

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flood Prediction System - Aleeya Fatima</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
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
            background: rgba(56, 189, 248, 0.2);
            color: #38bdf8;
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
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            border: none;
            border-radius: 10px;
            color: #ffffff;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 10px;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
            transition: transform 0.15s ease, background 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
            font-size: 16px;
            {% if "Likely" in result %}
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
            {% elif "No Flood" in result %}
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
        <h1>Flood Prediction System</h1>
        <p class="subtitle">Predict flood likelihood based on meteorological & topographical data</p>

        <form action="/predict" method="post">
            <div class="form-group">
                <label>Rainfall (mm)</label>
                <input type="number" step="any" name="rainfall" value="{{ rainfall }}" placeholder="e.g. 150.5" required>
            </div>
            <div class="form-group">
                <label>Elevation (meters)</label>
                <input type="number" step="any" name="elevation" value="{{ elevation }}" placeholder="e.g. 120" required>
            </div>
            <div class="form-group">
                <label>Slope</label>
                <input type="number" step="any" name="slope" value="{{ slope }}" placeholder="e.g. 2.4" required>
            </div>
            <div class="form-group">
                <label>Distance from River (meters)</label>
                <input type="number" step="any" name="distance" value="{{ distance }}" placeholder="e.g. 250" required>
            </div>
            <button type="submit">Check Flood Risk</button>
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
    return render_template_string(HTML_TEMPLATE, result="", rainfall="", elevation="", slope="", distance="")

@app.route("/predict", methods=["POST"])
def predict():
    rainfall = float(request.form.get("rainfall", 0))
    elevation = float(request.form.get("elevation", 0))
    slope = float(request.form.get("slope", 0))
    distance = float(request.form.get("distance", 0))
    result = flood_check(rainfall, elevation, slope, distance)
    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        rainfall=rainfall,
        elevation=elevation,
        slope=slope,
        distance=distance
    )

if __name__ == "__main__":
    print("Starting Flood Prediction Web App on http://127.0.0.1:5004 ...")
    app.run(port=5004, debug=True)
