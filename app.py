"""
ANN Lab Task 02 - Unified Web Portal
Student: Aleeya Fatima (24108108)
Department of Robotics & Artificial Intelligence, SZABIST Islamabad
Both Models: Flood Risk Predictor & Climate Change Economic Impact Predictor
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template_string

# ----------------- Model 1: Flood Prediction -----------------
flood_df = pd.read_csv("flood_dataset_classification.csv")
features_flood = ["Rainfall", "Elevation", "Slope", "distance"]
target_flood = "occured"

X_flood = flood_df[features_flood]
y_flood = flood_df[target_flood]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_flood, y_flood, test_size=0.25, random_state=7
)

scaler_flood = StandardScaler()
X_train_f_scaled = scaler_flood.fit_transform(X_train_f)
clf_flood = LogisticRegression()
clf_flood.fit(X_train_f_scaled, y_train_f)

def predict_flood(rainfall, elevation, slope, distance):
    sample = pd.DataFrame([[rainfall, elevation, slope, distance]], columns=features_flood)
    sample_scaled = scaler_flood.transform(sample)
    pred = clf_flood.predict(sample_scaled)[0]
    prob = clf_flood.predict_proba(sample_scaled)[0][1] * 100
    if pred == 1:
        return f"Flood Likely ({prob:.1f}% risk)", "danger"
    else:
        return f"No Flood Expected ({100 - prob:.1f}% safety margin)", "success"

# ----------------- Model 2: Climate Impact -----------------
climate_df = pd.read_csv("realistic_climate_change_impacts.csv")
median_impact = climate_df["EconomicImpact_USD"].median()
climate_df["HighImpact"] = (climate_df["EconomicImpact_USD"] > median_impact).astype(int)

features_climate = ["TemperatureAnomaly_C", "CO2Level_ppm", "PopulationAffected"]
target_climate = "HighImpact"

X_climate = climate_df[features_climate]
y_climate = climate_df[target_climate]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_climate, y_climate, test_size=0.25, random_state=7
)

scaler_climate = StandardScaler()
X_train_c_scaled = scaler_climate.fit_transform(X_train_c)
clf_climate = LogisticRegression()
clf_climate.fit(X_train_c_scaled, y_train_c)

def predict_climate(temperature, co2, population):
    sample = pd.DataFrame([[temperature, co2, population]], columns=features_climate)
    sample_scaled = scaler_climate.transform(sample)
    pred = clf_climate.predict(sample_scaled)[0]
    prob = clf_climate.predict_proba(sample_scaled)[0][1] * 100
    if pred == 1:
        return f"High Economic Impact ({prob:.1f}% probability)", "danger"
    else:
        return f"Low Economic Impact ({100 - prob:.1f}% probability of low)", "success"

# ----------------- Flask Unified Web Application -----------------
app = Flask(__name__)

PORTAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANN Lab Task 02 Portal • Aleeya Fatima (24108108)</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.5);
            --bg: #090d16;
            --card-bg: rgba(18, 26, 43, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.25) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 78, 59, 0.25) 0px, transparent 50%);
            min-height: 100vh;
            color: var(--text-main);
            padding: 40px 20px;
        }
        .header {
            text-align: center;
            max-width: 900px;
            margin: 0 auto 40px;
        }
        .header .badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            font-size: 13px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 999px;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .header h1 {
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p {
            font-size: 15px;
            color: var(--text-muted);
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 28px;
            max-width: 1100px;
            margin: 0 auto;
        }
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 32px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
        }
        .card-flood::before {
            background: linear-gradient(90deg, #0284c7, #38bdf8);
        }
        .card-climate::before {
            background: linear-gradient(90deg, #059669, #34d399);
        }
        .card-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card-desc {
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 24px;
            min-height: 38px;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 14px;
        }
        .form-group {
            margin-bottom: 14px;
        }
        label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        input[type="number"], input[type="text"] {
            width: 100%;
            padding: 12px 14px;
            background: rgba(10, 15, 29, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            color: #ffffff;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
        }
        input:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.25);
        }
        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            margin-top: auto;
            transition: transform 0.15s ease, opacity 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            opacity: 0.95;
        }
        .btn-flood {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
        }
        .btn-climate {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            box-shadow: 0 4px 15px rgba(5, 150, 105, 0.4);
        }
        .result {
            margin-top: 20px;
            padding: 14px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 14px;
            text-align: center;
        }
        .result.danger {
            background: rgba(239, 68, 68, 0.18);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
        }
        .result.success {
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #86efac;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 13px;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">SZABIST Islamabad • Artificial Neural Networks</span>
        <h1>ANN Lab Task #02 Predictive Portal</h1>
        <p>Interactive classification interface powered by Logistic Regression. Developed by <strong>Aleeya Fatima</strong> (Reg: 24108108) under the supervision of <strong>Mr. Hassaan Mujtaba</strong>.</p>
    </div>

    <div class="grid">
        <!-- Flood Predictor Card -->
        <div class="card card-flood">
            <h2 class="card-title">🌊 Flood Risk Predictor</h2>
            <p class="card-desc">Evaluates meteorological rainfall and terrain metrics to classify imminent flood risk (Accuracy: ~70.7%).</p>
            
            <form method="POST" action="/#flood">
                <input type="hidden" name="form_type" value="flood">
                <div class="form-row">
                    <div class="form-group">
                        <label>Rainfall (mm)</label>
                        <input type="number" step="any" name="rainfall" value="{{ r_val or '120' }}" required>
                    </div>
                    <div class="form-group">
                        <label>Elevation (m)</label>
                        <input type="number" step="any" name="elevation" value="{{ e_val or '50' }}" required>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Slope</label>
                        <input type="number" step="any" name="slope" value="{{ s_val or '2.5' }}" required>
                    </div>
                    <div class="form-group">
                        <label>Distance to River (m)</label>
                        <input type="number" step="any" name="distance" value="{{ d_val or '150' }}" required>
                    </div>
                </div>
                <button type="submit" class="btn-flood">Run Flood Classifier</button>
            </form>

            {% if flood_res %}
            <div class="result {{ flood_type }}">{{ flood_res }}</div>
            {% endif %}
        </div>

        <!-- Climate Impact Predictor Card -->
        <div class="card card-climate">
            <h2 class="card-title">🌍 Climate Economic Impact</h2>
            <p class="card-desc">Classifies economic loss severity (High vs. Low) using global environmental indicators (Accuracy: ~50.5%).</p>
            
            <form method="POST" action="/#climate">
                <input type="hidden" name="form_type" value="climate">
                <div class="form-group">
                    <label>Temperature Anomaly (°C)</label>
                    <input type="number" step="any" name="temperature" value="{{ t_val or '2.4' }}" required>
                </div>
                <div class="form-group">
                    <label>CO2 Level (ppm)</label>
                    <input type="number" step="any" name="co2" value="{{ c_val or '420' }}" required>
                </div>
                <div class="form-group">
                    <label>Population Affected</label>
                    <input type="number" step="any" name="population" value="{{ p_val or '1200000' }}" required>
                </div>
                <button type="submit" class="btn-climate">Classify Economic Impact</button>
            </form>

            {% if climate_res %}
            <div class="result {{ climate_type }}">{{ climate_res }}</div>
            {% endif %}
        </div>
    </div>

    <div class="footer">
        &copy; 2026 Aleeya Fatima (24108108) • Department of Robotics & Artificial Intelligence • SZABIST Islamabad
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    flood_res, flood_type = None, None
    climate_res, climate_type = None, None
    r_val, e_val, s_val, d_val = None, None, None, None
    t_val, c_val, p_val = None, None, None

    if request.method == "POST":
        form_type = request.form.get("form_type")
        if form_type == "flood":
            r_val = float(request.form.get("rainfall", 0))
            e_val = float(request.form.get("elevation", 0))
            s_val = float(request.form.get("slope", 0))
            d_val = float(request.form.get("distance", 0))
            flood_res, flood_type = predict_flood(r_val, e_val, s_val, d_val)
        elif form_type == "climate":
            t_val = float(request.form.get("temperature", 0))
            c_val = float(request.form.get("co2", 0))
            p_val = float(request.form.get("population", 0))
            climate_res, climate_type = predict_climate(t_val, c_val, p_val)

    return render_template_string(
        PORTAL_HTML,
        flood_res=flood_res, flood_type=flood_type,
        r_val=r_val, e_val=e_val, s_val=s_val, d_val=d_val,
        climate_res=climate_res, climate_type=climate_type,
        t_val=t_val, c_val=c_val, p_val=p_val
    )

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"Starting Unified ANN Lab Task 02 Portal on http://127.0.0.1:{port} ...")
    app.run(port=port, debug=True)
