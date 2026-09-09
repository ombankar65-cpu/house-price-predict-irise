import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained linear regression model
try:
    with open('linear.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception:
    model = None

# Integrated HTML/CSS with glassmorphism layout and drop-shadow depth effects
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            color: #f8fafc;
            padding: 30px 15px;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.125);
            border-radius: 20px;
            padding: 35px;
            width: 100%;
            max-width: 550px;
            /* Layered drop-shadow effects for visual depth */
            box-shadow: 
                0 20px 25px -5px rgba(0, 0, 0, 0.5),
                0 8px 10px -6px rgba(0, 0, 0, 0.3),
                0 0 40px rgba(99, 102, 241, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 
                0 25px 30px -5px rgba(0, 0, 0, 0.6),
                0 12px 15px -6px rgba(0, 0, 0, 0.4),
                0 0 50px rgba(99, 102, 241, 0.25);
        }

        h2 {
            text-align: center;
            margin-bottom: 24px;
            font-size: 1.8rem;
            letter-spacing: 0.5px;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .full-width {
            grid-column: span 2;
        }

        .form-group {
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 6px;
            font-size: 0.85rem;
            color: #cbd5e1;
        }

        input[type="number"] {
            width: 100%;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input[type="number"]:focus {
            border-color: #818cf8;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.4),
                0 0 0 3px rgba(129, 140, 248, 0.25);
        }

        button {
            width: 100%;
            padding: 14px;
            margin-top: 10px;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
            transition: opacity 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
        }

        button:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.55);
        }

        button:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .result-title {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }

        .result-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #38bdf8;
        }

        @media (max-width: 480px) {
            .grid-container {
                grid-template-columns: 1fr;
            }
            .full-width {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>House Price Predictor</h2>
        <form method="POST" action="/predict">
            <div class="grid-container">
                <div class="form-group">
                    <label>Square Footage</label>
                    <input type="number" step="any" name="Square_Footage" required placeholder="e.g. 1500">
                </div>
                <div class="form-group">
                    <label>Bedrooms</label>
                    <input type="number" step="any" name="Num_Bedrooms" required placeholder="e.g. 3">
                </div>
                <div class="form-group">
                    <label>Bathrooms</label>
                    <input type="number" step="any" name="Num_Bathrooms" required placeholder="e.g. 2">
                </div>
                <div class="form-group">
                    <label>Year Built</label>
                    <input type="number" step="any" name="Year_Built" required placeholder="e.g. 2018">
                </div>
                <div class="form-group">
                    <label>Lot Size (sq ft)</label>
                    <input type="number" step="any" name="Lot_Size" required placeholder="e.g. 5000">
                </div>
                <div class="form-group">
                    <label>Garage Size (cars)</label>
                    <input type="number" step="any" name="Garage_Size" required placeholder="e.g. 2">
                </div>
                <div class="form-group full-width">
                    <label>Neighborhood Quality (1-10)</label>
                    <input type="number" step="any" name="Neighborhood_Quality" required placeholder="e.g. 8">
                </div>
            </div>
            <button type="submit">Predict Price</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <div class="result-title">Estimated Value</div>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_LAYOUT, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_LAYOUT, prediction="Error: Model linear.pkl not found or unreadable.")
    
    try:
        # Extract features matching the model's exact expected sequence
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        final_input = np.array([features])
        pred = model.predict(final_input)[0]
        output = f"${round(float(pred), 2):,}"
        
        return render_template_string(HTML_LAYOUT, prediction=output)
    except Exception as e:
        return render_template_string(HTML_LAYOUT, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
