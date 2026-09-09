import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load pickle model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# Define your categorical mappings here
CATEGORICAL_OPTIONS = {
    "Category": ["Category A", "Category B", "Category C"],
    "Region": ["North", "South", "East", "West"]
}

# Single Inline HTML/CSS Template with dynamic shadow effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction Interface</title>
    <style>
        :root {
            --bg-color: #f4f7fe;
            --card-bg: #ffffff;
            --primary: #4318ff;
            --primary-hover: #3311db;
            --text-main: #1b2559;
            --text-secondary: #a3edbe;
            --border-color: #e0e5f2;
            --shadow-soft: 0px 18px 40px rgba(112, 144, 176, 0.12);
            --shadow-hover: 0px 24px 48px rgba(67, 24, 255, 0.18);
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            background-color: var(--card-bg);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 480px;
            box-shadow: var(--shadow-soft);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
        }

        .container:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
        }

        h2 {
            color: var(--text-main);
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 26px;
            font-weight: 700;
        }

        p.subtitle {
            color: #7090b0;
            font-size: 14px;
            margin-bottom: 28px;
        }

        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 8px;
        }

        input[type="number"], select {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            font-size: 14px;
            color: var(--text-main);
            background-color: #fff;
            box-sizing: border-box;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            box-shadow: inset 0px 2px 4px rgba(0,0,0,0.02);
        }

        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(67, 24, 255, 0.15);
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--primary);
            color: #fff;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0px 10px 20px rgba(67, 24, 255, 0.25);
            transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--primary-hover);
            box-shadow: 0px 14px 24px rgba(67, 24, 255, 0.35);
        }

        button:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 28px;
            padding: 16px;
            background: #f4f7fe;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border-color);
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.03);
        }

        .result-box h3 {
            margin: 0;
            color: var(--primary);
            font-size: 22px;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Linear Model Prediction</h2>
    <p class="subtitle">Select your values below to trigger predictions.</p>

    <form action="/predict" method="POST">
        <div class="form-group">
            <label for="numerical_feature">Numerical Input Value</label>
            <input type="number" step="any" name="numerical_feature" placeholder="e.g. 10.5" required>
        </div>

        {% for cat_name, options in cat_dict.items() %}
        <div class="form-group">
            <label for="{{ cat_name }}">{{ cat_name }}</label>
            <select name="{{ cat_name }}" id="{{ cat_name }}" required>
                {% for option in options %}
                    <option value="{{ option }}">{{ option }}</option>
                {% endfor %}
            </select>
        </div>
        {% endfor %}

        <button type="submit">Predict</button>
    </form>

    {% if prediction_text %}
    <div class="result-box">
        <h3>{{ prediction_text }}</h3>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, cat_dict=CATEGORICAL_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, cat_dict=CATEGORICAL_OPTIONS, prediction_text="Error: linear.pkl not loaded properly.")
    
    try:
        num_val = float(request.form.get('numerical_feature', 0))
        cat_inputs = {cat: request.form.get(cat) for cat in CATEGORICAL_OPTIONS.keys()}

        # Construct input DataFrame matching categorical type encoding
        input_data = {'numerical_feature': [num_val]}
        for cat_name, cat_val in cat_inputs.items():
            input_data[cat_name] = pd.Categorical([cat_val], categories=CATEGORICAL_OPTIONS[cat_name])

        df_input = pd.DataFrame(input_data)
        
        # Standard raw matrix prediction alternative if One-Hot Encoded manually:
        # feature_array = np.array([num_val, ...])
        # prediction = model.predict(feature_array.reshape(1, -1))[0]

        prediction = model.predict(df_input)[0]
        result_str = f"Result: {round(float(prediction), 4)}"

    except Exception as e:
        result_str = f"Prediction Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, cat_dict=CATEGORICAL_OPTIONS, prediction_text=result_str)

if __name__ == '__main__':
    app.run(debug=True)
