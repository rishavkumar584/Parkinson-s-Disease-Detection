from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the model
df = pd.read_csv("parkinsons.csv")
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

# Standardize the data
ss = StandardScaler()
X_scaled = ss.fit_transform(X)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_scaled, Y)

# Feature names
feature_names = X.columns.tolist()

# HTML template
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson's Disease Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        h1, h2 {
            text-align: center;
            color: #333;
        }
        label {
            font-size: 1em;
            margin-bottom: 5px;
            display: block;
            color: #555;
        }
        textarea, input {
            width: 100%;
            font-size: 1em;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            display: block;
            width: 100%;
            padding: 15px;
            background-color: #007BFF;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            font-size: 1.2em;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            font-size: 1.2em;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Parkinson's Disease Prediction</h1>

        <!-- Part 1: Textarea Input -->
        <h2>Part 1: Predict using Comma-Separated Values</h2>
        <form method="POST" action="/part1">
            <label for="features">Enter Features (comma-separated):</label>
            <textarea id="features" name="features" placeholder="Enter 22 feature values here...">{{ features }}</textarea>
            <button type="submit">Predict</button>
        </form>

        <!-- Part 2: Individual Feature Inputs -->
        <h2>Part 2: Predict using Individual Feature Inputs</h2>
        <form method="POST" action="/part2">
            {% for feature in feature_names %}
                <label for="{{ feature }}">{{ feature }}</label>
                <input type="text" id="{{ feature }}" name="{{ feature }}" placeholder="Enter {{ feature }} value" value="{{ request.form.get(feature, '') }}">
            {% endfor %}
            <button type="submit">Predict</button>
        </form>

        {% if result is not none %}
        <div class="result">
            <strong>Prediction Result:</strong> {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template_string(html_template, result=None, features="", feature_names=feature_names)

@app.route('/part1', methods=['POST'])
def part1():
    result = None
    features = request.form.get('features', "")

    try:
        input_features = [float(x.strip()) for x in features.split(',') if x.strip()]
        
        if len(input_features) != len(feature_names):
            raise ValueError("Incorrect number of features provided. Expected {} values.".format(len(feature_names)))

        input_data_np = np.asarray(input_features).reshape(1, -1)
        input_data_scaled = ss.transform(input_data_np)
        prediction = model.predict(input_data_scaled)[0]
        result = 'Positive, Parkinson\'s disease found' if prediction == 1 else 'Negative, No Parkinson\'s disease found'
    
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template_string(html_template, result=result, features=features, feature_names=feature_names)

@app.route('/part2', methods=['POST'])
def part2():
    result = None

    try:
        input_features = [float(request.form.get(feature, 0)) for feature in feature_names]

        input_data_np = np.asarray(input_features).reshape(1, -1)
        input_data_scaled = ss.transform(input_data_np)
        prediction = model.predict(input_data_scaled)[0]
        result = 'Positive, Parkinson\'s disease found' if prediction == 1 else 'Negative, No Parkinson\'s disease found'
    
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template_string(html_template, result=result, features="", feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
