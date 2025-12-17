from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('stroke_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure 'stroke_model.pkl' and 'scaler.pkl' are in the root directory.")
    model = scaler = None

def process_form_data(form):
    """
    Processes form data, encodes it, and prepares it for the model.
    Returns a dictionary of raw values and a list of features for the model.
    """
    # Numerical features
    raw_values = {
        'age': float(form['age']),
        'hypertension': int(form['hypertension']),
        'heart_disease': int(form['heart_disease']),
        'avg_glucose_level': float(form['avg_glucose_level']),
        'bmi': float(form['bmi'])
    }

    # Categorical features
    gender = form['gender']
    ever_married = form['ever_married']
    work_type = form['work_type']
    residence_type = form['residence_type']
    smoking_status = form['smoking_status']

    # Add categorical to raw_values for feature importance logic
    raw_values['smoking_status'] = smoking_status
    raw_values['work_type'] = work_type
    raw_values['residence_type'] = residence_type

    # --- Encoding ---
    # Gender: Male, Other (Female is baseline)
    gender_male = 1 if gender == 'Male' else 0
    gender_other = 1 if gender == 'Other' else 0

    # Ever Married: Yes (No is baseline)
    ever_married_yes = 1 if ever_married == 'Yes' else 0

    # Work Type: Private, Self-employed, children, Never_worked (Govt_job is baseline)
    work_type_private = 1 if work_type == 'Private' else 0
    work_type_self_employed = 1 if work_type == 'Self-employed' else 0
    work_type_children = 1 if work_type == 'Children' else 0
    work_type_never_worked = 1 if work_type == 'Never worked' else 0

    # Residence Type: Urban (Rural is baseline)
    residence_type_urban = 1 if residence_type == 'Urban' else 0

    # Smoking Status: formerly smoked, never smoked, smokes (Unknown is baseline)
    smoking_formerly = 1 if smoking_status == 'formerly smoked' else 0
    smoking_never = 1 if smoking_status == 'never smoked' else 0
    smoking_smokes = 1 if smoking_status == 'smokes' else 0

    # Combine all features in the correct order for the model
    features = [
        raw_values['age'], raw_values['hypertension'], raw_values['heart_disease'], 
        raw_values['avg_glucose_level'], raw_values['bmi'],
        gender_male, gender_other, ever_married_yes,
        work_type_private, work_type_self_employed, work_type_children, work_type_never_worked,
        residence_type_urban,
        smoking_formerly, smoking_never, smoking_smokes
    ]
    
    return raw_values, features

def get_top_risk_factors(raw_values):
    """
    Identifies the top 3 risk factors based on the user's input.
    This is a simplified heuristic and not based on model feature importance.
    """
    factors = []
    
    # Age
    if raw_values['age'] > 60:
        factors.append({'name': 'Advanced Age', 'value': int(raw_values['age']), 'impact': 3})
    
    # Hypertension
    if raw_values['hypertension'] == 1:
        factors.append({'name': 'Hypertension', 'value': 'Yes', 'impact': 3})
        
    # Heart Disease
    if raw_values['heart_disease'] == 1:
        factors.append({'name': 'Heart Disease', 'value': 'Yes', 'impact': 3})
        
    # BMI
    if raw_values['bmi'] >= 30:
        factors.append({'name': 'High BMI', 'value': raw_values['bmi'], 'impact': 2})
    elif raw_values['bmi'] >= 25:
        factors.append({'name': 'Overweight', 'value': raw_values['bmi'], 'impact': 1})
        
    # Glucose Level
    if raw_values['avg_glucose_level'] > 140: # Often indicates hyperglycemia
        factors.append({'name': 'High Glucose', 'value': raw_values['avg_glucose_level'], 'impact': 2})
        
    # Smoking
    if raw_values['smoking_status'] == 'smokes':
        factors.append({'name': 'Smoking', 'value': 'Currently smokes', 'impact': 2})

    return sorted(factors, key=lambda x: x['impact'], reverse=True)[:3]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not model or not scaler:
            return "Model not loaded", 500
            
        raw_values, features = process_form_data(request.form)
        scaled_input = scaler.transform([features])
        
        # Get probability instead of just prediction
        probability = model.predict_proba(scaled_input)[0][1] # Probability of class 1 (stroke)
        prediction_val = 1 if probability >= 0.1 else 0 # Use a 10% threshold for "High Risk"
        result = 'High Risk of Stroke' if prediction_val == 1 else 'Low Risk of Stroke'

        # Get top risk factors
        risk_factors = get_top_risk_factors(raw_values)
        
        # If the request is AJAX, return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'prediction': result, 'probability': round(probability * 100, 1), 'risk_factors': risk_factors})
        
        # Otherwise, render the template with the result (for non-JS users)
        return render_template('index.html', prediction=result) 

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)