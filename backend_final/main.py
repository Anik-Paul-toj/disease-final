from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator
import joblib
import pandas as pd
from typing import List

app = FastAPI()

# Load artifacts
model = joblib.load('random_forest_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# Load dataset to extract symptoms
dataset = pd.read_csv("updated_medical_dataset.csv")
valid_symptoms = set()
for symptoms in dataset['Symptoms']:
    valid_symptoms.update(symptom.strip() for symptom in symptoms.split(','))
valid_symptoms = list(valid_symptoms)

# Medications mapping
disease_to_medications = {
    "Flu": {"medications": ["Paracetamol 500mg", "Antihistamine 10mg"], "consult": "No"},
    "Diabetes": {"medications": ["Metformin 500mg", "Insulin 10 units"], "consult": "Yes"},
    "Hypertension": {"medications": ["Amlodipine 5mg", "Losartan 50mg"], "consult": "Yes"},
    "Asthma": {"medications": ["Salbutamol 100mcg", "Budesonide 200mcg"], "consult": "Yes"},
    "Migraine": {"medications": ["Ibuprofen 400mg", "Sumatriptan 50mg"], "consult": "No"},
    # Add more from your original dictionary...
}

# Request schema
class DiseasePredictionRequest(BaseModel):
    age: int
    gender: str
    duration: int
    symptoms: List[str]

    @root_validator(pre=True)
    def check_and_log(cls, values):
        print(f"Received input: {values}")
        return values

# Input validation
def validate_input(data):
    if data.age <= 0 or data.duration <= 0:
        raise HTTPException(status_code=400, detail="Age and duration must be positive.")
    if data.gender not in ["Male", "Female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'Male' or 'Female'.")
    if not data.symptoms or any(s not in valid_symptoms for s in data.symptoms):
        raise HTTPException(status_code=400, detail=f"Invalid or missing symptoms. Valid ones include: {', '.join(valid_symptoms)}")

# Endpoint
@app.post("/predict")
def predict_disease(request: DiseasePredictionRequest):
    validate_input(request)

    # Prepare input
    symptoms_str = ', '.join(sorted(request.symptoms))
    input_df = pd.DataFrame({
        'Age': [request.age],
        'Gender': [request.gender],
        'Disease Duration (days)': [request.duration],
        'Symptoms': [symptoms_str]
    })

    # Encode gender
    input_df['Gender'] = encoder.transform(input_df['Gender'])

    # Add symptom columns
    for symptom in valid_symptoms:
        input_df[f'Symptom_{symptom}'] = 1 if symptom in request.symptoms else 0

    # Reorder and fill
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Scale numeric
    input_df[['Age', 'Disease Duration (days)']] = scaler.transform(
        input_df[['Age', 'Disease Duration (days)']]
    )

    # Prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_prob = round(probabilities[model.classes_.tolist().index(prediction)], 2)

    top_preds = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:5]
    top_predictions = [{"disease": d, "probability": round(p, 2)} for d, p in top_preds]

    disease_info = disease_to_medications.get(prediction, {
        "medications": ["No specific medications found"],
        "consult": "Yes"
    })

    return {
        "predicted_disease": prediction,
        "probability": predicted_prob,
        "top_predictions": top_predictions,
        "medications": disease_info["medications"],
        "consult_doctor": disease_info["consult"]
    }
