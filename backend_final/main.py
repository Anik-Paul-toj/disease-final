from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load artifacts
model = joblib.load('random_forest_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# Load dataset
dataset = pd.read_csv("updated_medical_dataset.csv")

# Extract unique symptoms
valid_symptoms = set()
for symptoms in dataset['Symptoms']:
    valid_symptoms.update(symptom.strip() for symptom in symptoms.split(','))
valid_symptoms = list(valid_symptoms)

# Medication mapping
disease_to_medications = {
    "Flu": {"medications": ["Paracetamol 500mg", "Antihistamine 10mg"], "consult": "No"},
    "Diabetes": {"medications": ["Metformin 500mg", "Insulin 10 units"], "consult": "Yes"},
    "Hypertension": {"medications": ["Amlodipine 5mg", "Losartan 50mg"], "consult": "Yes"},
    "Asthma": {"medications": ["Salbutamol 100mcg", "Budesonide 200mcg"], "consult": "Yes"},
    "Migraine": {"medications": ["Ibuprofen 400mg", "Sumatriptan 50mg"], "consult": "No"},
    "COVID-19": {"medications": ["Paracetamol 500mg", "Vitamin C 1000mg"], "consult": "Yes"},
    "Hepatitis": {"medications": ["Tenofovir 300mg", "Entecavir 0.5mg"], "consult": "Yes"},
    "Osteoarthritis": {"medications": ["Paracetamol 500mg", "Diclofenac 50mg"], "consult": "Yes"},
    "Depression": {"medications": ["Fluoxetine 20mg", "Sertraline 50mg"], "consult": "Yes"},
    "Anxiety": {"medications": ["Alprazolam 0.5mg", "Clonazepam 0.25mg"], "consult": "Yes"},
    "Thyroid Disorder": {"medications": ["Levothyroxine 50mcg", "Methimazole 5mg"], "consult": "Yes"},
    "Pneumonia": {"medications": ["Azithromycin 500mg", "Ceftriaxone 1g"], "consult": "Yes"},
    "Crohn's Disease": {"medications": ["Mesalamine 800mg", "Prednisone 20mg"], "consult": "Yes"},
    "Allergic Rhinitis": {"medications": ["Levocetirizine 5mg", "Nasal Corticosteroids"], "consult": "No"},
    "Urinary Tract Infection (UTI)": {"medications": ["Nitrofurantoin 100mg", "Ciprofloxacin 500mg"], "consult": "Yes"},
    "Gastritis": {"medications": ["Omeprazole 20mg", "Ranitidine 150mg"], "consult": "No"},
    "Acne": {"medications": ["Benzoyl Peroxide 5%", "Clindamycin Gel"], "consult": "No"},
    "Eczema": {"medications": ["Hydrocortisone Cream", "Tacrolimus Ointment"], "consult": "No"},
    "Anemia": {"medications": ["Ferrous Sulfate 325mg", "Folic Acid 5mg"], "consult": "Yes"},
    "Chronic Kidney Disease": {"medications": ["Lisinopril 10mg", "Furosemide 40mg"], "consult": "Yes"},
    "Parkinson's Disease": {"medications": ["Levodopa 250mg", "Carbidopa 25mg"], "consult": "Yes"},
    "Multiple Sclerosis": {"medications": ["Interferon Beta 1a 30mcg", "Glatiramer Acetate 20mg"], "consult": "Yes"},
    "HIV/AIDS": {"medications": ["Tenofovir 300mg", "Emtricitabine 200mg"], "consult": "Yes"},
    "Tuberculosis": {"medications": ["Isoniazid 300mg", "Rifampicin 600mg"], "consult": "Yes"}
}

# Input schema
class DiseasePredictionRequest(BaseModel):
    age: int
    gender: str
    duration: int
    symptoms: list

# Input validation
def validate_input(data):
    if data.age <= 0 or data.duration <= 0:
        raise HTTPException(status_code=400, detail="Age and duration must be positive.")
    if data.gender not in ["Male", "Female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'Male' or 'Female'.")
    if not data.symptoms or any(s not in valid_symptoms for s in data.symptoms):
        raise HTTPException(status_code=400, detail="Invalid or missing symptoms.")

# Prediction endpoint
@app.post("/predict")
def predict_disease(request: DiseasePredictionRequest):
    validate_input(request)

    # Prepare input DataFrame
    symptoms_str = ', '.join(sorted(request.symptoms))
    input_df = pd.DataFrame({
        'Age': [request.age],
        'Gender': [request.gender],
        'Disease Duration (days)': [request.duration],
        'Symptoms': [symptoms_str]
    })

    # Encode gender
    input_df['Gender'] = encoder.transform(input_df['Gender'])

    # One-hot encode symptoms
    for symptom in valid_symptoms:
        col_name = f'Symptom_{symptom}'
        input_df[col_name] = 1 if symptom in request.symptoms else 0

    # Ensure correct column order and fill missing
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Scale numeric features
    input_df[['Age', 'Disease Duration (days)']] = scaler.transform(
        input_df[['Age', 'Disease Duration (days)']]
    )

    # Prediction
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]
    predicted_disease = prediction[0]
    predicted_prob = round(probabilities[model.classes_.tolist().index(predicted_disease)], 2)

    # Top 5 predictions
    top_preds = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:5]
    top_predictions = [{"disease": d, "probability": round(p, 2)} for d, p in top_preds]

    # Medication recommendation
    disease_info = disease_to_medications.get(predicted_disease, {
        "medications": ["No specific medications found"],
        "consult": "Yes"
    })

    return {
        "predicted_disease": predicted_disease,
        "probability": predicted_prob,
        "top_predictions": top_predictions,
        "medications": disease_info["medications"],
        "consult_doctor": disease_info["consult"]
    }
