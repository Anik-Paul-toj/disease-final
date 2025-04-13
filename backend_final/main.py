from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator
import joblib
import pandas as pd
from typing import List
import uvicorn 

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
    "Flu": {
        "medications": ["Paracetamol 500mg", "Antihistamine 10mg"],
        "consult": "No"
    },
    "Diabetes": {
        "medications": ["Metformin 500mg", "Insulin 10 units"],
        "consult": "Yes"
    },
    "Hypertension": {
        "medications": ["Amlodipine 5mg", "Losartan 50mg"],
        "consult": "Yes"
    },
    "Asthma": {
        "medications": ["Salbutamol 100mcg", "Budesonide 200mcg"],
        "consult": "Yes"
    },
    "Migraine": {
        "medications": ["Ibuprofen 400mg", "Sumatriptan 50mg"],
        "consult": "No"
    },
    "Arthritis": {
        "medications": ["Ibuprofen 200mg", "Methotrexate 10mg"],
        "consult": "Yes"
    },
    "Pneumonia": {
        "medications": ["Amoxicillin 500mg", "Azithromycin 250mg"],
        "consult": "Yes"
    },
    "Depression": {
        "medications": ["Sertraline 50mg", "Escitalopram 10mg"],
        "consult": "Yes"
    },
    "Gastroenteritis": {
        "medications": ["Loperamide 2mg", "Oral Rehydration Salts"],
        "consult": "No"
    },
    "Hypothyroidism": {
        "medications": ["Levothyroxine 50mcg", "Levothyroxine 100mcg"],
        "consult": "Yes"
    },
    "Cancer": {
        "medications": ["Chemotherapy", "Immunotherapy", "Pain management medication"],
        "consult": "Yes"
    },
    "Chronic Kidney Disease": {
        "medications": ["Angiotensin-converting enzyme (ACE) inhibitors", "Calcium channel blockers"],
        "consult": "Yes"
    },
    "Cold": {
        "medications": ["Paracetamol 500mg", "Vitamin C 1000mg"],
        "consult": "No"
    },
    "Bronchitis": {
        "medications": ["Cough syrup", "Ibuprofen 200mg"],
        "consult": "No"
    },
    "Chickenpox": {
        "medications": ["Acyclovir 800mg", "Calamine lotion"],
        "consult": "No"
    },
    "Tuberculosis": {
        "medications": ["Isoniazid 300mg", "Rifampicin 10mg"],
        "consult": "Yes"
    },
    "Hepatitis B": {
        "medications": ["Tenofovir 300mg", "Lamivudine 100mg"],
        "consult": "Yes"
    },
    "Malaria": {
        "medications": ["Chloroquine 250mg", "Artemether-Lumefantrine 20/120mg"],
        "consult": "Yes"
    },
    "Osteoporosis": {
        "medications": ["Calcium 500mg", "Alendronate 70mg"],
        "consult": "Yes"
    },
    "Epilepsy": {
        "medications": ["Phenytoin 100mg", "Valproate 500mg"],
        "consult": "Yes"
    },
    "Psoriasis": {
        "medications": ["Topical corticosteroids", "Methotrexate 15mg"],
        "consult": "Yes"
    },
    "Shingles": {
        "medications": ["Acyclovir 800mg", "Pain relief medication (Acetaminophen)"],
        "consult": "Yes"
    },
    "Stroke": {
        "medications": ["Aspirin 81mg", "Clopidogrel 75mg"],
        "consult": "Yes"
    },
    "Gout": {
        "medications": ["Allopurinol 100mg", "Colchicine 0.6mg"],
        "consult": "Yes"
    },
    "Anemia": {
        "medications": ["Ferrous sulfate 325mg", "Folic acid 1mg"],
        "consult": "Yes"
    },
    "Multiple Sclerosis": {
        "medications": ["Beta interferon", "Dimethyl fumarate 240mg"],
        "consult": "Yes"
    },
    "HIV/AIDS": {
        "medications": ["Truvada 200mg/300mg", "Efavirenz 600mg"],
        "consult": "Yes"
    },
    "COVID-19": {
        "medications": ["Remdesivir 200mg", "Dexamethasone 6mg"],
        "consult": "Yes"
    },
    "Acid Reflux (GERD)": {
        "medications": ["Omeprazole 20mg", "Ranitidine 150mg"],
        "consult": "No"
    },
    "Celiac Disease": {
        "medications": ["Gluten-free diet", "Lactase enzyme supplements"],
        "consult": "Yes"
    },
    "Gallstones": {
        "medications": ["Ursodeoxycholic acid 300mg", "Pain relief (Ibuprofen)"],
        "consult": "Yes"
    },
    "Lung Cancer": {
        "medications": ["Chemotherapy", "Radiation therapy", "Pain management"],
        "consult": "Yes"
    },
    "Heart Attack": {
        "medications": ["Aspirin 81mg", "Clopidogrel 75mg", "Nitroglycerin"],
        "consult": "Yes"
    },
    "Parkinson's Disease": {
        "medications": ["Levodopa 100mg", "Carbidopa 25mg"],
        "consult": "Yes"
    },
    "Alzheimer's Disease": {
        "medications": ["Donepezil 10mg", "Rivastigmine 6mg"],
        "consult": "Yes"
    },
    "Rheumatoid Arthritis": {
        "medications": ["Methotrexate 10mg", "Hydroxychloroquine 200mg"],
        "consult": "Yes"
    },
    "Ulcerative Colitis": {
        "medications": ["Mesalamine 500mg", "Prednisone 10mg"],
        "consult": "Yes"
    }
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
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))  # default to 8000 if not set
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
