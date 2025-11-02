# main.py
import uvicorn
import joblib
import pandas as pd
import peptides
from fastapi import FastAPI
from pydantic import BaseModel

# --- Helper Function (to match your training script) ---
def featurize(sequence):
    """Converts a peptide sequence into a dictionary of features."""
    try:
        p = peptides.Peptide(sequence)
        # Ensure these are the *exact same* features you used in train_model.py
        return {
            'charge': p.charge(pH=7.4),
            'hydrophobicity': p.hydrophobicity(scale="KyteDoolittle"),
            'aliphatic_index': p.aliphatic_index(),
            'instability_index': p.instability_index(),
            'molecular_weight': p.molecular_weight()
        }
    except Exception as e:
        return {
            'charge': 0, 
            'hydrophobicity': 0, 
            'aliphatic_index': 0, 
            'instability_index': 0, 
            'molecular_weight': 0
        }

# --- Load Your Trained Model ---
# (Render will find 'peptide_model.pkl' in your repo)
try:
    model = joblib.load("peptide_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'peptide_model.pkl' not found. Please upload it.")
    model = None

# --- Initialize the FastAPI App ---
app = FastAPI(
    title="Longevity Peptide Predictor API",
    description="An API to predict the longevity association of a peptide sequence."
)

# --- Define the Input & Output Models (Good Practice) ---
class PeptideInput(BaseModel):
    sequence: str

class PredictionOutput(BaseModel):
    sequence: str
    longevity_probability: float
    error: str = None

# --- Create the Prediction Endpoint ---
@app.post("/predict/", response_model=PredictionOutput)
def predict_peptide(payload: PeptideInput):
    """
    Predicts the longevity-associated probability of a given peptide sequence.
    """
    if model is None:
        return {
            "sequence": payload.sequence, 
            "longevity_probability": 0.0,
            "error": "Model not loaded. Please check server logs."
        }
    
    try:
        sequence = payload.sequence.strip().upper()
        features = featurize(sequence)
        features_df = pd.json_normalize([features])
        prediction_prob = model.predict_proba(features_df)
        longevity_prob = prediction_prob[0][1] # Probability of class '1'

        return {
            "sequence": sequence, 
            "longevity_probability": longevity_prob
        }
        
    except Exception as e:
        return {
            "sequence": payload.sequence, 
            "longevity_probability": 0.0,
            "error": str(e)
        }

# --- (Optional) Root endpoint for testing ---
@app.get("/")
def read_root():
    return {"status": "Longevity Peptide API is running. POST to /predict/"}

# --- This part allows the server to run (though gunicorn uses 'app' directly) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)