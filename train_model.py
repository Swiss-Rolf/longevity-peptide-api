# train_model.py
import pandas as pd
import peptides
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def featurize(sequence):
    """
    Converts a peptide sequence into a dictionary of features.
    This is the "vibe" that turns 'GHK' into numbers.
    """
    try:
        p = peptides.Peptide(sequence)
        # These features MUST match the ones in your Replit 'main.py'
        return {
            'charge': p.charge(pH=7.4),
            'hydrophobicity': p.hydrophobicity(scale="KyteDoolittle"),
            'aliphatic_index': p.aliphatic_index(),
            'instability_index': p.instability_index(),
            'molecular_weight': p.molecular_weight()
        }
    except Exception as e:
        print(f"Error processing sequence {sequence}: {e}")
        return {'charge': 0, 'hydrophobicity': 0, 'aliphatic_index': 0, 'instability_index': 0, 'molecular_weight': 0}

print("Starting model training...")

# 1. Load data
try:
    df = pd.read_csv("peptide_data.csv")
except FileNotFoundError:
    print("ERROR: 'peptide_data.csv' not found.")
    print("Make sure it's in the same folder as this script.")
    exit()

# 2. Featurize
features_list = df['sequence'].apply(featurize)
features_df = pd.json_normalize(features_list) # Converts list of dicts to DataFrame

# 3. Create X (features) and y (target)
X = features_df
y = df['is_longevity']

print(f"Loaded and featurized {len(df)} peptides.")

# 4. Split data for training and testing
# Note: For a small dataset, we'll train on ALL data for the final model
# But we'll still print a report based on a test split to see how it does.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Create a processing & training "Pipeline"
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Train the model on the TEST split (for reporting)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("\n--- Model Performance on Test Set ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Train the model on ALL data (for the final saved model)
print("\n--- Retraining on ALL data for final model ---")
final_model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
final_model_pipeline.fit(X, y)
print("Final model training complete.")

# 8. Save the final model
joblib.dump(final_model_pipeline, "peptide_model.pkl")
print("\n✅ Success! Model trained and saved as 'peptide_model.pkl'")