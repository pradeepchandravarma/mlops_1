import pandas as pd

def encode_extra_activities(extra_activities):
    mapping = {'Yes':1,'No':0}
    if extra_activities not in mapping:
        raise ValueError('Extracurricular Activities must be Yes or No')
    return mapping[extra_activities]


    
FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]

def preprocess_features(features):
    X = pd.DataFrame([features])
    
    #raise error if columns missing in payload
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
    raise ValueError(f"Missing columns: {missing}")
    
    #ensures correct order
    return X[FEATURES]





    
