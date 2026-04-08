from app.ml.ensemble_model import EnsembleIDS

# load models
ids = EnsembleIDS("models")

# example feature vector (must match 23 features)
features = [
    1000, 10, 5000, 500,
    20, 10000, 50, 10, 5, 100,
    1,1,0,0,0,0,
    12345,80,
    1,0,0,
    0.002,400
]

label, conf = ids.predict("192.168.1.10", features)

print("Prediction:", label)
print("Confidence:", conf)