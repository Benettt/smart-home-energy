"""
Run this ONCE to train and save the model:
    python train_model.py
This generates: model/energy_model.pkl
"""

import numpy as np
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Appliance config (watts) ──────────────────────────────────────────────────
APPLIANCES = {
    "AC / Heater":        2000,
    "Water Heater":       1500,
    "Washing Machine":    800,
    "Refrigerator":       150,
    "Dishwasher":         600,
    "Microwave":          1200,
    "Lights":             200,
    "TV / Entertainment": 180,
    "Computer / Laptop":  120,
    "Electric Oven":      2200,
}

APPLIANCE_NAMES  = list(APPLIANCES.keys())
APPLIANCE_WATTS  = list(APPLIANCES.values())
NUM_FEATURES     = len(APPLIANCES)

CLASS_NAMES      = ["🟢 Low", "🔵 Normal", "🟠 High", "🔴 Peak"]
CLASS_COLORS     = ["#0e9f6e", "#1a56db", "#ff8800", "#e02424"]
COST_PER_KWH     = 0.12


# ── Generate synthetic training data ─────────────────────────────────────────
def generate_training_data(n_samples: int = 5000):
    np.random.seed(42)
    X, y = [], []

    for _ in range(n_samples):
        hours = np.array([
            np.random.uniform(0,  12),
            np.random.uniform(0,   4),
            np.random.uniform(0,   3),
            np.random.uniform(16, 24),
            np.random.uniform(0,   2),
            np.random.uniform(0,   2),
            np.random.uniform(0,  10),
            np.random.uniform(0,   8),
            np.random.uniform(0,   8),
            np.random.uniform(0,   2),
        ])
        hours = np.clip(hours + np.random.normal(0, 0.3, NUM_FEATURES), 0, 24)
        kwh   = sum(hours[i] * APPLIANCE_WATTS[i] / 1000 for i in range(NUM_FEATURES))

        if   kwh < 8:  label = 0
        elif kwh < 16: label = 1
        elif kwh < 28: label = 2
        else:          label = 3

        X.append(hours)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ── Train & Save ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SmartEnergy MLP — Training")
    print("=" * 50)

    print("\n📊 Generating training data...")
    X, y = generate_training_data(5000)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")

    print("\n🧠 Training MLP Neural Network...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            verbose=True,
        ))
    ])

    pipeline.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipeline.predict(X_train)) * 100
    val_acc   = accuracy_score(y_val,   pipeline.predict(X_val))   * 100

    print(f"\n✅ Train Accuracy: {train_acc:.1f}%")
    print(f"✅ Val   Accuracy: {val_acc:.1f}%")

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/energy_model.pkl")
    print("\n💾 Model saved to model/energy_model.pkl")
    print("🚀 Now run:  streamlit run app.py")