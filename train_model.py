"""
Run this ONCE to train and save the CNN model:
    python train_model.py
This generates: model/energy_cnn.keras
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from tensorflow.keras import layers

# ── Appliance config (watts) ──────────────────────────────────────────────────
APPLIANCES = {
    "AC / Heater":       2000,
    "Water Heater":      1500,
    "Washing Machine":   800,
    "Refrigerator":      150,
    "Dishwasher":        600,
    "Microwave":         1200,
    "Lights":            200,
    "TV / Entertainment":180,
    "Computer / Laptop": 120,
    "Electric Oven":     2200,
}

APPLIANCE_NAMES = list(APPLIANCES.keys())
APPLIANCE_WATTS = list(APPLIANCES.values())
NUM_FEATURES    = len(APPLIANCES)   # 10 inputs

CLASS_NAMES     = ["🟢 Low", "🔵 Normal", "🟠 High", "🔴 Peak"]
CLASS_COLORS    = ["#0e9f6e", "#1a56db", "#ff8800", "#e02424"]
COST_PER_KWH    = 0.12

# ── Generate synthetic training data ─────────────────────────────────────────

def generate_training_data(n_samples: int = 5000):
    """
    Each sample = daily usage hours for each appliance (0–24h).
    Label = consumption category based on total daily kWh.
    """
    np.random.seed(42)
    X, y = [], []

    for _ in range(n_samples):
        # Random daily usage hours per appliance
        hours = np.array([
            np.random.uniform(0,  12),   # AC
            np.random.uniform(0,   4),   # Water Heater
            np.random.uniform(0,   3),   # Washing Machine
            np.random.uniform(16, 24),   # Refrigerator (always on)
            np.random.uniform(0,   2),   # Dishwasher
            np.random.uniform(0,   2),   # Microwave
            np.random.uniform(0,  10),   # Lights
            np.random.uniform(0,   8),   # TV
            np.random.uniform(0,   8),   # Computer
            np.random.uniform(0,   2),   # Oven
        ])

        # Add noise
        hours = np.clip(hours + np.random.normal(0, 0.3, NUM_FEATURES), 0, 24)

        # Daily kWh
        kwh = sum(hours[i] * APPLIANCE_WATTS[i] / 1000 for i in range(NUM_FEATURES))

        # Label by quartile thresholds
        if   kwh < 8:   label = 0   # Low
        elif kwh < 16:  label = 1   # Normal
        elif kwh < 28:  label = 2   # High
        else:           label = 3   # Peak

        X.append(hours)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ── Build CNN ─────────────────────────────────────────────────────────────────

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(NUM_FEATURES, 1)),

        layers.Conv1D(64,  kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(4, activation="softmax"),
    ], name="SmartHomeEnergyCNN")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Train & Save ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  SmartEnergy CNN — Training")
    print("=" * 50)

    print("\n📊 Generating training data...")
    X, y = generate_training_data(5000)
    X = X.reshape(-1, NUM_FEATURES, 1)

    split    = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"   Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print(f"   Classes: {dict(zip(CLASS_NAMES, [int((y==i).sum()) for i in range(4)]))}\n")

    model = build_model()
    model.summary()

    print("\n🧠 Training CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=64,
        verbose=1,
    )

    final_acc     = history.history["accuracy"][-1] * 100
    final_val_acc = history.history["val_accuracy"][-1] * 100
    print(f"\n✅ Train Accuracy: {final_acc:.1f}%")
    print(f"✅ Val   Accuracy: {final_val_acc:.1f}%")

    os.makedirs("model", exist_ok=True)
    model.save("model/energy_cnn.keras")
    print("\n💾 Model saved to model/energy_cnn.keras")
    print("🚀 Now run:  streamlit run app.py")