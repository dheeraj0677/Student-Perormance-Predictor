"""
neural_networks/student_performance_system.py
---------------------------------------------
Standalone training + evaluation script.
Run directly to train, evaluate, and generate SHAP insights
without starting the FastAPI server.

Usage:
    cd Student-Perormance-Predictor
    python neural_networks/student_performance_system.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import shap
import matplotlib.pyplot as plt

# Project root resolved from this file's location
ROOT = Path(__file__).resolve().parent.parent


# ── 1. DATA LOADING & PREPROCESSING ─────────────────────────────────────────
def load_and_preprocess_data(filepath):
    """Load UCI student dataset and apply preprocessing pipeline."""
    # UCI dataset uses ';' as separator
    df = pd.read_csv(filepath, sep=';')

    # Features (X) are the first 32 columns; Target (y) is G3
    X = df.drop(columns=['G3'])
    y = df['G3']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing pipeline: Standard Scaling + One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    X_processed  = preprocessor.fit_transform(X)
    feature_names = (
        numerical_cols
        + preprocessor.named_transformers_['cat']
          .get_feature_names_out(categorical_cols).tolist()
    )

    return X_processed, y, feature_names, preprocessor


# ── 2. NEURAL NETWORK MODEL ARCHITECTURE (with Fallback) ────────────────────
def build_model(input_dim):
    """
    Build a 4-layer fully-connected neural network.
    Primary:  TensorFlow/Keras  Sequential DNN
    Fallback: Scikit-Learn      MLPRegressor (same hidden layers)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        # Prefer Lion optimizer (faster convergence); fall back to Adam
        try:
            optimizer = tf.keras.optimizers.Lion(learning_rate=0.0001)
        except AttributeError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # ── Architecture: Input → 64 → 32 → 16 → 1 (regression) ─────────────
        model_tf = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')   # predicts G3 (0–20)
        ])

        model_tf.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model_tf

    except (ImportError, ModuleNotFoundError):
        print("\n[SYSTEM NOTE] TensorFlow not available. Falling back to Scikit-Learn MLPRegressor.\n")
        from sklearn.neural_network import MLPRegressor

        # Scikit-Learn wrapper that mimics Keras .predict / .fit interface
        class SKLearnWrapper:
            def __init__(self, input_dim):
                self.model = MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16),
                    max_iter=200,
                    random_state=42
                )

            def fit(self, X, y, **kwargs):
                self.model.fit(X, y)
                return self

            def predict(self, X, **kwargs):
                # SHAP and Keras expect shape (N, 1)
                return self.model.predict(X).reshape(-1, 1)

        return SKLearnWrapper(input_dim)


# ── 3. EVALUATION (MSE, Precision, Recall) ───────────────────────────────────
def evaluate_student_status(y_true, y_pred, threshold=10):
    """
    Convert regression output to binary 'at-risk' labels and compute metrics.
    At-risk = predicted G3 < threshold (default: 10 = failing grade).
    """
    at_risk_true = (y_true < threshold).astype(int)
    at_risk_pred = (y_pred < threshold).astype(int)

    precision = precision_score(at_risk_true, at_risk_pred)
    recall    = recall_score(at_risk_true, at_risk_pred)
    mse       = mean_squared_error(y_true, y_pred)

    return mse, precision, recall


# ── 4. PERSONALIZED CURRICULUM PLANNER ───────────────────────────────────────
def generate_curriculum_plan(student_shap_values, feature_names):
    """Map the top-3 most negative SHAP features to actionable advice."""
    advice_map = {
        'studytime': "Increase weekly study hours; follow the 'pomodoro' technique for focus.",
        'absences':  "Attend all lectures; high absences are significantly lowering your grade.",
        'failures':  "Enroll in remedial sessions to strengthen foundation in past topics.",
        'goout':     "Balance social time; reduce evening outings to prioritize exam prep.",
        'health':    "Maintain a healthy routine; physical well-being affects cognitive focus.",
        'internet':  "Use online educational resources (e.g., Khan Academy) to supplement learning.",
        'Dalc':      "Limit weekday alcohol consumption to improve morning alertness.",
        'Walc':      "Reduce weekend alcohol intake to ensure better weekend study momentum.",
    }

    indices = np.argsort(student_shap_values)[:3]
    top_negative_features = [feature_names[i] for i in indices]

    plan = []
    for feat in top_negative_features:
        base_feat = next((k for k in advice_map.keys() if k in feat), None)
        if base_feat:
            plan.append(f"- {feat}: {advice_map[base_feat]}")
        else:
            plan.append(f"- {feat}: Requires attention to improve final performance.")

    return plan


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    filepath = str(ROOT / "student_data" / "student-mat.csv")
    X, y, feature_names, preprocessor = load_and_preprocess_data(filepath)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Total features after encoding: {X.shape[1]}")

    # Build and train the neural network
    model = build_model(X.shape[1])
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Predictions
    y_pred = model.predict(X_test).flatten()

    # Evaluation metrics
    mse, prec, rec = evaluate_student_status(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"  MSE:                             {mse:.4f}")
    print(f"  Precision (At-Risk Detection):   {prec:.4f}")
    print(f"  Recall    (At-Risk Detection):   {rec:.4f}")

    # ── XAI: SHAP Explainability ──────────────────────────────────────────────
    print("\nInitializing SHAP Explainer...")
    # Use a small background sample from training data for speed
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 20))

    # Explain one student prediction (first test sample for demo)
    student_idx = 0
    student_shap = explainer.shap_values(X_test[student_idx:student_idx + 1], nsamples=100)

    print(f"\nStudent #{student_idx} Case Study:")
    print(f"  Actual Grade:    {y_test.iloc[student_idx]}")
    print(f"  Predicted Grade: {y_pred[student_idx]:.2f}")

    # Flatten SHAP values (list for multi-output, array for single-output)
    s_vals = student_shap[0] if isinstance(student_shap, list) else student_shap[0]

    curriculum = generate_curriculum_plan(s_vals, feature_names)
    print("\nPersonalized Curriculum Plan (Top 3 Improvement Areas):")
    for item in curriculum:
        print(item)

    print("\nSHAP Force Plot generated (individual interpretation).")
    # To display interactively in a notebook:
    # shap.force_plot(explainer.expected_value, s_vals, X_test[student_idx],
    #                 feature_names=feature_names, matplotlib=True)
