"""
neural_networks/core_engine.py
------------------------------
Core ML engine: data preparation, neural network training,
inference, SHAP explainability, and advice generation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap
import os
import pickle

# Project root (two levels up from this file: neural_networks/ -> root)
ROOT = Path(__file__).resolve().parent.parent


class StudentSuccessEngine:
    def __init__(self, data_path=None):
        # Default: student_data/ lives at the project root
        if data_path is None:
            data_path = str(ROOT / "student_data" / "student-mat.csv")
        self.data_path = data_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.explainer = None
        self.X_train = None
        self.y_train = None

    def prepare_data(self):
        df = pd.read_csv(self.data_path, sep=';')
        X = df.drop(columns=['G3'])
        y = df['G3']

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )

        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = numerical_cols + self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()

        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        return X_processed.shape[1]

    def train(self, epochs=200):
        """
        Train the neural network.
        Primary:  TensorFlow Sequential DNN  (128 → 64 → 32 → 1)
        Fallback: Scikit-Learn MLPRegressor  (same hidden-layer structure)
        """
        input_dim = self.prepare_data()

        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, callbacks

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            # ── Improved Neural Network ───────────────────────────────────────
            self.model = models.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='linear')   # regression output
            ])

            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # ── Train with early stopping to prevent over/under-fitting ───────
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True
            )
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )

            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=16,
                validation_split=0.15,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            val_mae = min(history.history.get('val_mae', [999]))
            print(f"[MODEL] Best validation MAE: {val_mae:.3f}")

        except (ImportError, ModuleNotFoundError):
            print("\n[SYSTEM NOTE] TensorFlow unavailable. Using Scikit-Learn MLPRegressor fallback.\n")
            from sklearn.neural_network import MLPRegressor

            # Wrapper that mirrors the Keras .predict / .fit interface
            class TFMock:
                def __init__(self):
                    self.m = MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        max_iter=500,
                        learning_rate='adaptive',
                        random_state=42
                    )

                def fit(self, x, y):
                    self.m.fit(x, y)

                def predict(self, x, **kwargs):
                    return self.m.predict(x).reshape(-1, 1)

            self.model = TFMock()
            self.model.fit(self.X_train, self.y_train)

        # ── SHAP KernelExplainer (model-agnostic) ────────────────────────────
        background = shap.sample(self.X_train, 50)
        self.explainer = shap.KernelExplainer(self.model.predict, background)

    def predict_single(self, input_data_dict):
        """Run inference + compute SHAP values for one student."""
        df_input = pd.DataFrame([input_data_dict])
        X_input = self.preprocessor.transform(df_input)

        prediction_raw = self.model.predict(X_input, verbose=0)
        prediction = (
            prediction_raw[0][0]
            if hasattr(prediction_raw, 'shape') and len(prediction_raw.shape) > 1
            else prediction_raw[0]
        )

        print(f"[DEBUG] Model prediction: {prediction}")

        shap_values = self.explainer.shap_values(X_input, nsamples=100)
        print(f"[DEBUG] SHAP values type: {type(shap_values)}")

        if isinstance(shap_values, list):
            print(f"[DEBUG] SHAP values is list of length {len(shap_values)}")
            s_vals = shap_values[0][0]
        else:
            print(f"[DEBUG] SHAP values shape: {shap_values.shape}")
            s_vals = shap_values[0]

        print(f"[DEBUG] Final s_vals shape: {s_vals.shape}")
        return float(prediction), s_vals

    def get_advice(self, s_vals):
        """Map top-3 negative SHAP contributors to actionable advice."""
        advice_map = {
            'studytime': "Boost focal study hours; consider active recall methods.",
            'absences':  "Prioritize attendance; missing classes is the primary driver of lower scores.",
            'failures':  "Seek tutoring for foundational gaps identified from past failures.",
            'goout':     "Moderate social outings; focus on time allocation for revision.",
            'freetime':  "Optimize free time for creative hobbies or light revision.",
            'health':    "Prioritize sleep and nutrition to improve morning cognitive focus.",
            'Dalc':      "Limit weekday alcohol to maintain academic alertness.",
            'Walc':      "Maintain weekend discipline to ensure consistent study cycles.",
            'Medu':      "Engagement with family educational backgrounds can provide extra support.",
            'internet':  "Leverage internet for research rather than distraction.",
            'schoolsup': "Utilize extra school support sessions more effectively.",
        }

        s_vals_1d = s_vals.flatten() if hasattr(s_vals, 'flatten') else s_vals
        indices = np.argsort(s_vals_1d)[:3]
        top_neg = []
        for i in indices:
            feat = self.feature_names[i]
            base_feat = next((k for k in advice_map.keys() if k in feat), None)
            if base_feat:
                top_neg.append({
                    "feature": feat,
                    "advice": advice_map[base_feat],
                    "impact": float(s_vals_1d[i]),
                })
            else:
                top_neg.append({
                    "feature": feat,
                    "advice": "General area for focus and improvement.",
                    "impact": float(s_vals_1d[i]),
                })
        return top_neg

    def save_shap_html(self, s_vals, output_path):
        """Generate and save an interactive SHAP force plot as HTML."""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 0:
            ev = ev[0]

        s_vals_1d = s_vals.flatten() if hasattr(s_vals, 'flatten') else s_vals

        print(f"[DEBUG] Force plot: ev={ev}, s_vals_shape={s_vals_1d.shape}")

        force_plot = shap.force_plot(
            ev,
            s_vals_1d,
            feature_names=self.feature_names,
            out_names="Predicted G3"
        )
        shap.save_html(output_path, force_plot)
