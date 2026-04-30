import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap
import os
import pickle

class StudentSuccessEngine:
    def __init__(self, data_path="student_data/student-mat.csv"):
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
        
        self.X_train, X_test, self.y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        return X_processed.shape[1]

    def train(self, epochs=50):
        input_dim = self.prepare_data()
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            try:
                optimizer = tf.keras.optimizers.Lion(learning_rate=0.0001)
            except AttributeError:
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            #neural networks concept    
            self.model = models.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='linear')
            ])
            
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=32, verbose=0)
        except (ImportError, ModuleNotFoundError):
            print("\n[SYSTEM NOTE] TensorFlow unavailable. Using Scikit-Learn MLPRegressor fallback.\n")
            from sklearn.neural_network import MLPRegressor
            
            # Create a simple wrapper to match the expected interface
            class TFMock:
                def __init__(self):
                    self.m = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
                def fit(self, x, y): self.m.fit(x, y)
                def predict(self, x, **kwargs): return self.m.predict(x).reshape(-1, 1)

            self.model = TFMock()
            self.model.fit(self.X_train, self.y_train)
            
        # Initialize SHAP explainer after training
        background = shap.sample(self.X_train, 50)
        self.explainer = shap.KernelExplainer(self.model.predict, background)

    def predict_single(self, input_data_dict):
        # input_data_dict should contain values for all 32 features
        df_input = pd.DataFrame([input_data_dict])
        X_input = self.preprocessor.transform(df_input)
        
        prediction_raw = self.model.predict(X_input, verbose=0)
        prediction = prediction_raw[0][0] if hasattr(prediction_raw, 'shape') and len(prediction_raw.shape) > 1 else prediction_raw[0]
        
        print(f"[DEBUG] Model prediction: {prediction}")
        
        shap_values = self.explainer.shap_values(X_input, nsamples=100)
        print(f"[DEBUG] SHAP values type: {type(shap_values)}")
        
        # KernelExplainer for single output returns (N, M)
        # For multi-output it returns a list of (N, M)
        if isinstance(shap_values, list):
            print(f"[DEBUG] SHAP values is list of length {len(shap_values)}")
            s_vals = shap_values[0][0] # First output, first sample
        else:
            print(f"[DEBUG] SHAP values shape: {shap_values.shape}")
            s_vals = shap_values[0] # First sample
            
        print(f"[DEBUG] Final s_vals shape: {s_vals.shape}")
        
        return float(prediction), s_vals

    def get_advice(self, s_vals):
        advice_map = {
            'studytime': "Boost focal study hours; consider active recall methods.",
            'absences': "Prioritize attendance; missing classes is the primary driver of lower scores.",
            'failures': "Seek tutoring for foundational gaps identified from past failures.",
            'goout': "Moderate social outings; focus on time allocation for revision.",
            'freetime': "Optimize free time for creative hobbies or light revision.",
            'health': "Prioritize sleep and nutrition to improve morning cognitive focus.",
            'Dalc': "Limit weekday alcohol to maintain academic alertness.",
            'Walc': "Maintain weekend discipline to ensure consistent study cycles.",
            'Medu': "Engagement with family educational backgrounds can provide extra support.",
            'internet': "Leverage internet for research rather than distraction.",
            'schoolsup': "Utilize extra school support sessions more effectively."
        }
        
        # Identify top 3 negative contributors
        # Ensure s_vals is 1D
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
                    "impact": float(s_vals_1d[i])
                })
            else:
                top_neg.append({
                    "feature": feat,
                    "advice": "General area for focus and improvement.",
                    "impact": float(s_vals_1d[i])
                })
        return top_neg

    def save_shap_html(self, s_vals, output_path):
        # Handle expected_value (can be list or scalar)
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 0:
            ev = ev[0]
            
        # Ensure s_vals is 1D
        s_vals_1d = s_vals.flatten() if hasattr(s_vals, 'flatten') else s_vals
        
        print(f"[DEBUG] Force plot: ev={ev}, s_vals_shape={s_vals_1d.shape}")
        
        # Generate interactive force plot HTML
        force_plot = shap.force_plot(
            ev, 
            s_vals_1d, 
            feature_names=self.feature_names,
            out_names="Predicted G3"
        )
        shap.save_html(output_path, force_plot)
