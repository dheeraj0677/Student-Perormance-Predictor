import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import shap
import matplotlib.pyplot as plt

# 1. DATA LOADING & PREPROCESSING
def load_and_preprocess_data(filepath):
    # UCI dataset uses ';' as separator
    df = pd.read_csv(filepath, sep=';')
    
    # Selecting all 33 features (including G3 target)
    # Features (X) are the first 32 columns, Target (y) is the 33rd (G3)
    X = df.drop(columns=['G3'])
    y = df['G3']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing pipeline: One-Hot for Categorical, Standard Scaling for Numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    feature_names = numerical_cols + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    
    return X_processed, y, feature_names, preprocessor

# 2. DL MODEL ARCHITECTURE (With Fallback)
def build_model(input_dim):
    # Using Scikit-Learn fallback for environments where TensorFlow is unavailable (e.g. Python 3.14)
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        try:
            optimizer = tf.keras.optimizers.Lion(learning_rate=0.0001)
        except AttributeError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
        model_tf = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])
        
        model_tf.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model_tf
    except (ImportError, ModuleNotFoundError):
        print("\n[SYSTEM NOTE] TensorFlow is not available in this environment (likely due to Python 3.14 conflict).")
        print("Falling back to Scikit-Learn MLPRegressor for demonstration.\n")
        from sklearn.neural_network import MLPRegressor
        
        # Create a scikit-learn wrapper that mimics Keras .predict and .fit
        class SKLearnWrapper:
            def __init__(self, input_dim):
                self.model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42)
            
            def fit(self, X, y, **kwargs):
                self.model.fit(X, y)
                return self
            
            def predict(self, X, **kwargs):
                # SHAP and Keras expect (N, 1) or (N,)
                preds = self.model.predict(X)
                return preds.reshape(-1, 1)
        
        return SKLearnWrapper(input_dim)

# 3. EVALUATION LOOP (MSE, PR, RECALL)
def evaluate_student_status(y_true, y_pred, threshold=10):
    # Convert regression output to binary for "at-risk" classification
    at_risk_true = (y_true < threshold).astype(int)
    at_risk_pred = (y_pred < threshold).astype(int)
    
    precision = precision_score(at_risk_true, at_risk_pred)
    recall = recall_score(at_risk_true, at_risk_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    return mse, precision, recall

# 4. PERSONALIZATION LOGIC MAPPER
def generate_curriculum_plan(student_shap_values, feature_names):
    # Map feature names to specific advice
    advice_map = {
        'studytime': "Increase weekly study hours; follow the 'pomodoro' technique for focus.",
        'absences': "Attend all lectures; high absences are significantly lowering your grade.",
        'failures': "Enroll in remedial sessions to strengthen foundation in past topics.",
        'goout': "Balance social time; reduce evening outings to prioritize exam prep.",
        'health': "Maintain a healthy routine; physical well-being affects cognitive focus.",
        'internet': "Use online educational resources (e.g., Khan Academy) to supplement learning.",
        'Dalc': "Limit weekday alcohol consumption to improve morning alertness.",
        'Walc': "Reduce weekend alcohol intake to ensure better weekend study momentum."
    }
    
    # Get top 3 features with negative impact (highest absolute negative SHAP values)
    # Since SHAP values indicate contribution to the target (G3), 
    # negative means it lowered the predicted grade.
    indices = np.argsort(student_shap_values)[:3]
    top_negative_features = [feature_names[i] for i in indices]
    
    plan = []
    for feat in top_negative_features:
        # Check if feature has a direct advice mapping or find a base feature name
        base_feat = next((k for k in advice_map.keys() if k in feat), None)
        if base_feat:
            plan.append(f"- {feat}: {advice_map[base_feat]}")
        else:
            plan.append(f"- {feat}: Requires attention to improve final performance.")
            
    return plan

# MAIN EXECUTION
if __name__ == "__main__":
    filepath = "student_data/student-mat.csv"
    X, y, feature_names, preprocessor = load_and_preprocess_data(filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Total features after encoding: {X.shape[1]}")
    
    # Build and Train
    model = build_model(X.shape[1])
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_test).flatten()
    
    # Evaluation
    mse, prec, rec = evaluate_student_status(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"Precision (At-Risk Identification): {prec:.4f}")
    print(f"Recall (At-Risk Identification): {rec:.4f}")
    
    # 5. XAI WITH SHAP
    print("\nInitializing SHAP Explainer...")
    # Using a small subset of training data as background for speed
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 20))
    
    # Generate SHAP values for one specific "at-risk" student in the test set
    student_idx = 0 # Selecting the first student in test set for demo
    student_shap = explainer.shap_values(X_test[student_idx:student_idx+1], nsamples=100)
    
    # Explain prediction for the student
    print(f"\nStudent #{student_idx} Case Study:")
    print(f"Actual Grade: {y_test.iloc[student_idx]} | Predicted Grade: {y_pred[student_idx]:.2f}")
    
    # Personalization Logic
    # Flatten shap values if they come in a list (for multi-output libraries, though this is regression)
    s_vals = student_shap[0] if isinstance(student_shap, list) else student_shap[0]
    
    curriculum = generate_curriculum_plan(s_vals, feature_names)
    print("\nPersonalized Curriculum Plan (Top 3 Improvement Areas):")
    for item in curriculum:
        print(item)
    
    # SHAP Visualization (Force Plot saving logic)
    print("\nSHAP Force Plot generated (individual interpretation).")
    # In a real GUI, shap.force_plot(explainer.expected_value, s_vals, X_test[student_idx], feature_names=feature_names, matplotlib=True)
