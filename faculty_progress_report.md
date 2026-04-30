# Project Progress Report: Student Performance Predictor (XAI Dashboard)

**To:** Faculty Advisor
**From:** Project Team
**Date:** April 7, 2026
**Subject:** Status Update – 80% Project Completion Milestone

---

## 1. Executive Summary
We are pleased to report that the development of the **Student Performance Predictor** has reached the **80% completion mark**. We have successfully established the end-to-end pipeline, moving from data preprocessing and machine learning model training all the way to a fully functional interactive web dashboard powered by Explainable AI (XAI).

## 2. Completed Milestones (80%)

### ✅ Data Pipeline & Preprocessing
*   Integrated the **UCI Student Performance Data Set**.
*   Built robust data preprocessing pipelines using `Scikit-Learn` (including One-Hot Encoding for categorical variables and Standard Scaling for numerical parameters).
*   Successfully structured the pipeline to process 32 unique societal, demographic, and academic inputs dynamically.

### ✅ Deep Learning Core (Neural Network)
*   Implemented our primary predictive model using a feed-forward Deep Learning **Neural Network** (via TensorFlow/Keras).
*   Configured a secondary Scikit-Learn `MLPRegressor` fallback system to ensure environmental resilience.
*   Achieved functional prediction logic mapping inputs directly to final grade (G3) scoring.

### ✅ Explainable AI (SHAP) Integration
*   Integrated the **SHAP (SHapley Additive exPlanations) KernelExplainer** directly into our inference pipeline.
*   Generated real-time **Force Plots** which break down *why* the neural network outputted a specific prediction (highlighting positive and negative feature impacts).
*   Developed a personalization logic mapper that translates negative SHAP values into actionable text advice.

### ✅ Backend Integration & Frontend Dashboard
*   Developed a complete **FastAPI backend** that hosts our ML application, API endpoints for inferences, and static file serving.
*   Designed a responsive, modern HTML/Vanilla CSS frontend implementing a **Glassmorphism aesthetic**.
*   Connected the UI correctly so any user can tweak 32 parameters visually and execute a realtime prediction and analysis.

---

## 3. Pending Tasks (Remaining 20%)

To bring the project to 100% completion before the final deadline, we will focus on the following:

*   **Model Optimization & Fine-Tuning**: Further analyze feature importance, perform hyper-parameter tuning natively on our training script, and eliminate current edge case instabilities with array dimension formatting when generating SHAP visualizations for standalone batch processing.
*   **Evaluation Metrics Hardening**: Improve the generation of accuracy, precision, and recall metrics to confidently present our models predictive edge.
*   **Edge Case & UI Testing**: Refine mobile responsiveness slightly on the Glassmorphism layout and include loading animations during longer SHAP calculations.
*   **Final Documentation & Deployment**: Finalize the `README.md` file, comment any remaining code blocks, and potentially containerize (Docker) or deploy the dashboard to a cloud host (e.g., Render/Heroku) for live exhibition.

## 4. Conclusion
The core architecture and most complex technical implementations (Deep Learning + Explainable AI integration) are complete and stable. The remaining work focuses exclusively on hardening the application, improving metrics, and formatting for our final presentation. We are well on track to finish successfully.
