from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import uuid
import numpy as np
from core_engine import StudentSuccessEngine

app = FastAPI()

# Global engine instance
engine = StudentSuccessEngine()
PLOT_DIR = "static/plots"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

class StudentData(BaseModel):
    school: str = "GP"
    sex: str = "F"
    age: int = 18
    address: str = "U"
    famsize: str = "GT3"
    Pstatus: str = "A"
    Medu: int = 4
    Fedu: int = 4
    Mjob: str = "at_home"
    Fjob: str = "teacher"
    reason: str = "course"
    guardian: str = "mother"
    traveltime: int = 2
    studytime: int = 2
    failures: int = 0
    schoolsup: str = "yes"
    famsup: str = "no"
    paid: str = "no"
    activities: str = "no"
    nursery: str = "yes"
    higher: str = "yes"
    internet: str = "no"
    romantic: str = "no"
    famrel: int = 4
    freetime: int = 3
    goout: int = 4
    Dalc: int = 1
    Walc: int = 1
    health: int = 3
    absences: int = 6
    G1: int = 5
    G2: int = 6

@app.on_event("startup")
async def startup_event():
    print("Initializing Student Success Engine...")
    engine.train(epochs=30)
    print("Engine Ready!")

@app.post("/api/predict")
async def predict_student(data: StudentData):
    try:
        print(f"Received prediction request for age {data.age}, G1 {data.G1}, G2 {data.G2}")
        input_dict = data.model_dump()
        prediction, s_vals = engine.predict_single(input_dict)
        print(f"Prediction successful: {prediction}")
        
        # Generate a unique ID for this plot
        plot_id = str(uuid.uuid4())
        plot_path = os.path.join(PLOT_DIR, f"{plot_id}.html")
        engine.save_shap_html(s_vals, plot_path)
        print(f"SHAP plot saved to {plot_path}")
        
        # Get personalized advice
        advice = engine.get_advice(s_vals)
        
        return {
            "prediction": float(prediction),
            "status": "At-Risk" if prediction < 10 else "Succeeding",
            "plot_id": plot_id,
            "advice": advice
        }
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shap-plot/{plot_id}")
async def get_shap_plot(plot_id: str):
    plot_path = os.path.join(PLOT_DIR, f"{plot_id}.html")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    
    with open(plot_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    return HTMLResponse(content=html_content)

# Mount static files (must be after specific API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
