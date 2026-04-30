"""
backend/app.py
--------------
FastAPI server for the Student Success Predictor.
Imports the ML engine from the neural_networks package.
"""
import sys
import os
from pathlib import Path

# ── Add project root to sys.path so 'neural_networks' package is importable ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uuid
import numpy as np

from neural_networks.core_engine import StudentSuccessEngine

app = FastAPI()


# ── Disable browser caching for static files during development ─────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        if request.url.path.endswith(('.html', '.css', '.js')):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
        return response


app.add_middleware(NoCacheMiddleware)


# Global engine instance
engine = StudentSuccessEngine()

# SHAP plot HTML files are saved inside frontend/plots/
PLOT_DIR = ROOT / "frontend" / "plots"
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
    engine.train(epochs=200)
    print("Engine Ready!")


@app.post("/api/predict")
async def predict_student(data: StudentData):
    try:
        print(f"Received prediction request for age {data.age}, G1 {data.G1}, G2 {data.G2}")
        input_dict = data.model_dump()
        prediction, s_vals = engine.predict_single(input_dict)
        print(f"Prediction successful: {prediction}")

        # Generate a unique ID for this SHAP plot
        plot_id = str(uuid.uuid4())
        plot_path = PLOT_DIR / f"{plot_id}.html"
        engine.save_shap_html(s_vals, str(plot_path))
        print(f"SHAP plot saved to {plot_path}")

        # Get personalized advice
        advice = engine.get_advice(s_vals)

        return {
            "prediction": float(prediction),
            "status": "At-Risk" if prediction < 10 else "Succeeding",
            "plot_id": plot_id,
            "advice": advice,
        }
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/shap-plot/{plot_id}")
async def get_shap_plot(plot_id: str):
    plot_path = PLOT_DIR / f"{plot_id}.html"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")

    with open(plot_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


# Serve the frontend/ directory as root (must be after API routes)
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
