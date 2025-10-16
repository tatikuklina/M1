
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Literal
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Heart Attack Prediction API")

class PatientData(BaseModel):
    systolic_blood_pressure: float
    blood_sugar: float
    age: float

class PredictionResult(BaseModel):
    prediction: int
    risk_level: str
    message: str

# Загрузка pipeline
try:
    full_pipeline = joblib.load('my_trained_pipeline_1610.pkl')
    print("Pipeline успешно загружен")
    print(f"Тип pipeline: {type(full_pipeline)}")
    print(f"Шаги pipeline: {list(full_pipeline.named_steps.keys())}")
except Exception as e:
    print(f"Ошибка загрузки pipeline: {e}")
    full_pipeline = None

html_interface = """<!DOCTYPE html>
<html>
<head>
    <title>Предсказание риска сердечного приступа</title>
    <style>
        body { font-family: Arial; max-width: 500px; margin: 50px auto; padding: 20px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { width: 100%; padding: 8px; margin: 5px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
        .high-risk { background: #ffcccc; border: 1px solid #ff0000; }
        .low-risk { background: #ccffcc; border: 1px solid #00ff00; }
        .error { background: #ffe6cc; border: 1px solid #ff9900; }
    </style>
</head>
<body>
    <h2>Предсказание риска сердечного приступа</h2>
    <form onsubmit="predictRisk(event)">
        <div class="form-group">
            <label>Систолическое артериальное давление:</label>
            <input type="number" id="bp" placeholder="например, 0.46" required step="0.01">
        </div>
        <div class="form-group">
            <label>Содержание сахара в крови:</label>
            <input type="number" id="sugar" placeholder="например, 0.10" required step="0.01">
        </div>
        <div class="form-group">
            <label>Возраст:</label>
            <input type="number" id="age" placeholder="например, 0.46" required step="0.01">
        </div>
        <button type="submit">Проверка риска</button>
    </form>
    <div id="result"></div>

    <script>
        async function predictRisk(event) {
            event.preventDefault();

            const data = {
                systolic_blood_pressure: parseFloat(document.getElementById('bp').value),
                blood_sugar: parseFloat(document.getElementById('sugar').value),
                age: parseInt(document.getElementById('age').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                displayResult(result);
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    '<div class="result error">Error: ' + error + '</div>';
            }
        }

        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            if (data.prediction === -1) {
                resultDiv.innerHTML = '<div class="result error">' +
                    '<h3>Error</h3><p>' + data.message + '</p></div>';
            } else if (data.prediction === 1) {
                resultDiv.innerHTML = '<div class="result high-risk">' +
                    '<h3>Высокий риск</h3><p>' + data.message + '</p></div>';
            } else {
                resultDiv.innerHTML = '<div class="result low-risk">' +
                    '<h3>Низкий риск</h3><p>' + data.message + '</p></div>';
            }
        }
    </script>
</body>
</html>"""



@app.get("/", response_class=HTMLResponse)
async def main_page():
    return html_interface

@app.get("/model-info")
async def model_info():
    """Информация о загруженном pipeline"""
    if full_pipeline is not None:
        return {
            "pipeline_type": type(full_pipeline).__name__,
            "pipeline_steps": list(full_pipeline.named_steps.keys()),
            "features": ['systolic_blood_pressure', 'blood_sugar', 'age']
        }
    else:
        return {"error": "Pipeline не загружен"}

@app.post("/predict", response_model=PredictionResult)
async def predict_heart_attack(patient_data: PatientData):
    """
    Предсказание риска сердечного приступа с использованием вашего pipeline
    """
    if full_pipeline is None:
        return PredictionResult(
            prediction=-1,
            risk_level="Ошибка",
            message="Pipeline не загружен"
        )

    # Создаю DataFrame с теми же признаками, что и при обучении
    input_data = pd.DataFrame({
        'systolic_blood_pressure': [patient_data.systolic_blood_pressure],
        'blood_sugar': [patient_data.blood_sugar],
        'age': [patient_data.age]
    })

    try:
        # Использую ВЕСЬ pipeline для предсказания
        prediction = full_pipeline.predict(input_data) #!!!!!!prediction = full_pipeline.predict(input_data)[0]

        risk_level = "Высокий риск" if prediction == 1 else "Низкий риск"
        message = "Рекомендуется провести детальное исследование здоровья" if prediction == 1 else "Риск сердечного приступа низкий"

        return PredictionResult(
            prediction=int(prediction),
            risk_level=risk_level,
            message=message
        )

    except Exception as e:
        return PredictionResult(
            prediction=-1,
            risk_level="Ошибка",
            message=f"Ошибка предсказания: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
