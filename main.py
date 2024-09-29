from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import BytesIO, StringIO
import pickle

app = FastAPI()

templates = Jinja2Templates(directory="templates")

try:
    with open("catboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("catboost_model_age.pkl", "rb") as f:
        model_age = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-csv/", response_class=StreamingResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV.")
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении CSV файла: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV файл пустой или неправильно отформатирован.")

    try:
        features = df.values
        age_prediction = model_age.predict(features)

        predictions = model.predict(features)
        # Добавление предсказаний в DataFrame
        aboba = pd.DataFrame({
            'viewer_uid': df['viewer_uid'].astype(int),
            'sex': predictions.astype(int),
            'age': age_prediction.astype(int),
        })
        # merged_val = aboba.groupby('viewer_uid').agg({'sex': lambda x: x.mode()[0]}).reset_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

    response_stream = StringIO()
    aboba.to_csv(response_stream, index=False)
    response_stream.seek(0)

    return StreamingResponse(
        response_stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
