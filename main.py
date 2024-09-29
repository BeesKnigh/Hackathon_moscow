import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import BytesIO, StringIO
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def assign_random_age(row):
    if row['age_class'] == 0:
        return np.random.randint(9, 21)
    elif row['age_class'] == 1:
        return np.random.randint(20, 31)
    elif row['age_class'] == 2:
        return np.random.randint(30, 41)
    elif row['age_class'] == 3:
        return np.random.randint(40, 61)


try:
    with open("catboost_model_new.pkl", "rb") as f:
        model = pickle.load(f)
    with open("catboost_model_age_new.pkl", "rb") as f:
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
        encoding_data = ['ua_client_type', 'ua_device_type']
        encoding_ohe = OneHotEncoder(handle_unknown='ignore')
        ohe_data = encoding_ohe.fit_transform(all_train_data[encoding_data])
        ohe_df = pd.DataFrame(ohe_data.toarray(), columns=encoding_ohe.get_feature_names_out(encoding_data))

        encoding_data = ['event_timestamp', 'region', 'ua_os', 'ua_client_name', 'rutube_video_id']

        le = LabelEncoder()
        encoded_data = {}

        for col in encoding_data:
            encoded_data[col] = le.fit_transform(all_train_data[col])

        encoded_df = pd.DataFrame(encoded_data)

        encoded_df.columns = [f"{col}_encoded" for col in encoded_df.columns]

        merged_data = pd.concat([encoded_df.reset_index(drop=True), merged_data_ohe.reset_index(drop=True)], axis=1)

        merged_data.drop(columns=[col for col in encoding_data if col in merged_data.columns], inplace=True)

        features = df.values
        age_prediction = model_age.predict(features)

        predictions = model.predict(features)
        # Добавление предсказаний в DataFrame
        aboba = pd.DataFrame({
            'viewer_uid': df['viewer_uid'].astype(int),
            'age': age_prediction.astype(int).apply(assign_random_age),
            'sex': predictions.astype(int),
            'age_class': age_prediction.astype(int),
        })
        merged_val = aboba.groupby('viewer_uid').agg({'sex': lambda x: x.mode()[0]}).reset_index()
        merged_val_age = merged_val.groupby('viewer_uid').agg({'age_class': lambda x: x.mode()[0]}).reset_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

    response_stream = StringIO()
    merged_val_age.to_csv(response_stream, index=False)
    response_stream.seek(0)

    return StreamingResponse(
        response_stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
