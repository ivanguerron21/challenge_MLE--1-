import fastapi
from fastapi import Request, HTTPException
from challenge.model import DelayModel
import os
import pandas as pd

app = fastapi.FastAPI()
model = DelayModel()
saved_model = 'saved_model' 
if not os.path.exists(saved_model):
    os.mkdir(saved_model)
if len(os.listdir(saved_model)) == 0:
    data = pd.read_csv(filepath_or_buffer="data/data.csv", low_memory=False)
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    try:
        data = await request.json()
        data = data['flights']
        data = pd.DataFrame(data)
        features = model.preprocess(data=data)
        preds = model.predict(features)
        pred = {'predict': preds}
        return pred
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))