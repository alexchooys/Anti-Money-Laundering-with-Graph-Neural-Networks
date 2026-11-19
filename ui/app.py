"""FastAPI interface: verified experiment replay and genuine live tree inference."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from src.service import replay, live

app = FastAPI(title='AML Temporal Robustness Lab')


class LiveInput(BaseModel):
    features: list[float]


@app.get('/', response_class=HTMLResponse)
def home():
    return Path(__file__).with_name('index.html').read_text()


@app.get('/api/replay')
def replay_api(model: str='XGBoost', policy: str='recency', step: int=49, budget: int=10):
    try: return replay(model, policy, step, budget)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content={'error': str(exc)})


@app.post('/api/predict')
def predict_api(body: LiveInput):
    try: return live(body.features)
    except (ValueError, TypeError) as exc:
        return JSONResponse(status_code=400, content={'error': str(exc)})
