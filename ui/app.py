"""Deliverable 3 web interface with three-seed replay, export and validated batch inference."""
from pathlib import Path
from fastapi import FastAPI,Query
from fastapi.responses import HTMLResponse,JSONResponse,Response
from pydantic import BaseModel,Field
from src import service
app=FastAPI(title='AML Temporal Robustness Lab v2')
class LiveInput(BaseModel):features:list
class CSVInput(BaseModel):csv_text:str=Field(max_length=2_000_000)
def attempt(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except (ValueError,TypeError,FileNotFoundError) as e:return JSONResponse(status_code=400,content={'error':str(e)})
@app.get('/',response_class=HTMLResponse)
def home():return Path(__file__).with_name('index.html').read_text()
@app.get('/api/comparison')
def comparison():return service.comparison()
@app.get('/api/example')
def example():return service.example()
@app.get('/api/template')
def template():return Response(service.template_csv(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=real_example_features.csv'})
@app.get('/api/replay')
def replay(model:str='XGBoost',policy:str='recency',seed:int=0,step:int=49,budget:int=10):return attempt(service.replay,model,policy,step,budget,seed)
@app.get('/api/export')
def export(model:str='XGBoost',policy:str='recency',seed:int=0,step:int=49,budget:int=10):
    try:body=service.export_queue(model,policy,seed,step,budget)
    except (ValueError,TypeError) as e:return JSONResponse(status_code=400,content={'error':str(e)})
    return Response(body,media_type='text/csv',headers={'Content-Disposition':'attachment; filename=review_queue.csv'})
@app.post('/api/predict')
def predict(body:LiveInput):return attempt(service.live,body.features)
@app.post('/api/batch')
def batch(body:CSVInput):return attempt(service.batch_csv,body.csv_text)
