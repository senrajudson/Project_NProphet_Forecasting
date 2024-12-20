from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import logging
import os

app = FastAPI()

# Variáveis globais para armazenar a última previsão
last_forecast = {}

class ForecastPost(BaseModel):
    symbol: str
    predict_size: int
    train_size: int
    start_time: str = None  # Novo campo opcional para data e hora de início

class ForecastGet(BaseModel):
    flag: int
    predicted_value: float
    last_prediction_time: str

def get_data(symbol, train_size, start_time=None):
    if start_time:
        start_time_dt = pd.to_datetime(start_time)
        start_timestamp = int(start_time_dt.timestamp())
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, start_timestamp, train_size)
    else:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, train_size)

    if rates is None:
        raise HTTPException(status_code=500, detail=f"Erro ao obter os dados: {mt5.last_error()}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['dayOfWeek'] = df['time'].dt.day_name()
    return df

def round_to_half(value):
    return round(value * 2) / 2

def forecast_neuralprophet_rolling_with_open_price(df, predict_size=30):
    from neuralprophet import NeuralProphet
    model = NeuralProphet()
    df = df.rename(columns={'time': 'ds', 'close': 'y'})
    df = df[['ds', 'y', 'open', "tick_volume", "real_volume"]]
    df['ds'] = pd.to_datetime(df['ds'])

    last_date = df['ds'].iloc[-1]
    dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=predict_size, freq='T')
    future_dates = pd.DataFrame({'ds': dates})
    future_dates['open'] = df['open'].iloc[-1]
    future_dates['tick_volume'] = df['tick_volume'].iloc[-1]
    future_dates['real_volume'] = df['real_volume'].iloc[-1]
    future_dates['y'] = None

    model.add_future_regressor('open')
    model.add_future_regressor('tick_volume')
    model.add_future_regressor('real_volume')
    model.fit(df)

    forecast = model.predict(future_dates)
    forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)
    future_dates['yhat1'] = forecast['yhat1'].values

    last_diff = df['y'].iloc[-1] - future_dates['yhat1'].iloc[-1]
    flag = 1 if last_diff < 0 else 0

    result = {
        "flag": flag,
        "predicted_value": future_dates['yhat1'].iloc[-1],
        "last_prediction_time": str(future_dates['ds'].iloc[-1]),
    }
    return result, future_dates

# Rotas da API
@app.on_event("startup")
def startup_event():
    load_dotenv()
    LOGIN = os.getenv('LOGIN')
    PASSWORD = os.getenv('PASSWORD')
    SERVER = os.getenv('SERVER')

    if not mt5.initialize():
        raise HTTPException(status_code=500, detail="Erro ao inicializar MetaTrader5")
    
    if not mt5.login(int(LOGIN), PASSWORD, SERVER):
        raise HTTPException(status_code=500, detail=f"Erro ao fazer login: {mt5.last_error()}")

@app.post("/forecast/")
def post_forecast(request: ForecastPost):
    global last_forecast
    df = get_data(request.symbol, request.train_size, request.start_time)
    result, _ = forecast_neuralprophet_rolling_with_open_price(df, request.predict_size)
    last_forecast = result
    return result

@app.get("/forecast/", response_model=ForecastGet)
def get_last_forecast():
    if not last_forecast:
        raise HTTPException(status_code=404, detail="Nenhuma previsão encontrada")
    return last_forecast
