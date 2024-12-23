from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import uvicorn
import torch
import os

app = FastAPI()

# Variáveis globais para armazenar a última previsão
last_forecast = {}

class ForecastPost(BaseModel):
    symbol: str
    predict_size: int
    train_size: int
    start_time: str = None  # Novo campo opcional para data e hora de início
    lagged_regressor : int = 0 # campo opcional para cálculo de regressores usando lagged_regressor
    mode : str = "Future" # campo opcional para escolha do modo

class ForecastGet(BaseModel):
    flag: int
    predicted_value: float
    last_prediction_time: str

def round_to_half(value):
    if pd.isna(value):
        return value
    return round(value * 2) / 2

def get_data(symbol, train_size, predict_size, start_time=None, lagged_regressor=0, mode="Future"):

    if mode == "Future":
        data = train_size
    if mode == "Lag":
        data = lagged_regressor+train_size+(predict_size*2)

    if start_time:
        start_time_dt = pd.to_datetime(start_time)
        start_timestamp = int(start_time_dt.timestamp())
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, start_timestamp, data)

    else:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, data)

    if rates is None:
        raise HTTPException(status_code=500, detail=f"Erro ao obter os dados: {mt5.last_error()}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df['dayOfWeek'] = df['time'].dt.day_name()

    df = df.rename(columns={'time': 'ds', 'close': 'y'})
    df = df[['ds', 'y', 'open', 'tick_volume', 'real_volume']]
    
    return df

def lagging_df(df, predict_size):

    colunas_para_shift = ['open', 'tick_volume', 'real_volume']

    # Aplicando o shift nas colunas especificadas
    df[colunas_para_shift] = df[colunas_para_shift].shift(periods=predict_size)

    # Removendo as linhas com valores nulos
    df.dropna(subset=['open'], inplace=True)

    return df

def forecast_neuralprophet_rolling_with_open_price(df, predict_size=30, lagged_regressor=0, mode="Future"):
    from neuralprophet import NeuralProphet
    model = NeuralProphet()
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())

    if mode == 'Lag':
        ## para usar lagging
        df = lagging_df(df, predict_size)
        df.dropna(subset=['y'], inplace=True)

    if mode == "Future":

        last_date = df['ds'].iloc[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=predict_size, freq='T')
        future_dates = pd.DataFrame({'ds': dates})
        future_dates['open'] = df['open'].iloc[-1]
        future_dates['tick_volume'] = df['tick_volume'].iloc[-1]
        future_dates['real_volume'] = df['real_volume'].iloc[-1]
        future_dates['y'] = None

        # print(future_dates)

        df = pd.concat([df, future_dates], ignore_index=True)

        model.add_future_regressor('open')
        model.add_future_regressor('tick_volume')
        model.add_future_regressor('real_volume')

    if mode == "Lag":

        model.add_lagged_regressor('open', n_lags=lagged_regressor)
        model.add_lagged_regressor('tick_volume', n_lags=lagged_regressor)
        model.add_lagged_regressor('real_volume', n_lags=lagged_regressor)


    model.fit(df)
    forecast = model.predict(df)

    forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)
    
    df['yhat1'] = forecast['yhat1'].values

    # print(df.tail(predict_size+5)) 

    yhat_pred = df['yhat1'].iloc[-1] 

    # Removendo as linhas com valores nulos
    # df.dropna(subset=['y'], inplace=True)

    last_diff = yhat_pred - df['y'].iloc[-(predict_size+1)]
    flag = 1 if last_diff > 0 else 0

    print(df.tail(predict_size+5))

    result = {
        "flag": flag,
        "predicted_value": yhat_pred,
        "last_prediction_time": df['ds'].iloc[-1].strftime("%Y.%m.%d %H:%M:%S"),
    }

    # print(df)
    return result, df

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
    df = get_data(request.symbol, request.train_size, request.predict_size, request.start_time, request.lagged_regressor, request.mode)
    result, _ = forecast_neuralprophet_rolling_with_open_price(df, request.predict_size, request.lagged_regressor, request.mode)
    last_forecast = result
    return result

@app.get("/forecast/", response_model=ForecastGet)
def get_last_forecast():
    if not last_forecast:
        raise HTTPException(status_code=404, detail="Nenhuma previsão encontrada")
    return last_forecast

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)