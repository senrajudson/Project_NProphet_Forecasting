from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import os

def get_data(symbol, timeframe, num_bars, start_pos = 0):

    # solicitamos 10 barras de GBPUSD D1 do dia atual
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)

    if rates is None:
        print("Erro ao obter os dados:", mt5.last_error())
        mt5.shutdown()
        quit()

    # Criar um DataFrame com os dados retornados
    df = pd.DataFrame(rates)

    print(df)

    # Renomear as colunas para mais familiaridade
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'tick_volume', 'real_volume': "real_volume"})

    # Converter a coluna 'time' de segundos para datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Adicionar o nome do dia da semana
    df['dayOfWeek'] = df['time'].dt.day_name()

    return df

def round_to_half(value):
    if pd.isna(value):
        return value
    return round(value * 2) / 2

def forecast_neuralprophet_rolling_with_open_price(df, predict_size=30):
    from neuralprophet import NeuralProphet
    # Treinar o modelo
    model = NeuralProphet()
    result = []

    # Renomear colunas
    df = df.rename(columns={'time': 'ds', 'close': 'y'})
    df = df[['ds', 'y', 'open', "tick_volume", "real_volume"]]

    # Renomear colunas
    # df = df.rename(columns={'Data/Hora': 'ds', 'Fechamento': 'y'})  # 'time' vira 'ds' e 'close' vira 'y'
    # df = df[['ds', 'y', "Abertura", "Volume"]]
    # df.rename(columns={'Abertura': 'open', 'Volume': 'real_volume'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds'])

    # Criar timestamps futuros
    last_date = df['ds'].iloc[-1]
    dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1),
                                 periods=predict_size, freq='T')
    
    future_dates = pd.DataFrame({'ds': dates})

    # future_dates = model.make_future_dataframe(df, periods=predict_size)
    # future_dates['open'] = df['open'].iloc[-1]
    # future_dates['tick_volume'] = df['tick_volume'].iloc[-1]
    # future_dates['real_volume'] = df['real_volume'].iloc[-1]
    # future_dates['y'] = None
    # print(future_dates)
 
    model.add_lagged_regressor('open')
    model.add_lagged_regressor('tick_volume')
    model.add_lagged_regressor('real_volume')
    model.fit(df)

    # Fazer a previsão
    forecast = model.predict(future_dates)

    forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)
    
    future_dates['yhat1'] = forecast['yhat1'].values

    # Calcular o último valor da diferença
    last_diff = df['y'].iloc[-1] - future_dates['yhat1'].iloc[-1]

    # Definir 0 ou 1 com base na soma
    flag = 1 if last_diff < 0 else 0

    # Retornar o resultado

    result = [flag, future_dates['yhat1'].iloc[-1], future_dates['ds'].iloc[-1]]

    return result, future_dates

if __name__ == "__main__":
    # Exemplo de chamada
        
    load_dotenv()

    LOGIN = os.getenv('LOGIN')
    PASSWORD = os.getenv('PASSWORD')
    SERVER = os.getenv('SERVER')

    is_initialized = mt5.initialize()

    if is_initialized:
        print('initialized: ', is_initialized)
        print('\n')
    else:
        print('initialized: ', is_initialized)
        print(mt5.last_error())

    is_logged_in = mt5.login(int(LOGIN), PASSWORD, SERVER)

    # settings
    symbol = 'WDO@D'
    timeframe = mt5.TIMEFRAME_M1
    start_pos = 0
    train_size = 110
    predict_size = 30
    
    df = get_data(symbol, timeframe, train_size)

    # df = pd.read_csv("btc_last_100.csv")
    result, forecast = forecast_neuralprophet_rolling_with_open_price(df, predict_size)
    # comparison.to_csv(f"forecast_neuralprophet_rolling.csv", index=False)

    # Exibir previsões
    print(forecast)
    print(result)
