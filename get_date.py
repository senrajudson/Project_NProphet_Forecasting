from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import os

def get_data(symbol, timeframe, num_bars, start_pos = 0):

    # solicitamos 10 barras de GBPUSD D1 do dia atual
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, num_bars)

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

# Função para arredondar para o múltiplo mais próximo de 0,5
def round_to_half(value):
    return round(value * 2) / 2

# def forecast_neuralprophet_rolling(df, train_size=50, predict_size=10, max_data=500):
#     from neuralprophet import NeuralProphet
#     import pandas as pd

#     # Limitar o dataframe a 'max_data'
#     df = df.head(max_data)
    
#     # Renomear colunas
#     df = df.rename(columns={'time': 'ds', 'close': 'y'})
#     df = df[['ds', 'y']]

#     results = []
#     total_points = len(df)

#     if total_points < train_size + predict_size:
#         raise ValueError("O dataframe tem poucos dados para realizar o teste.")

#     for start in range(0, total_points - train_size - predict_size + 1, predict_size):
#         train_df = df.iloc[start:start + train_size]
#         test_df = df.iloc[start + train_size:start + train_size + predict_size]

#         model = NeuralProphet()
#         model.fit(train_df)
#         forecast = model.predict(test_df)

#         forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)

#         comparison = test_df[['ds', 'y']].copy()
#         # comparison['yhat1'] = forecast['yhat1'].values
#         comparison['yhat1'] = forecast['yhat1'].values
#         # Manter somente as linhas a cada 10 índices
#         comparison = comparison.iloc[::predict_size].reset_index(drop=True)

#         results.append(comparison)

#     return pd.concat(results, ignore_index=True)

# def forecast_neuralprophet_rolling_with_open_price(df, train_size=50, predict_size=10, max_data=500):
#     from neuralprophet import NeuralProphet
#     import pandas as pd

#     # Limitar o dataframe a 'max_data'
#     df = df.head(max_data)
    
#     # Renomear colunas
#     df = df.rename(columns={'time': 'ds', 'close': 'y'})
#     df = df[['ds', 'y', "open"]]

#     results = []
#     total_points = len(df)

#     if total_points < train_size + predict_size:
#         raise ValueError("O dataframe tem poucos dados para realizar o teste.")

#     for start in range(0, total_points - train_size - predict_size + 1, predict_size):
#         train_df = df.iloc[start:start + train_size]
#         test_df = df.iloc[start + train_size:start + train_size + predict_size]

#         model = NeuralProphet()
#         model.add_future_regressor('open')
#         model.fit(train_df)
#         forecast = model.predict(test_df)

#         forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)

#         comparison = test_df[['ds', 'y']].copy()
#         # comparison['yhat1'] = forecast['yhat1'].values
#         comparison['yhat1'] = forecast['yhat1'].values
#         # Manter somente as linhas a cada 10 índices
#         comparison = comparison.iloc[::predict_size].reset_index(drop=True)

#         results.append(comparison)

#     return pd.concat(results, ignore_index=True)

def forecast_neuralprophet_rolling_with_volume(df, train_size=50, predict_size=10, max_data=500):
    from neuralprophet import NeuralProphet
    import pandas as pd

    # Limitar o dataframe a 'max_data'
    df = df.head(max_data)
    
    # Renomear colunas
    df = df.rename(columns={'time': 'ds', 'close': 'y'})  # 'time' vira 'ds' e 'close' vira 'y'
    df = df[['ds', 'y', "open", "tick_volume", "real_volume"]]  # Incluir 'volume' como variável de regressor

    results = []
    total_points = len(df)

    if total_points < train_size + predict_size:
        raise ValueError("O dataframe tem poucos dados para realizar o teste.")

    for start in range(0, total_points - train_size - predict_size + 1, predict_size):
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + predict_size]

        # Usar o último valor dos regressores do conjunto de treinamento para preencher o conjunto de teste
        last_open = train_df['open'].iloc[-1]
        last_tick_volume = train_df['tick_volume'].iloc[-1]
        last_real_volume = train_df['real_volume'].iloc[-1]

        test_df['open'] = last_open
        test_df['tick_volume'] = last_tick_volume
        test_df['real_volume'] = last_real_volume



        model = NeuralProphet()
        model.add_future_regressor('open')
        model.add_future_regressor('tick_volume')
        model.add_future_regressor('real_volume')
        model.fit(train_df)
        forecast = model.predict(test_df)

        forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)

        comparison = test_df[['ds', 'y']].copy()
        comparison['yhat1'] = forecast['yhat1'].values
        comparison["open"] = test_df['open']
        comparison["tick_volume"] = test_df['tick_volume']
        comparison["real_volume"] = test_df['real_volume']

        comparison = comparison.iloc[::predict_size].reset_index(drop=True)

        results.append(comparison)

    return pd.concat(results, ignore_index=True)

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
    num_bars = 1200
    train_size = 50
    predict_size = 30

    df = get_data(symbol, timeframe, num_bars, start_pos)
    comparison = forecast_neuralprophet_rolling_with_volume(df, train_size, predict_size, max_data=num_bars)

    file = f"forecast_neuralprophet_rolling_with_volume_{predict_size}_min_{num_bars}"

    df.to_csv(f"new_stats.csv", index=False)
    comparison.to_csv(f"{file}.csv", index=False)

    # Exibir previsões
    print(comparison)
    print(file)
