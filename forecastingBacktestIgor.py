from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import torch
import os

# Função para arredondar para o múltiplo mais próximo de 0,5
def round_to_half(value):
    if pd.isna(value):
        return value
    return round(value * 2) / 2

def get_data(symbol, timeframe, num_bars, predict_size, train_size, lagged_regressor=0, start_pos = 0):

    # solicitamos 10 barras de GBPUSD D1 do dia atual
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, num_bars+predict_size+lagged_regressor+train_size)

    if rates is None:
        print("Erro ao obter os dados:", mt5.last_error())
        mt5.shutdown()
        quit()

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

def future_df(df, train_size, predict_size):

    # Verifica se o DataFrame possui linhas suficientes
    if len(df) < train_size + predict_size:
        raise ValueError("O DataFrame não possui linhas suficientes para os tamanhos especificados.")

    # Bloco de treinamento
    train_block = df.iloc[:train_size]
    # Bloco de previsão
    predict_block = df.iloc[train_size:train_size + predict_size].copy()

    last_train_row = train_block.iloc[-1]
    # Modifica o bloco de previsão
    predict_block['open'] = last_train_row['open']
    predict_block['tick_volume'] = last_train_row['tick_volume']
    predict_block['real_volume'] = last_train_row['real_volume']
    predict_block['close'] = None

    # Combina os blocos
    result_df = pd.concat([train_block, predict_block])

    return result_df

def forecast_neuralprophet_rolling_with_volume(df, train_size=50, predict_size=10, max_data=600, lagged_regressor=0, mode="Lag"):
    from neuralprophet import NeuralProphet

    # Limitar o dataframe a 'max_data'
    df = df.head(max_data)

    cols_toregressor = ['open', 'tick_volume', 'real_volume']

    results = []
    total_points = len(df)

    if total_points < train_size + predict_size:
        raise ValueError("O dataframe tem poucos dados para realizar o teste.")

    for start in range(0, total_points - train_size - predict_size + 1, predict_size):
        model = NeuralProphet(n_forecasts=predict_size)
        train_df = df.iloc[start:start + train_size]
        
        if mode == "Future":
            test_df = df.iloc[start + train_size:start + train_size + predict_size]
            test_df[cols_toregressor] = train_df[cols_toregressor].iloc[-1].values
            test_df["y"] = None

            # model.add_future_regressor(cols_toregressor)

            model.add_future_regressor('open')
            model.add_future_regressor('tick_volume')
            model.add_future_regressor('real_volume')

        if mode == "Lag":
            ## hist 180
            test_df = model.make_future_dataframe(train_df, n_historic_predictions=predict_size, periods=predict_size)

            model.add_lagged_regressor(cols_toregressor)

        model.fit(train_df, freq="1min")

        forecast = model.predict(test_df)

        if mode == "Lag":
            # Obter apenas as colunas que começam com 'yhat'
            yhat_cols = [col for col in forecast.columns if col.startswith('yhat')]

            # Empilhar os valores previstos em uma única coluna
            forecast_long = forecast[['ds'] + yhat_cols].melt(
                id_vars=['ds'], 
                value_vars=yhat_cols, 
                var_name='yhat_column', 
                value_name='yhat'
            )

            # Ajustar o índice para alinhar corretamente as previsões com as linhas do dataframe de comparação
            forecast_long['index'] = forecast_long.groupby('yhat_column').cumcount()

            # Criar dataframe de comparação
            comparison = df.iloc[start + train_size:start + train_size + predict_size][['ds', 'y']].copy()
            comparison.reset_index(drop=True, inplace=True)
            comparison[cols_toregressor] = train_df[cols_toregressor]

            # Combinar previsões ao dataframe de comparação
            comparison = pd.merge(
                comparison.reset_index(),
                forecast_long[['index', 'yhat']],
                left_index=True,
                right_on='index',
                how='left'
            )

            comparison['yhat'] = comparison['yhat'].apply(round_to_half)  # Aplicar round_to_half se necessário
            comparison.drop(columns=['index'], inplace=True)

        if mode == 'Future':

            forecast['yhat1'] = forecast['yhat1'].apply(round_to_half)

            test_df['y'] = df['y'].iloc[start + train_size:start + train_size + predict_size]
            comparison = test_df[['ds', 'y']].copy()
            comparison['yhat'] = forecast['yhat1'].values
            comparison[cols_toregressor] = test_df[cols_toregressor]


        # comparison = comparison.iloc[::predict_size].reset_index(drop=True)
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
    num_bars = 180
    train_size = 50
    predict_size = 30
    lagged_regressor = 0
    mode = 'Lag'

    df = get_data(symbol, timeframe, num_bars, predict_size, train_size)
    df.to_csv(f"new_stats_{mode}.csv", index=False)

    # df = pd.read_csv("btc_last_6000.csv")
    comparison = forecast_neuralprophet_rolling_with_volume(df, train_size, predict_size, num_bars , lagged_regressor, mode)
    file = f"forecast_neuralprophet_rolling_with_volume_{train_size}_{predict_size}_{lagged_regressor}_min_{num_bars}_{mode}"
    comparison.to_csv(f"{file}.csv", index=False)

    # Exibir previsões
    print(comparison)
    print(file)
