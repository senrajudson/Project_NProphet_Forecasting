def train_and_forecast_single(
    df_treinamento,
    #dado_previsao,
    time_col,
    target_col,
    lagged_regressor_cols=None,
    future_regressor_cols=None,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
):
    """
    Treina e prevê com o modelo NeuralProphet para uma única janela.

    Parâmetros:
    - df: DataFrame com os dados (deve conter colunas "ds" e "y").
    - dado_treinamento: Número de passos passados para treinamento.
    - dado_previsao: Número de passos para previsão.
    - lagged_regressor_cols: Lista de regressoras defasadas (opcional).
    - future_regressor_cols: Lista de regressoras futuras (opcional).
    - yearly_seasonality: Ativar sazonalidade anual.
    - weekly_seasonality: Ativar sazonalidade semanal.
    - daily_seasonality: Ativar sazonalidade diária.

    Retorna:
    - Um DataFrame com as previsões.
    """

    # Renomear as Colunas
    df_treinamento.rename(columns={time_col: "ds", target_col: "y"}, inplace=True)
    df_treinamento["ds"] = pd.to_datetime(df_treinamento["ds"])


    model = NeuralProphet(
        # yearly_seasonality=False,  # Desabilita sazonalidade anual
        # weekly_seasonality=False,  # Desabilita sazonalidade semanal
        # daily_seasonality=False,  # Desabilita sazonalidade diária
        # # n_forecasts=forecast_size,  # Define o número de previsões
        # # n_lags=history_size,  # Define o número de defasagens (lags)
        #n_lags=40,        # Usar 600 valores anteriores
        n_forecasts=40,    # Prever os próximos 40 valores
    )

    # Define o dispositivo (CPU ou GPU)
    if torch.cuda.is_available():
        model.device =  "cuda"
    else:
        model.device = "cpu"

    print(f"model.device: {model.device}")

        # Adiciona as regressoras defasadas
    if len(lagged_regressor_cols)>0:
        model.add_lagged_regressor(lagged_regressor_cols, n_lags=40)


    if len(future_regressor_cols)>0:
        model.add_future_regressor(future_regressor_cols, n_lags=40)


    # Retirar as colunas que não são necessárias
    # Identificar colunas extras que não são "ds", "y", ou regressoras especificadas
    cols_to_keep = ["ds", "y"]  # Colunas principais
    if lagged_regressor_cols:
        cols_to_keep.extend(lagged_regressor_cols,)  # Adicionar regressoras defasadas
    if future_regressor_cols:
        cols_to_keep.extend(future_regressor_cols)  # Adicionar regressoras futuras

    # Filtrar o DataFrame para manter apenas as colunas necessárias
    df_treinamento = df_treinamento[cols_to_keep]

    

    print("Colunas do DataFrame:", df_treinamento.columns.tolist())

    print(df_treinamento)

    # Treinar o modelo na janela de treinamento
    model.fit(df_treinamento,  freq="1min",)

    # Create a new dataframe reaching 365 into the future for our forecast, n_historic_predictions also shows historic data
    df_future = model.make_future_dataframe(df_treinamento, n_historic_predictions=False, periods=40)

    print (df_future)
    print(len(df_future))

    # Previsão na janela de 
    df_previsao = model.predict(df_future)

    print(df_previsao.columns.tolist())

    print(df_previsao)

    print(len(df_previsao))


    df_previsao.to_csv("./igor.csv")




    # # Valores para o resumo:
    # # Último valor real do dataset de treinamento 
    # ultimo_valor_real_treinamento = dado_treinamento["y"].iloc[-1]

    # #Primeiro valor real do dataset de previsão
    # primeiro_valor_real_previsao = previsao["y"].iloc[0]

    # #Primeiro valor previsto do dataset de previsão
    # primeiro_valor_previsto = previsao["yhat1"].iloc[1]

    # #Último valor previsto do dataset de previsão
    # ultimo_valor_previsto = previsao["yhat1"].iloc[-1]

    # # 2. Último valor Real do dataset de previsão
    # ultimo_previsao_real = previsao["y"].iloc[-1]


    # ###############  Calcula se subiu ou deceu real
    # diferenca_real = ultimo_previsao_real - ultimo_valor_real_treinamento 

    # diferenca_prev = ultimo_valor_previsto - primeiro_valor_previsto 

    # if (diferenca_real > 0 and diferenca_prev > 0) or (diferenca_real < 0 and diferenca_prev < 0):
    #     resultado = "acertou"  # Sinais iguais
    # else:
    #     resultado = "errou"  # Sinais diferentes

    # # Adicionar os dados ao resumo
    #     previsao_resumida = {
    #         "Ult. Val. Real. DT:trei": ultimo_valor_real_treinamento,
    #         "Prim. Val. Real DT:val": primeiro_valor_real_previsao,
    #         "Prim. Val. Prev DT:val": primeiro_valor_previsto,
    #         "Ult. Val. Real DT:val": ultimo_previsao_real,
    #         "Ult. Val. Prev DT:val ": ultimo_valor_previsto,
    #         "dif ult - prim real": diferenca_real,
    #         "dif ult - prim prev": diferenca_prev,
    #         "Resultado tendencia modelo": resultado
    #     }
    

    return  