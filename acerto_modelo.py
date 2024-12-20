import pandas as pd
import numpy as np

file = 'forecast_neuralprophet_rolling_with_volume_30_min_300.csv'

# Carregar os dados
df = pd.read_csv(file)

# Vou usar um critério simples de "outlier" com base no desvio padrão
# threshold = 10000  # Este valor pode ser ajustado dependendo do seu contexto

# # Identificar valores aberrantes (outliers) na coluna 'yhat1'
# df['is_outlier'] = (df['yhat1'].abs() > threshold)

# Remover ou corrigir os outliers
# df.loc[df['is_outlier'], 'yhat1'] = np.nan  # Substituir os outliers por NaN

# Preencher valores NaN com a última previsão válida (método simples)
df['yhat1'].fillna(method='ffill', inplace=True)

# Criar uma coluna de direção real (1 para subida, -1 para queda)
df['real_direction'] = (df['y'] - df['y'].shift(1)).apply(lambda x: 1 if x > 0 else -1)

# Criar uma coluna de direção prevista (1 para subida, -1 para queda) comparando y com yhat1
df['predicted_direction'] = (df['y'] - df['yhat1'].shift(1)).apply(lambda x: 1 if x > 0 else -1)

# Comparar a direção real com a prevista
df['correct'] = df['real_direction'] == df['predicted_direction']

# Calcular a diferença real entre os valores consecutivos
df['real_diff'] = df['y'] - df['y'].shift(1)

df['predicted_diff'] = df['y'] - df['yhat1'].shift(1)

# df = df[(df['predicted_diff'] >= 5) | (df['predicted_diff'] <= -5)]

# Somar os ganhos e perdas
df['score'] = df.apply(lambda row: row['real_diff'] if row['correct'] else -row['real_diff'], axis=1)

df['score'] = df.apply(lambda row: abs(row['score']) if row['correct'] else -abs(row['score']), axis=1)

df.to_csv(f"{file}_results.csv", index=False)

# Somar os pontos totais
total_score = df['score'].sum()

# Calcular a porcentagem de acertos
accuracy = df['correct'].mean() * 100

# Calcular o número de dias distintos
unique_days = pd.to_datetime(df['ds']).dt.date.nunique()

# Exibir as colunas de direção e acertos
print("\nDireção real, direção prevista, acerto e pontuação:")
print(df[['ds', 'y', 'yhat1', 'real_direction', 'predicted_direction', 'correct', 'score', 'predicted_diff']])

# Printar o resultado
print(f'Porcentagem de acertos do modelo: {accuracy:.2f}%')
# print(f'Ganhos: {total_score * 10} reais em {unique_days} dias')
