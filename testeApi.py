import requests
import time

BASE_URL = "http://127.0.0.1:8000"  # URL base da API
FORECAST_ENDPOINT = "/forecast"

# Testando o endpoint POST /forecast
def test_post_forecast():
    payload = {
        "symbol": "WDO@D",   # Substitua pelo símbolo usado no MetaTrader5
        "train_size": 50,     # Tamanho do dataset de treino
        "predict_size": 30,    # Tamanho da previsão
        "start_time": "2024-05-28T12:52:00",  # Data e hora do passado (exemplo)
    }

    response = requests.post(f"{BASE_URL}{FORECAST_ENDPOINT}", json=payload)

    if response.status_code == 200:
        print("POST /forecast:")
        print(response.json())
    else:
        print(f"Erro ao testar POST /forecast: {response.status_code} - {response.text}")

# Testando o endpoint GET /forecast
def test_get_forecast():
    response = requests.get(f"{BASE_URL}{FORECAST_ENDPOINT}")

    if response.status_code == 200:
        print("GET /forecast:")
        print(response.json())
    else:
        print(f"Erro ao testar GET /forecast: {response.status_code} - {response.text}")

if __name__ == "__main__":
    if test_post_forecast():
        time.sleep(60)
        test_get_forecast()
