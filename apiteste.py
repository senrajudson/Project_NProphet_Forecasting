from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class InputData(BaseModel):
    text: str
    number: int

@app.post("/process/")
def process_data(data: InputData):
    response = {
        "message": f"Received text: {data.text}",
        "number_squared": data.number ** 2
    }
    logging.info(f"Resposta da API: {response}")  # Log da resposta
    return response
