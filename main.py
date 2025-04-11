# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель данных запроса
class CalculationRequest(BaseModel):
    input1: float
    input2: float

# Модель данных ответа
class CalculationResponse(BaseModel):
    result: float

@app.post("/calculate", response_model=CalculationResponse)
async def calculate(request: CalculationRequest):
    # Ваша логика вычислений
    result = request.input1 + request.input2  # пример
    
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)
