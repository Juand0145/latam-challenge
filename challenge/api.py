import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import logging

# Import DelayModel with handling for different import paths
try:
    from challenge.model import DelayModel
except ImportError:
    from model import DelayModel

app = fastapi.FastAPI()

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    Fecha_I: str
    Fecha_O: str
    DIA: int
    AÑO: int
    Vlo_I: str
    Ori_I: str
    Des_I: str
    Emp_I: str
    Vlo_O: str
    Ori_O: str
    Des_O: str
    Emp_O: str
    DIANOM: str
    SIGLAORI: str
    SIGLADES: str

class PredictRequest(BaseModel):
    flights: List[FlightData]

class PredictResponse(BaseModel):
    predict: List[int]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the DelayModel
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", response_model=PredictResponse)
async def post_predict(request: PredictRequest):

    # Función para determinar si una fecha está en temporada alta
    def is_high_season(date):
        if (date.month == 12 and date.day >= 15) or (date.month == 1 or date.month == 2 or (date.month == 3 and date.day <= 3)) or (date.month == 7 and 15 <= date.day <= 31) or (date.month == 9 and 11 <= date.day <= 30):
            return 1
        return 0

    # Función para determinar el período del día
    def get_period_day(hour):
        if 5 <= hour <= 11:
            return 'morning'
        elif 12 <= hour <= 18:
            return 'afternoon'
        else:
            return 'night'

    if model is None:
        logger.error("Model is not initialized")
        raise HTTPException(status_code=500, detail="Model is not initialized")


    #Preprocessing data for prediction
    try:
        # Convert request data to DataFrame
        flights_list = [flight.dict() for flight in request.flights]

        # Transformar cada diccionario
        for flight in flights_list:
            # Convertir las fechas a datetime
            fecha_i = pd.to_datetime(flight['Fecha_I'])
            fecha_o = pd.to_datetime(flight['Fecha_O'])
            
            # Añadir columnas adicionales
            flight['high_season'] = is_high_season(fecha_i)
            flight['min_diff'] = (fecha_o - fecha_i).total_seconds() / 60
            flight['period_day'] = get_period_day(fecha_i.hour)
            flight['delay'] = 1 if flight['min_diff'] > 15 else 0

        # Crear la estructura final
        payload = {"flights": flights_list}
    except Exception as e:
        logger.error(f"Data preprocessing error: {e}")


        raise HTTPException(status_code=400, detail=f"Data preprocessing error: {str(e)}")
    
    # Make predictions
    try:
        predictions = model.predict(payload)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
    return PredictResponse(predict=predictions)
