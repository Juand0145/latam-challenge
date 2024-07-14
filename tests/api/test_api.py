import unittest
from fastapi.testclient import TestClient
from challenge.api import app
import numpy as np
from challenge.model import DelayModel

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_should_get_predict(self):
        data = {
        "flights": [
            {
                "OPERA": "Air Canada",
                "TIPOVUELO": "N",
                "MES": 3,
                "Fecha_I": "2020-01-01 08:00:00",
                "Fecha_O": "2020-01-01 08:15:00",
                "DIA": 1,
                "AÑO": 2020,
                "Vlo_I": "100",
                "Ori_I": "SCL",
                "Des_I": "LIM",
                "Emp_I": "LA",
                "Vlo_O": "100",
                "Ori_O": "SCL",
                "Des_O": "LIM",
                "Emp_O": "LA",
                "DIANOM": "Wednesday",
                "SIGLAORI": "Santiago",
                "SIGLADES": "Lima",
                "high_season": 0,
                "min_diff": 15,
                "period_day": "morning",
                "delay": 1
            }
        ]
        }
        # Mock the model's predict method to always return [0]
        DelayModel.predict = lambda self, x: [0]
    
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13,
                    "Fecha_I": "2020-01-01 08:00:00",
                    "Fecha_O": "2020-01-01 08:15:00",
                    "DIA": 1,
                    "AÑO": 2020,
                    "Vlo_I": "100",
                    "Ori_I": "SCL",
                    "Des_I": "LIM",
                    "Emp_I": "LA",
                    "Vlo_O": "100",
                    "Ori_O": "SCL",
                    "Des_O": "LIM",
                    "Emp_O": "LA",
                    "DIANOM": "Wednesday",
                    "SIGLAORI": "Santiago",
                    "SIGLADES": "Lima"
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13,
                    "Fecha_I": "2020-01-01 08:00:00",
                    "Fecha_O": "2020-01-01 08:15:00",
                    "DIA": 1,
                    "AÑO": 2020,
                    "Vlo_I": "100",
                    "Ori_I": "SCL",
                    "Des_I": "LIM",
                    "Emp_I": "LA",
                    "Vlo_O": "100",
                    "Ori_O": "SCL",
                    "Des_O": "LIM",
                    "Emp_O": "LA",
                    "DIANOM": "Wednesday",
                    "SIGLAORI": "Santiago",
                    "SIGLADES": "Lima"
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13,
                    "Fecha_I": "2020-01-01 08:00:00",
                    "Fecha_O": "2020-01-01 08:15:00",
                    "DIA": 1,
                    "AÑO": 2020,
                    "Vlo_I": "100",
                    "Ori_I": "SCL",
                    "Des_I": "LIM",
                    "Emp_I": "LA",
                    "Vlo_O": "100",
                    "Ori_O": "SCL",
                    "Des_O": "LIM",
                    "Emp_O": "LA",
                    "DIANOM": "Wednesday",
                    "SIGLAORI": "Santiago",
                    "SIGLADES": "Lima"
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
