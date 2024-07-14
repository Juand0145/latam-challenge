import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
from imblearn.over_sampling import SMOTE


class DelayModel:
    def __init__(self):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
        ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
       # Apply data Preprocesiong
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])
        data['high_season'] = data['Fecha-I'].apply(lambda x: 1 if (x.month == 12 and x.day >= 15) or (x.month == 3 and x.day <= 3) or (x.month == 7 and x.day >= 15 and x.day <= 31) or (x.month == 9 and x.day >= 11 and x.day <= 30) else 0)
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        data['period_day'] = data['Fecha-I'].dt.hour.apply(lambda x: 'morning' if 5 <= x <= 11 else ('afternoon' if 12 <= x <= 18 else 'night'))

        # Get the label Value
        data['delay'] = data['min_diff'].apply(lambda x: 1 if x > 15 else 0)

        # Feature selection
        features = ['high_season', 'period_day', 'DIA', 'MES', 'AÑO', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES']
        # features = ['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']

        X = pd.get_dummies(data[features], drop_first=True)
        
        if target_column:
            y = data[target_column]
            return X, y
        else:
            return X

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Balancing the classes
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(features, target)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Model training
        self._model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        joblib.dump(self._model, 'model.pkl')

    def predict(
        self,
        payload
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        # Dowload model
        if self._model is None:
            self._model = joblib.load('model.pkl')

        # Definir las columnas del DataFrame
        columns = [
            'high_season', 'DIA', 'MES', 'AÑO', 'period_day_morning',
            'period_day_night', 'TIPOVUELO_N', 'OPERA_Aeromexico',
            'OPERA_Air Canada', 'OPERA_Air France', 'OPERA_Alitalia',
            'OPERA_American Airlines', 'OPERA_Austral', 'OPERA_Avianca',
            'OPERA_British Airways', 'OPERA_Copa Air', 'OPERA_Delta Air',
            'OPERA_Gol Trans', 'OPERA_Grupo LATAM', 'OPERA_Iberia',
            'OPERA_JetSmart SPA', 'OPERA_K.L.M.', 'OPERA_Lacsa',
            'OPERA_Latin American Wings', 'OPERA_Oceanair Linhas Aereas',
            'OPERA_Plus Ultra Lineas Aereas', 'OPERA_Qantas Airways',
            'OPERA_Sky Airline', 'OPERA_United Airlines', 'SIGLADES_Arica',
            'SIGLADES_Asuncion', 'SIGLADES_Atlanta', 'SIGLADES_Auckland N.Z.',
            'SIGLADES_Balmaceda', 'SIGLADES_Bariloche', 'SIGLADES_Bogota',
            'SIGLADES_Buenos Aires', 'SIGLADES_Calama', 'SIGLADES_Cancun',
            'SIGLADES_Castro (Chiloe)', 'SIGLADES_Cataratas Iguacu',
            'SIGLADES_Ciudad de Mexico', 'SIGLADES_Ciudad de Panama',
            'SIGLADES_Cochabamba', 'SIGLADES_Concepcion', 'SIGLADES_Copiapo',
            'SIGLADES_Cordoba', 'SIGLADES_Curitiba, Bra.', 'SIGLADES_Dallas',
            'SIGLADES_Florianapolis', 'SIGLADES_Guayaquil', 'SIGLADES_Houston',
            'SIGLADES_Iquique', 'SIGLADES_Isla de Pascua', 'SIGLADES_La Paz',
            'SIGLADES_La Serena', 'SIGLADES_Lima', 'SIGLADES_Londres',
            'SIGLADES_Los Angeles', 'SIGLADES_Madrid', 'SIGLADES_Melbourne',
            'SIGLADES_Mendoza', 'SIGLADES_Miami', 'SIGLADES_Montevideo',
            'SIGLADES_Neuquen', 'SIGLADES_Nueva York', 'SIGLADES_Orlando',
            'SIGLADES_Osorno', 'SIGLADES_Paris', 'SIGLADES_Pisco, Peru',
            'SIGLADES_Puerto Montt', 'SIGLADES_Puerto Natales',
            'SIGLADES_Puerto Stanley', 'SIGLADES_Punta Arenas',
            'SIGLADES_Punta Cana', 'SIGLADES_Punta del Este', 'SIGLADES_Quito',
            'SIGLADES_Rio de Janeiro', 'SIGLADES_Roma', 'SIGLADES_Rosario',
            'SIGLADES_San Juan, Arg.', 'SIGLADES_Santa Cruz', 'SIGLADES_Sao Paulo',
            'SIGLADES_Sydney', 'SIGLADES_Temuco', 'SIGLADES_Toronto',
            'SIGLADES_Tucuman', 'SIGLADES_Ushuia', 'SIGLADES_Valdivia',
            'SIGLADES_Washington'
        ]

        print("-"*100)
        print(self._model.predict(payload))
        print("-"*100)

        try:
            predictions = self._model.predict(payload)
            return predictions.tolist()
        except Exception:
            pass


        # Crear un DataFrame vacío con las columnas definidas
        df_payload = pd.DataFrame(columns=columns)

        # Convertir el payload a un DataFrame
        rows = []
        for flight in payload['flights']:
            row = {col: 0 for col in columns}  # Inicializar todas las columnas con 0
            row.update({
                "high_season": int(flight["high_season"]),
                "DIA": int(flight["DIA"]),
                "MES": int(flight["MES"]),
                "AÑO": int(flight["AÑO"]),
                "period_day_morning": 1 if flight["period_day"] == "morning" else 0,
                "period_day_night": 1 if flight["period_day"] == "night" else 0,
                "TIPOVUELO_N": 1 if flight["TIPOVUELO"] == "N" else 0,
                f"OPERA_{flight['OPERA']}": 1,
                f"SIGLADES_{flight['SIGLADES']}": 1
            })
            rows.append(row)

        # Crear un DataFrame con las filas y concatenarlas
        df_payload = pd.concat([df_payload, pd.DataFrame(rows)], ignore_index=True)

        # Convertir todas las columnas al tipo de dato adecuado
        df_payload = df_payload.astype({
    "high_season": int,
    "DIA": int,
    "MES": int,
    "AÑO": int,
    "period_day_morning": int,
    "period_day_night": int,
    "TIPOVUELO_N": int,
    "OPERA_Aeromexico": int,
    "OPERA_Air Canada": int,
    "OPERA_Air France": int,
    "OPERA_Alitalia": int,
    "OPERA_American Airlines": int,
    "OPERA_Austral": int,
    "OPERA_Avianca": int,
    "OPERA_British Airways": int,
    "OPERA_Copa Air": int,
    "OPERA_Delta Air": int,
    "OPERA_Gol Trans": int,
    "OPERA_Grupo LATAM": int,
    "OPERA_Iberia": int,
    "OPERA_JetSmart SPA": int,
    "OPERA_K.L.M.": int,
    "OPERA_Lacsa": int,
    "OPERA_Latin American Wings": int,
    "OPERA_Oceanair Linhas Aereas": int,
    "OPERA_Plus Ultra Lineas Aereas": int,
    "OPERA_Qantas Airways": int,
    "OPERA_Sky Airline": int,
    "OPERA_United Airlines": int,
    "SIGLADES_Arica": int,
    "SIGLADES_Asuncion": int,
    "SIGLADES_Atlanta": int,
    "SIGLADES_Auckland N.Z.": int,
    "SIGLADES_Balmaceda": int,
    "SIGLADES_Bariloche": int,
    "SIGLADES_Bogota": int,
    "SIGLADES_Buenos Aires": int,
    "SIGLADES_Calama": int,
    "SIGLADES_Cancun": int,
    "SIGLADES_Castro (Chiloe)": int,
    "SIGLADES_Cataratas Iguacu": int,
    "SIGLADES_Ciudad de Mexico": int,
    "SIGLADES_Ciudad de Panama": int,
    "SIGLADES_Cochabamba": int,
    "SIGLADES_Concepcion": int,
    "SIGLADES_Copiapo": int,
    "SIGLADES_Cordoba": int,
    "SIGLADES_Curitiba, Bra.": int,
    "SIGLADES_Dallas": int,
    "SIGLADES_Florianapolis": int,
    "SIGLADES_Guayaquil": int,
    "SIGLADES_Houston": int,
    "SIGLADES_Iquique": int,
    "SIGLADES_Isla de Pascua": int,
    "SIGLADES_La Paz": int,
    "SIGLADES_La Serena": int,
    "SIGLADES_Lima": int,
    "SIGLADES_Londres": int,
    "SIGLADES_Los Angeles": int,
    "SIGLADES_Madrid": int,
    "SIGLADES_Melbourne": int,
    "SIGLADES_Mendoza": int,
    "SIGLADES_Miami": int,
    "SIGLADES_Montevideo": int,
    "SIGLADES_Neuquen": int,
    "SIGLADES_Nueva York": int,
    "SIGLADES_Orlando": int,
    "SIGLADES_Osorno": int,
    "SIGLADES_Paris": int,
    "SIGLADES_Pisco, Peru": int,
    "SIGLADES_Puerto Montt": int,
    "SIGLADES_Puerto Natales": int,
    "SIGLADES_Puerto Stanley": int,
    "SIGLADES_Punta Arenas": int,
    "SIGLADES_Punta Cana": int,
    "SIGLADES_Punta del Este": int,
    "SIGLADES_Quito": int,
    "SIGLADES_Rio de Janeiro": int,
    "SIGLADES_Roma": int,
    "SIGLADES_Rosario": int,
    "SIGLADES_San Juan, Arg.": int,
    "SIGLADES_Santa Cruz": int,
    "SIGLADES_Sao Paulo": int,
    "SIGLADES_Sydney": int,
    "SIGLADES_Temuco": int,
    "SIGLADES_Toronto": int,
    "SIGLADES_Tucuman": int,
    "SIGLADES_Ushuia": int,
    "SIGLADES_Valdivia": int,
    "SIGLADES_Washington": int
})
        
        predictions = self._model.predict(df_payload)
        return predictions.tolist()
