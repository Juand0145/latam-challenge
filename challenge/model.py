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
        # Feature engineering
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])
        data['high_season'] = data['Fecha-I'].apply(lambda x: 1 if (x.month == 12 and x.day >= 15) or (x.month == 3 and x.day <= 3) or (x.month == 7 and x.day >= 15 and x.day <= 31) or (x.month == 9 and x.day >= 11 and x.day <= 30) else 0)
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        data['period_day'] = data['Fecha-I'].dt.hour.apply(lambda x: 'morning' if 5 <= x <= 11 else ('afternoon' if 12 <= x <= 18 else 'night'))
        if target_column:
            data['delay'] = data['min_diff'].apply(lambda x: 1 if x > 15 else 0)

        # Feature selection
        features = ['high_season', 'min_diff', 'period_day', 'DIA', 'MES', 'AÑO', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES']
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
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            self._model = joblib.load('model.pkl')
        
        predictions = self._model.predict(features)
        return predictions.tolist()

if __name__ == "__main__":
    model = DelayModel()
    data = pd.read_csv('data/data.csv')
    X, y = model.preprocess(data, target_column='delay')
    model.fit(X, y)
    # For prediction, you would preprocess new data and call model.predict(new_features)
