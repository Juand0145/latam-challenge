import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = ['high_season', 'min_diff', 'DIA', 'MES', 'AÑO', 'period_day_morning',
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
       'SIGLADES_Washington']

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="data/data.csv")
        

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.Series)
        assert target.name == self.TARGET_COL[0]


    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)


    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] > 0.60
        assert report["0"]["f1-score"] > 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30


    def test_model_predict(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)