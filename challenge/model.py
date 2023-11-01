import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.exceptions import NotFittedError
import os
from typing import Tuple, Union, List
from datetime import datetime
import json

class DelayModel:

    def __init__(
        self
    ):
        self._model = XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = 4.44) # Model should be saved in this attribute.
        self.save_model_path = 'saved_model'

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ):
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
        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
            
            if ((fecha >= range1_min and fecha <= range1_max) or 
                (fecha >= range2_min and fecha <= range2_max) or 
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0
            
        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff
        
        def check_features(feat: pd.DataFrame) -> bool:
            dict_opera = open('challenge/opera.json')
            dict_opera = json.load(dict_opera)
            opera_list = dict_opera['OPERA']
            for col in feat.columns:
                if col.__contains__('TIPOVUELO'):
                    if col.replace('TIPOVUELO_', '') not in ['N', 'I']:
                        return False
                elif col.__contains__('OPERA'):
                    if col.replace('OPERA_', '') not in opera_list:
                        return False
                elif col.__contains__('MES'):
                    month = int(col.replace('MES_', ''))
                    if month > 12 or month < 0:
                        return False
            return True
                    
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        if target_column == None:
            check = check_features(features)

            if not check:
                raise Exception('Non-existent feature in input data')
            
            features = features.reindex(top_10_features, axis=1, fill_value=0)
            return features
        else:
            features = features.reindex(top_10_features, axis=1, fill_value=0)
            data['high_season'] = data['Fecha-I'].apply(is_high_season)
            data['min_diff'] = data.apply(get_min_diff, axis = 1)
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            target = data[[target_column]]
            return features, target

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
        self._model.fit(features, target)
        if self.save_model_path:
            if not os.path.exists(self.save_model_path):
                os.mkdir(self.save_model_path)
            self._model.save_model(self.save_model_path + '/model.json')

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
        try:
            preds = self._model.predict(features).tolist()
            return preds
        except NotFittedError:
            self._model.load_model(self.save_model_path + '/model.json')
            return self._model.predict(features).tolist()