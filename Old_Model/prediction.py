import datetime
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas.io.json import json_normalize
from xlrd import open_workbook
from xlutils.copy import copy


def prediction(dataframe, model):

    dataframe.shape
    date_time = pd.to_datetime(dataframe.pop('Zeit'), format='%d/%m/%Y %H:%M')
    dataframe.replace('', 0, inplace=True)
    dataframe = dataframe.astype('float')
    ziel = dataframe[['Graz-DB', 'Graz-MG', 'Graz-S', 'Graz-OP', 'Graz-N', 'Graz-W', 'Lustb']].replace(0, np.NaN).T.mean().values
    luftdruck = dataframe[['Luftdruck']].replace(0, np.NaN).T.mean()
    dataframe['ziel'] = ziel
    dataframe = dataframe[['ziel']]



    dataframe['Luftdruck'] = luftdruck


    timestamp_s = date_time.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    dataframe['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    dataframe['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    dataframe['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    dataframe['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    dataframe = dataframe.fillna(0)

    file1 = open('EA2_values.txt', "r")            
    
     
    mean_and_sdt = file1.read().split('\n')
    len(mean_and_sdt) // 2
    train_mean = [float(x.split(' ')[-1]) for x in mean_and_sdt[:len(mean_and_sdt) // 2]]
    train_std = [float(x.split(' ')[-1]) for x in mean_and_sdt[:len(mean_and_sdt) // 2]]
    file1.close()
    example = (dataframe - train_mean) / train_std


    output = pd.DataFrame(model.predict(np.array([tf.constant(example)]))[0])

    vorhersage = output * train_std[0] + train_mean[0]
    vorhersage2 = vorhersage.loc[0, 0]
    print(
        "Der vorhergesagte Feinstaubdurchschnittswert(PM10) in 6 Stunden im Raum Graz beträgt ",
        vorhersage2)


