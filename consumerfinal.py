from kafka import KafkaConsumer
from kafka import TopicPartition
import os
import datetime

from prediction2 import prediction2 as prediction
import pandas as pd
import tensorflow as tf
import sys
import json

#load the model
model = tf.keras.models.load_model('EA2-Modell', custom_objects=None, compile=True, options=None)

#load train values
file1 = open("EA2_values.txt","r") 
mean_and_sdt = file1.read().split('\n')
len(mean_and_sdt)//2
train_mean = [float(x.split(' ')[-1]) for x in  mean_and_sdt[:len(mean_and_sdt)//2]]
train_std = [float(x.split(' ')[-1]) for x in  mean_and_sdt[len(mean_and_sdt)//2:]]
file1.close()

consumer = KafkaConsumer(
    'JSONFINAL',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id=None,
    auto_offset_reset='earliest')

consumer.subscribe('JSONFINA')
print("Consuming messages from Topic")
message = consumer.poll()

rowx=0
colx=0

#consumer.close(autocommit=True);

columns=['Zeit', 'Graz-DB', 'Graz-MG', 'Graz-S', 'Graz-OP', 'Graz-N', 'Graz-W', 'Lustb', 'Luftdruck']
#Empty Dataframe mit Index
dfAll = pd.DataFrame(columns=columns)


prediction_count = 0
message_count = 0
for message in consumer:

    df = pd.DataFrame([message.value])
    dfAll=dfAll.append(df, ignore_index=True)
    message_count += 1

    if message_count == 48:
        dfPrediction = dfAll.copy()
        prediction(dfPrediction, model, train_std, train_mean)
        prediction_count += 1
        print('Success for prediction #', prediction_count)
        dfAll.drop(index=0, axis=0, inplace=True)
        dfAll.reset_index(drop=True, inplace=True)
        message_count = 47
        if prediction_count == 1600:
            break




