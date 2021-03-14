from kafka import KafkaProducer
from time import sleep
from json import dumps
import json
import pandas as pd


import csv
import time

#producer config
producer = KafkaProducer(bootstrap_servers=
                         ['localhost:9092'], value_serializer=lambda v: json.dumps(v).encode('utf-8'))


# Load from Json

# with open (filename) as f:
#     data = json.load(f)
#     producer.send_message(topic, data.encode ('utf-8'))

#producer final

with open('jsonFinal.json') as f:
    data = json.load(f)
    index = 0
    for i in data:
        if index == 80000:
            break
        print(data[index])
        producer.send('JSONFINA', data[index])
        index += 1
        sleep(2)


#hardcoded solution scuffed

#producer.send('JSO2', {'Zeit': '1/1/15 0:30', 'DB': '98.4', 'Graz-M': '91.4', 'Graz-S': '88.4', 'Graz-O': '68.4',
 #                      'Graz-N': '28.4', 'Graz-W': '92.4', 'Lustb': '68.4'})

print("Message Sent to JSONtopic")

producer.flush(timeout=10)
producer.close(timeout=5)
