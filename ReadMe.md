# Particulate Matter Forecast

___

## Contents:

EA2-Modell --> Machine Learning Modell Binaries
EndabgabeMoser --> Jupyter Notebooks Moser² 
Kafka_Cluster --> Apache Kafka Configuration file with Zookeeper, Kafka Broker and Landoop Web UI
Old_Model --> Old ML Model and prediction python script
Screenshots
consumerfinal.py --> Kafka Python consumer, which reads Data from a Kafka Topic and produces predictions with the ML model
producer.py --> Kafka Python producer, produces Sensordata and sends it to a Kafka Topic
preditions2.py --> function, uses the EA2-Modell and the trained mean values "EA2_values" to make a prediction on the given dataframe
EA2_values --> Trained mean values
jsonFinal.json --> Sensor data in Json format



## FH-Server Testingenvironment:

1. Connect via RDP 10.25.6.5 (VPN requiered)
2. Username: Administrator / Password: Pa55w.rd
3. Select Linux VM Test02
4. Username: test / Password: 1234


## System Requirements:

1. CPU with AVX/AVX2 support
2. GPU with CUDA support beneficial
3. Linux operating system with GUI (Ubuntu 20.04.2 Desktop)
4. 8GB of RAM
5. Internet connection

___

## Configuration:

### Install Docker

Update and Upgrade the apt package index and install packages to allow apt to use a repository over HTTPS:

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

Add Docker’s official GPG key:

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

Use the following command to set up the stable repository.

```
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt update
sudo apt upgrade
apt-cache policy docker-ce
```

Install Docker (Docker version 20.10.3):

```
sudo apt install docker-ce
```

Check if Docker installation worked: "Status Active"

```
sudo systemctl status docker
```

Run the Docker test image to verify installation (might need root permissions)

```
sudo docker run hello-world
```


### Set up the Project

Download or clone this repository. Do not move or rename any directories/files, this could lead to errors.

`git clone https://github.com/MrObergeil/PMF_FINAL.git`

Install Docker Compose (version 1.26.0)
```
sudo curl -L "https://github.com/docker/compose/releases/download/1.26.0/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose

```

Set user permissions

```
sudo chmod +x /usr/local/bin/docker-compose
```

Verify installation

```
docker-compose --version
```


#### Install dependencies for the ML-model

Install Python and additional requirements needed to run the ML-model and Python Kafka Consumer

Check Python installation with:

```
python3 --version
```

Install Python if its not installed or is an older version than 3.8

```
sudo apt-get update
sudo apt-get install python 3.8
```

Install pip3 ( version 20.0.2):

```
sudo apt-get install python3-pip
```

Check pip version:

```
pip3 --version
```

Install python virtual environment

```
sudo apt-get install python3-venv
```

Create a python virtual environment (PMF-venv) in the main directory of the repository (../PMF_FINAL)

```
cd PMF_FINAL

python3 -m venv PMF-venv
```

Activate the virtual environment

```
source PMF-venv/bin/activate
```

Within the active environment we can install the required pip3 packages. You can check the versions of the packages with "pip show nameofpackage"

```
pip show "seaborn/tensorflow-cpu....".
```

This also shows you any required other pip3 packages for each individual package (should be installed by pip automatically, if not manually add them using the pip3 install command.)

Install requred python packages

- Seaborn (version 0.11.1)
- Tensorflow (version 2.4.1)
- Kafka-Python (version 2.0.2)
- Xlrd (version 2.0.1)
- Xlutils (version 2.0.0)

```
pip3 install numpy==1.19.2 seaborn==0.11.1 tensorflow-cpu==2.4.1 kafka-python==2.0.2 xlrd==2.0.1 xlutils==2.0.0
```

In your terminal switch to the folder "Kafka_Cluster", inside the "PMF_FINAL" repository. This folder  contains the docker-compose.yml file, with the configurations for the Kafka Cluster.

```
cd home/<user>/<downloadLocation>/PMF_FINAL/Kafka_Cluster
```

Start Docker Container using docker-compose on the Yaml-File, the Kafka Cluster is declared as "kafka-cluster" in the Yaml-File.

```
sudo docker-compose up kafka-cluster

```

This will download the required Docker-Image and start your Apache Kafka Cluster.

Wait 1-2 min for the cluster to start up.

Access LensesIO-Webinterface via `127.0.0.1:3030` to monitor and check the status of your Kafka Cluster

Your Kafka Cluster should now be ready to use

Switch to a new terminal and leave the terminal which runs the cluster running in the background
---

## Launch the Kafka Producer and the Kafka Consumer to start making predictions:

We are now set to start the Kafka Producer to write messages to our topic. The topic name is specified in the producer.py file as well as the path to the json file containing the data. Currently the producer is producing to a topic "JSONFINA" (line 31 producer.send('JSONFINA'. data[index])). It is also possible to change the waiting time between messages at line 33 sleep(2), 2 means it is currently set to wait 2 seconds between every new message.
Start the producer located in the PMF_FINAL directory from within the virtual environment:

```
cd /PMF_FINAL
source PMF-venv/bin/activate
python3 producer.py
```

To confirm that the producer is working as intended confirm that your topic has been created in Apache Kafka using the webinterface at `127.0.0.1:3030/kafka-topics-ui/`

The next step is to start our consumer who will consume from our topic "JSONFINA" and make predictions.
Start the consumer by opening a new terminal and starting the consumer.py from within the virtual environment.

Lets activate our virtual environment in the PMF_FINAL-main directory and start the Kafka Consumer.

```
cd /PMF_FINAL
source PMF-venv/bin/activate
python3 consumerfinal.py
```

The consumer is currently set up to make 1600 predictions and makes one prediction every time new message(Sensordata) arrives. This can be changed in the pythonscript "consumerfinal.py" at line 53  "prediction_count==1600".
The consumer initally waits for 48 messages (24h of data) to make the first prediction, then every time a new message arrives he makes a new prediction incoperating that new data.
