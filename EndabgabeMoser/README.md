# Trainingsumgebung PMP

In diesem Ordner befinden sich

- XLSX mit Feinstaubwerten und Wetterdaten der Stadt Graz (Erhalten vom Land Stmk)
- Jupyter Notebooks Endabgabe
  - EA2: Visualisierung und minimale Beschreibung der Daten vom Land Stmk
  - EA2-verarbeiten: Auffüllen, Engineering und Analyse der Daten; Trainieren der Modelle.
- Python script zum Ausführen des Modells in der Runtime
- Ordner Anhang, mit einigen der Notebooks von der Entwicklung.
  - Diese sind nicht ohne weiteres ausführbar, da sie die Entwicklungs-DB verwenden, aber geben einen Einblick in die Herangehensweise.

Erklärungen, warum die verfolgten Ansätze gewählt wurden, sowie eine Interpretation der Ergebnisse sowie ein kurzes Schlusswort finden sich im EA2-verarbeiten Notebook.

Getestet wurde die Umgebung wie beschrieben auf einem Ubuntu Server.

## Installierte Software Pakete

Mit dieser Software installiert und den .xlsx Dateien im selben Ordner wie die Jupyter Notebooks lassen sich diese ausführen.

per apt

```
libhdf5-serial-dev
graphviz
python3-opencv
vim
tmux
htop
ranger
docker-ce
mongodb
nodejs
npm
```

im python venv per pip3

```
numpy==1.18.0
scipy
matplotlib
h5py==2.10.0
pydot-ng
tensorflow-cpu
jupyterlab
pandas
pymongo
xlrd
seaborn
```

## Ursprüngliche Entwicklungsumgebung

Alle Herangehensweisen, die wir gegangen sind um ein Vorhersagemodell zu entwickeln befinden sich auf dem FH-Server, erreichbar unter:

- Server IP: 10.25.2.103
- Jupyter Lab Port: 8888
- Jupyter Lab Passwort: pmf

Läuft das jupyter-lab nicht, kann dieses durch aktivieren des virtual environments (~/MachineLearning/venvs/cpu) und ausführen von `jupyter-lab` gestartet werden.

## Erklärung zur ersten Abgabe

Diese Abgabe soll die Missverständnisse der ersten Abgabe ausbessern. Daher sind hier nicht nur der Code, der zum entgültigen Modell führt, sondern auch die Schritte dazwischen inkludiert.

In der ersten Abgabe wurde die Mittelung der PM10 Werte von mehreren Stationen vorgenommen, weil es eine schnelle und verlässliche Art der Bereinigung der Daten darstellen sollte (Angleichung von Ausreißern / Auffüllen von fehlenden Werten). Da es sich bei den vorliegenden Daten laut Land Steiermark um Halbstunden-Mittelwerte und daher ohnehin um eine "virtuelle Größe" handelt, schien eine weitere Errechnung von Durchschnittswerten als unproblematisch.

Die Ersetzung von fehlenden Werten durch Nullen, wurde durch das Buch *Deep Learning with Python* von FRANÇOIS CHOLLET auf Seite 102 beeinflusst: "In general, with neural networks, it’s safe to input missing values as 0, with the condition that 0 isn’t already a meaningful value. The network will learn from exposure to the data that the value 0 means missing data and will start ignoring the value."
Dabei wurde darauf geachtet, dass dies nach der Normalisierung der Features erfolgt.

Die Auswahl des Luftdruck als Input Feature wurde, wie schon in Besprechungen erwähnt, getroffen, weil das Modell das beste Ergebnis erzielte. Auch wenn die Unterschiede in der Leistung nicht signifikant sind, wollten wir damit zeigen, wie ein Modell mit zusätzlichen Features zu trainieren ist.
