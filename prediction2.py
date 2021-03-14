import datetime
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas.io.json import json_normalize
from xlrd import open_workbook
from xlutils.copy import copy

#(r'\s+(+\.|#', np.NaN, regex = True).replace('', np.NaN)

def prediction2(dataframe, model, train_std, train_mean):

	date_time = pd.to_datetime(dataframe.pop('Zeit'), format='%d/%m/%Y %H:%M')
	dataframe.replace('', 0, inplace=True)
	df = dataframe.astype('float')


	# In[478]:


	df = df[['Graz-N']]


	# In[479]:


	timestamp_s = date_time.map(datetime.datetime.timestamp)
	day = 24*60*60
	year = (365.2425)*day

	df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
	df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
	df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
	df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


	# Shape muss gleich sein!

	# In[480]:


	df = df[-48:]


	# # Normieren

	# In[481]:


	df_norm = df


	# In[482]:


	#file1 = open("EA2_values.txt","r") 
	#mean_and_sdt = file1.read().split('\n')
	#len(mean_and_sdt)//2
	#train_mean = [float(x.split(' ')[-1]) for x in  mean_and_sdt[:len(mean_and_sdt)//2]]
	#train_std = [float(x.split(' ')[-1]) for x in  mean_and_sdt[len(mean_and_sdt)//2:]]
	#file1.close()


	# In[483]:


	df_norm = (df_norm - train_mean) / train_std


	# In[484]:


	df_norm = df_norm.interpolate(method='linear')


	# # Vorhersage

	# In[487]:




	# In[488]:


	vorhersage = pd.DataFrame(model.predict(np.array([tf.constant(df_norm)]))[0])


	# In[489]:


	vorhersage = vorhersage * train_std[0] + train_mean[0]


	# In[490]:


	vorhersage2 = vorhersage.loc[0, 0]
	
	print("Der vorhergesagte Feinstaubdurchschnittswert(PM10) in 6 Stunden im Raum Graz beträgt ", vorhersage2)






