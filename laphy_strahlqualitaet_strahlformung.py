import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
import numpy as np

#Aufgabenteil 3
fpath : str = 'data/laphy_strahlqualitaet_strahlformung/3/Export_2022-06-07_09.38.58_#001_Data.xls'
df = pd.read_excel(fpath, sheet_name='results')

stagePos = df['StagePosition'].values
x1 = df['BeamWidth4SigmaX_um'].values
y2 = df['BeamWidth4SigmaY_um'].values

fig, ax = plt.subplots()
ax.plot(stagePos, x1, 'x')
ax.set_xlabel('Stage Position / mm')
ax.set_ylabel('Beam With X / um')
ax.grid(True, which='both', axis='both')
#plt.show()

#Aufgabenteil 7 nicht sicher ob wir die Plots üverhaupt brauchen
fpath : str = 'data/laphy_strahlqualitaet_strahlformung/3/Export_2022-06-07_09.38.58_#001_Data.xls'
df = pd.read_excel(fpath, sheet_name='results')

stagePos = df['StagePosition'].values
x1 = df['BeamWidth4SigmaX_um'].values
y2 = df['BeamWidth4SigmaY_um'].values

fig, ax = plt.subplots()
ax.plot(stagePos, x1, 'x')
ax.set_xlabel('Stage Position / mm')
ax.set_ylabel('Beam With X / um')
ax.grid(True, which='both', axis='both')
#plt.show()

#Aufgabenteil 7 M2
fpath : str = 'data/laphy_strahlqualitaet_strahlformung/7/Export_2022-06-07_10.36.37_#003_Results_lesbar.csv'

df = pd.read_csv(fpath, sep=';')
stagePos = df['Z'].values
x = df['X\''].values
y = df['Y\''].values

fig, ax = plt.subplots()
ax.plot(stagePos, x, 'x')
ax.set_xlabel('Stage Position / mm')
ax.set_ylabel('Beam With X / um')
ax.grid(True, which='both', axis='both')
plt.show()
print('')