import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
import numpy as np


def read_dat(filepath):
    return pd.read_table(filepath, sep="\t")


path = 'data\laphy_kohaerenz_interferometer\pd_schwelle_34_52mA_18_05kOhm.dat'
df = read_dat(path)
x = df['Time'].values
y = df['Amplitude'].values

fig = plt.figure()
plt.scatter(x, y)
plt.show()


print('Ich bin ein debug print statement zum breakpoint setzen')
