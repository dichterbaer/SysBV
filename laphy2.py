
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
import numpy as np


sheets = [0, 1, 2, 3, 4, 5, 6]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
labels = ['20°C', '25°C', '30°C', '35°C', '40°C', '45°C', '50°C']
temps = [20, 25, 30, 35, 40, 45, 50]
fpath : str = './data/laphy_02.xlsx'
df = pd.read_excel(fpath, sheet_name=sheets)
model = LinearRegression()
intercepts = []
fig, ax = plt.subplots()
ax.set_xlim([0, 2500])
ax.set_ylim([801, 812])
for sheet in sheets:
    if(sheet==6):
        x = df[sheet]['Waste Thermal Power (mW)'].values
        x2 = x[1:]
        y = df[sheet]['Wellenlänge (nm)'].values
        y2 = y[1:]
        x_fit = np.array([0,2500])
        model.fit(x2.reshape(-1, 1), y2)
        intercepts.append(model.intercept_)
        text = ax.text(2325, y[y.shape[0]-1], labels[sheet])
        ax.add_patch(Ellipse((x[0], y[0]), width=80, height = 0.3, color = 'r', fill = False))
        ax.plot(x, y, '*', color=colors[sheet], label=labels[sheet])
        ax.plot(x_fit, model.predict(x_fit.reshape(-1, 1)), '-.', color=colors[sheet])
    else:    
        x = df[sheet]['Waste Thermal Power (mW)'].values
        y = df[sheet]['Wellenlänge (nm)'].values
        x_fit = np.array([0,2500])
        model.fit(x.reshape(-1, 1), y)
        intercepts.append(model.intercept_)
        text = ax.text(2325, y[y.shape[0]-1], labels[sheet])
        ax.plot(x, y, '*', color=colors[sheet], label=labels[sheet])
        ax.plot(x_fit, model.predict(x_fit.reshape(-1, 1)), '-.', color=colors[sheet])

ax.set_title("Power-Averaged Wavelength vs Waste Thermal Power")
ax.set_yticks(np.arange(800, 813, 1))
ax.set_xlabel('Waste Thermal Power / mW')
ax.set_ylabel('Wavelength / nm')
ax.grid(True, which='both', axis='y')

plt.show()

# sheets = [0, 2, 4, 5]
# fig, ax = plt.subplots()
# ax.set_xlim([0, 2500])
# ax.set_ylim([800, 813])
# for sheet in sheets:
#     if(sheet==6):
#         x = df[sheet]['Waste Thermal Power (mW)'].values
#         x2 = x[1:]
#         y = df[sheet]['Wellenlänge (nm)'].values
#         y2 = y[1:]
#         x_fit = np.array([0,2500])
#         model.fit(x2.reshape(-1, 1), y2)
#         intercepts.append(model.intercept_)
#         text = ax.text(2325, y[y.shape[0]-1], labels[sheet])
#         ax.add_patch(Ellipse((x[0], y[0]), width=80, height = 0.3, color = 'r', fill = False))
#         ax.plot(x, y, '*', color=colors[sheet], label=labels[sheet])
#         ax.plot(x_fit, model.predict(x_fit.reshape(-1, 1)), '-.', color=colors[sheet])
#     else:    
#         x = df[sheet]['Waste Thermal Power (mW)'].values
#         y = df[sheet]['Wellenlänge (nm)'].values
#         x_fit = np.array([0,2500])
#         model.fit(x.reshape(-1, 1), y)
#         intercepts.append(model.intercept_)
#         text = ax.text(2325, y[y.shape[0]-1], labels[sheet])
#         ax.plot(x, y, '*', color=colors[sheet], label=labels[sheet])
#         ax.plot(x_fit, model.predict(x_fit.reshape(-1, 1)), '-.', color=colors[sheet])

# ax.set_title("Power-Averaged Wavelength vs Waste Thermal Power")
# ax.set_yticks(np.arange(800, 813, 1))
# ax.set_xlabel('Waste Thermal Power / mW')
# ax.set_ylabel('Wavelength / nm')
# ax.grid(True, which='both', axis='y')

# plt.show()


fix, ax = plt.subplots()
intercepts = np.array(intercepts)
temps = np.array(temps)
model.fit(temps.reshape(-1, 1), intercepts)
t = "$\lambda$ = " + str(round(model.coef_[0], 3)) + "$ nm/°C * T_{j} +" + str(round(model.intercept_, 3)) + " nm$"
plt.plot(temps, intercepts, '*')
plt.plot(temps, model.predict(temps.reshape(-1, 1)), '-.')
ax.set_xlabel('Junction Temperature / C')
ax.set_ylabel('Wavelength / nm')
ax.grid(True, which='both', axis='both')
text = ax.text(x=22, y=809, s=t, fontsize=10)
plt.show()

print()