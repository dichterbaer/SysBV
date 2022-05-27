from pickle import FALSE
from cv2 import repeat
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import N
from matplotlib.animation import FuncAnimation

def X_cos(t): #Cosinuswerte im Intervall
    Xout = np.cos(t)
    return  Xout


def K_fft(signal):
    w = np.arange(0,signal.shape[0],1)
    T, S = np.meshgrid(w, w)    
    M = signal.shape[0]   
    k_out = np.exp(-2*np.pi*1j*S*T/M) 
    return  k_out
#
# Aufgabe 1b)
k = np.array([1,4,6,4,1])*(1/16)
x = np.arange(0, 21, 1)
kk= np.transpose(np.zeros(x.size))

kk[0:k.size]=k
K = kk

#K[0:kk.size]= kk[0:kk.size]
for i in range(0, x.size-1): #create K with shifted k
    kk = np.roll(kk, 1)
    #K = np.append(K , k_temp)
    K = np.vstack((K, kk))

#apply K to x
yK = np.dot(K, np.transpose(x))

#vergleich np implementierung mit händisch (passt aber nicht)
yk_np = np.convolve(x, k)

yk = np.zeros(x.size)
# letzter wert passt noch nicht ganz
#apply k to x 
for i in range(0, x.size-1): 
    yk[i] = np.sum(np.dot(x[0:5], k)) 
    x = np.roll(x, -1)

lena = cv2.imread('data/lena.png', 0)
cv2.imshow('Original', lena)

lena_glatt = cv2.filter2D(lena, -1, k)
lena_glatt_cv = cv2.GaussianBlur(lena, (1,5), 0)
diff = lena_glatt_cv - lena_glatt 
cv2.imshow('Gefiltert Händisch', lena_glatt)
cv2.imshow('Gefiltert OpenCV', lena_glatt_cv)
cv2.imshow('Differenz Händisch vs OpenCV', diff)
cv2.waitKey(0)

#Aufgabe 2a) animiert (ist noch nicht richtig. sollte keine matrixmultiplikation sein sondern ein diskretisierter kernel)

fig, (ax1, ax2, ax3) = plt.subplots(3,1)
fig.set_size_inches(6, 10)
ax1.set_xlim([0,200*np.pi+1])
ax1.set_ylim([-1.5 ,1.5])
ax1.title.set_text('Input Signal')
ax2.set_xlim([0,200*np.pi+1])
ax2.set_ylim([-10 ,10])
ax2.title.set_text('Fourier händisch')
ax3.set_xlim([0,200*np.pi+1])
ax3.set_ylim([-10 ,10])
ax3.title.set_text('Fourier mit np.fft')
line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2)
line3, = ax3.plot([], [], lw=2)
title = ax1.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax1.transAxes, ha="center")

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3


def animate(i):
    n = 10**i
    n = i
    x = np.linspace(0, 200*np.pi, n)
    cos = X_cos(x)  
    f_h = np.matmul(K_fft(x), X_cos(x))
    f_fft = np.fft.fft(cos) 
    line1.set_data(x, cos)
    line2.set_data(x, f_h)
    line3.set_data(x, f_fft)
    title.set_text("n = " + str(n))
    return line1, line2, line3, title,

inputs = np.arange(10, 500, 5) #Abtastpunkte (10-500 in 10er Schritten)
interval = 100 #pause zwischen frames in ms

anim = FuncAnimation(fig, animate, init_func=init, frames=inputs, interval=interval, blit=True)
plt.show()
#Aufgabe 2a)
# plt.figure(1)
# plt.subplot(211)
# plt.title('Händisch')
# plt.plot(x,f_h,'r')
# plt.subplot(212)
# plt.title('np.fft.ftt(signal)')
# plt.plot(x,f_fft,'g')
# plt.show()

print()


