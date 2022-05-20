import numpy as np
import cv2


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
diff = lena_glatt - lena_glatt_cv
cv2.imshow('Gefiltert Händisch', lena_glatt)
cv2.imshow('Gefiltert OpenCV', lena_glatt_cv)
cv2.imshow('Differenz Händisch vs OpenCV', diff)
cv2.waitKey(0)

print()

