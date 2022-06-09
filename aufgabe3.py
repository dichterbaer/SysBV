import numpy as np
import cv2

def create_cos(N, alpha, beta, gamma):
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    return np.cos(2*np.pi*(X*alpha + Y*beta + gamma))



a = 1/0.2*np.cos(np.pi/6)
b = 1/0.2*np.sin(np.pi/6)
c = 0

#Aufgabe 1b)
cos_b = create_cos(256, a, b, c)
cv2.imshow('bill cos_b', cos_b)
cv2.waitKey()
cv2.destroyAllWindows()

#Aufgabe 1c)
a = 8
b = 4 
c = 0.25
cos_c = create_cos(256, a, b, c)
cv2.imshow('cos_c', cos_c)
cv2.waitKey()
cv2.destroyAllWindows()

#Aufgabe 1d)
a = 128*np.cos(np.pi)
b = 128*np.sin(np.pi)
a = 128
b = 128
c = 0
cos_d = create_cos(256, a, b, c)
fft_d = np.imag(np.fft.fft2(cos_d))
cv2.imshow('cos_d', cos_d)
cv2.imshow('fft_d', fft_d)
cv2.waitKey()
cv2.destroyAllWindows()