import numpy as np
import cv2

def create_cos(N, alpha, beta, gamma):
    x = np.linspace(0, 1, N+1)
    x = x[:-2]
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


#Aufgabe 2a)

img = cv2.imread('data/how60c.tif', 0)
img_fft = np.fft.fft2(img)
img_fft_shift = np.fft.fftshift(img_fft)
cv2.imshow('Original', img)
cv2.imshow('Fourier', np.abs(img_fft))
cv2.imshow('Fourier shift', np.abs(img_fft_shift))
cv2.waitKey()
cv2.destroyAllWindows()