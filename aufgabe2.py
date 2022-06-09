import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import N
from matplotlib.animation import FuncAnimation

def K_fft(x):
    w = np.arange(0,x,1)
    T, S = np.meshgrid(w, w) 
    k_out = np.exp(-2*np.pi*1j*S*T/x) 
    return  k_out


def cosine_wave(f,overSampRate,phase,nCyl):
	"""
	Generate sine wave signal with the following parameters
	Parameters:
		f : frequency of sine wave in Hertz
		overSampRate : oversampling rate (integer)
		phase : desired phase shift in radians
		nCyl : number of cycles of sine wave to generate
	Returns:
		(t,g) : time base (t) and the signal g(t) as tuple
	Example:
		f=10; overSampRate=30;
		phase = 1/3*np.pi;nCyl = 5;
		(t,g) = sine_wave(f,overSampRate,phase,nCyl)
	"""
	fs = overSampRate*f # sampling frequency
	t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base
	g = np.cos(2*np.pi*f*t+phase) # replace with cos if a cosine wave is desired
	return (t,g) # return time base and signal g(t) as tuple


while(1):   
    frq = float(input("Enter Frequency: "))
    oversampling = float(input("Enter oversampling rate: "))
    nCyl = float(input("Enter number of cycles: "))
    fs = frq*oversampling
    shift_factor = 0
    phase = shift_factor*np.pi
    (t,x) = cosine_wave(frq, oversampling, phase, nCyl)
    plot_x_vals = np.arange(0, x.size, 1)
    x_shift = (np.arange(start = -x.size/2,stop = x.size/2))*fs/x.size  #DFT Sample points  
    #x_shift = np.arange(start = -x.size/2,stop = x.size/2) + 0.5 #DFT Sample points  
    f_h = np.matmul(K_fft(x.size), x)
    f_shift = np.fft.fftshift(f_h)
    f_fft = np.fft.fft(x)
    fig, axs = plt.subplots(4)
    fig.set_size_inches(6, 10)
    axs[0].plot(t, x)
    axs[0].set_title('Input Cosine with Frq: ' + str(frq))
    axs[0].set_xlabel('Time /s')
    axs[0].set_ylabel('Amplitude')
    axs[1].plot(plot_x_vals, f_h)
    axs[1].set_title('DFT without np')
    axs[1].set_xlabel('Sample points')
    axs[1].set_ylabel('Dft value')
    axs[2].plot(plot_x_vals, f_fft)
    axs[2].set_title('DFT with np.fft')
    axs[2].set_xlabel('Sample points')
    axs[2].set_ylabel('Dft value')
    axs[3].plot(x_shift, f_shift)   
    axs[3].set_title('Shifted version of Plot2')
    axs[3].set_xlabel('Frequency /Hz')
    axs[3].set_ylabel('Dft value')
    #add margin between subplots
    fig.subplots_adjust(hspace=0.7)
    plt.show()


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

#vergleich np implementierung mit händisch (randbehandlung)
yk_np = np.convolve(x, k, mode='same')

yk = np.zeros(x.size)
# letzter wert passt noch nicht ganz
#apply k to x 
for i in range(0, x.size-1): 
    yk[i] = np.sum(np.dot(x[0:5], k)) 
    x = np.roll(x, -1)


print("Aufgabe 1b)")
print("yK: " + str(yK)) 
print("yk: " + str(yk)) 
print("yk_np: " + str(yk_np)) 


# Aufgabe 1c)

lena = cv2.imread('data/lena.png', 0)
cv2.imshow('Original', lena)

lena_glatt = cv2.filter2D(lena, -1, k)
lena_glatt_cv = cv2.GaussianBlur(lena, (1,5), 0)
diff = lena_glatt_cv - lena_glatt 
cv2.imshow('Gefiltert Händisch', lena_glatt)
cv2.imshow('Gefiltert OpenCV', lena_glatt_cv)
cv2.imshow('Differenz Händisch vs OpenCV', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Aufgabe 2a) animiert 

# interval_end = 2

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
# fig.set_size_inches(6, 10)
# ax1.set_xlim([0,interval_end])
# ax1.set_ylim([-1.5 ,1.5])
# ax1.title.set_text('Input Cosinus')
# ax2.set_xlim([0,interval_end])
# ax2.set_ylim([-10 ,10])
# ax2.title.set_text('Fourier händisch')
# ax3.set_xlim([0,interval_end])
# ax3.set_ylim([-10 ,10])
# ax3.title.set_text('Fourier mit np.fft')
# ax4.set_xlim([0,interval_end])
# ax4.set_ylim([-10 ,10])
# ax4.title.set_text('Differenz Fourier Händisch - np.fft')
# ax4.title.set_text('Fourier Händisch 2')
# line1, = ax1.plot([], [], lw=2)
# line2, = ax2.plot([], [], lw=2)
# line3, = ax3.plot([], [], lw=2)
# line4, = ax4.plot([], [], lw=2)
# title = ax1.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
#                 transform=ax1.transAxes, ha="center")

# def init():
#     line1.set_data([], [])
#     line2.set_data([], [])
#     line3.set_data([], [])
#     line4.set_data([], [])
#     return line1, line2, line3, line4


# def animate(i):
#     #N = i
#     #n = np.arange(N)
#     #k = 10
#     #x = np.cos(2 * np.pi * (k * n / N) + 2 * (np.random.rand(N) - 0.5)) 
#     #n = i
#     # x = np.linspace(0, interval_end*np.pi, 50)
#     # x = np.linspace(0, interval_end*np.pi, n)
#     #cos = X_cos(x)  
#     #f_h = np.convolve(K_fft(x), X_cos(x))
#     x, cos = gen_cosine_signal(2, i)
#     f_h = np.matmul(K_fft(i), cos)
#     #f_fft = np.fft.fft(cos) 
#     #f_h2 = np.abs(dft(cos))
#     #f_diff = f_h - f_fft
#     line1.set_data(x, cos)
#     line2.set_data(x, f_h)
#     #line3.set_data(x, f_fft)
#     #line4.set_data(x, f_h2)
#     title.set_text("Abtastpunkte: " + str(i))
#     return line1, line2, line3, line4, title,

# inputs = np.arange(10, 500, 5) #Abtastpunkte (10-500 in 10er Schritten)
# interval = 500 #pause zwischen frames in ms

# anim = FuncAnimation(fig, animate, init_func=init, frames=inputs, interval=interval, blit=True)
# plt.show()
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


