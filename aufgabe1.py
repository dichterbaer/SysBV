
import numpy as np
import matplotlib.pyplot as plt

#Functions

def K_exp(t,s): #Kernel Problem 1
    dt = t[2] - t[1]
    mask = np.zeros(shape = (t.shape[0], t.shape[0]))           # das
    k_out = np.zeros(shape = (t.shape[0], t.shape[0]))          # geht
    T, S = np.meshgrid(t,s)                                     # bestimmt
    for x in range(T.shape[0]):                                 # auch   
        for y in range(S.shape[0]):                             # in 
            if T[x,y]>0 and T[x,y]<S[x,y]:                      # zwei
                mask[x,y] = 1                                   # Zeilen
                k_out[x,y] = dt*np.exp(S[x,y]*(T[x,y]-S[x,y]))  # wenn man am ende merkt, dass man mask gar nicht benutzt aber es trotzdem funktioniert(┛ಠ_ಠ)┛彡┻━┻
    return  k_out


#Inputfunktionen
def X(t): #Sprungfunktion
    Xout = (t >= 0)
    return  Xout


def X_sin(t): #Sinuswerte im Intervall
    Xout = np.sin(t)
    return  Xout


#Problem 2
def K_sin(t,s): #Kernel von Problem 2
    dt = t(2) - t(1)
    for S in s:
        for T in t:
            if T>0 & T<S:
                print(T)
    return  dt*np.sin(np.pi*(S - T))   


#Problem Sheet 1

N=40 # Abtastung
#N=400;
tau = 1

#Problem 1
a1=-2 # Interval
b1=10 # Interval

s1=np.linspace(start=a1, stop=b1, num=N)
t1=np.linspace(start=a1, stop=b1, num=N)

#Problem 2
a2=0 # Interval
b2=12 # Interval

s2=np.linspace(start=a1, stop=b1, num=N)
t2=np.linspace(start=a1, stop=b1, num=N)


#Problem 1
#y1=K_exp(t1,s1)*X(t1) #Funktion ohne shift mit Jump-Input
y1 = np.matmul(K_exp(t1,s1), X(t1)) #Funktion ohne shift mit Jump-Input
#y1_sin=K_exp(t1,s1)*X_sin(t1) #Funktion ohne shift mit Sinus-Input
y1_sin = np.matmul(K_exp(t1,s1), X_sin(t1)) #Funktion ohne shift mit Sinus-Input
#y1_tau=K_exp(t1,s1)*X(t1-tau) #Funktion mit shift mit Jump-Input
y1_tau=np.matmul(K_exp(t1,s1), X(t1-tau)) #Funktion mit shift mit Jump-Input
#y1_sin_tau=K_exp(t1,s1)*X_sin(t1-tau) #Funktion mit shift mit Sinus-Input
y1_sin_tau=np.matmul(K_exp(t1,s1), X_sin(t1-tau)) #Funktion mit shift mit Sinus-Input

y1_m_sin=1/(s+1/s)*(sin(s)-cos(s)/s-exp(-s^2)); #händisch bestimmtes Integral vom Kernel mit Sinus Funktion als Input
y1_mv_sin=eval(subs(y1_m_sin, s, s1)); #Bestimme Funktionswerte im Intervall s1


#Problem 1
plt.figure()

plt.subplot(411)
plt.plot(s1,y1,'r')
plt.plot(s1,X(t1),'g')
plt.plot(s1, y1_mv,'b')
plt.legend('Output Numerical','Input','Output Analytical')
plt.ylabel('y(s)')
plt.xlabel('s')
plt.title('Problem 1: jump')


plt.subplot(412)
plt.plot(s1,y1_tau,'r')
plt.plot(s1,X(t1-tau),'g')
plt.ylabel('y(s)')
plt.xlabel('s')
plt.title('Problem 1: jump verschoben')


plt.subplot(413)
plt.plot(s1,y1_sin,'r')
plt.plot(s1,X_sin(t1),'g')
plt.plot(s1, y1_mv_sin,'b')
plt.legend('Output Numerical','Input','Output Analytical')
plt.ylabel('y(s)')
plt.xlabel('s')
plt.title('Problem 1: oscillatory')


plt.subplot(414)
plt.plot(s1,y1_sin_tau,'r')
plt.plot(s1,X_sin(t1-tau),'g')
plt.ylabel('y(s)')
plt.xlabel('s')
plt.title('Problem 1: oscillatory shifted')
plt.show

