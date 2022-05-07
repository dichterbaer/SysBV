
from cmath import tau
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
    dt = t[2] - t[1]       
    k_out = np.zeros(shape = (t.shape[0], t.shape[0]))        
    T, S = np.meshgrid(t,s)                                   
    for x in range(T.shape[0]):                               
        for y in range(S.shape[0]):                           
            if T[x,y]>0 and T[x,y]<S[x,y]:                                 
                k_out[x,y] = dt*np.sin(np.pi*(S[x,y]-T[x,y]))
    return  k_out 

def Problem_one(N):
    tau = 1

    #Problem 1
    a1=-2 # Interval
    b1=10 # Interval

    s1=np.linspace(start=a1, stop=b1, num=N)
    t1=np.linspace(start=a1, stop=b1, num=N)

    #Problem 1
    #y1=K_exp(t1,s1)*X(t1) #Funktion ohne shift mit Jump-Input
    y1 = np.matmul(K_exp(t1,s1), X(t1)) #Funktion ohne shift mit Jump-Input
    #y1_sin=K_exp(t1,s1)*X_sin(t1) #Funktion ohne shift mit Sinus-Input
    y1_sin = np.matmul(K_exp(t1,s1), X_sin(t1)) #Funktion ohne shift mit Sinus-Input
    #y1_tau=K_exp(t1,s1)*X(t1-tau) #Funktion mit shift mit Jump-Input
    y1_shifted=np.matmul(K_exp(t1,s1), X(t1-tau)) #Funktion mit shift mit Jump-Input
    #y1_sin_tau=K_exp(t1,s1)*X_sin(t1-tau) #Funktion mit shift mit Sinus-Input
    y1_sin_shifted=np.matmul(K_exp(t1,s1), X_sin(t1-tau)) #Funktion mit shift mit Sinus-Input

    #manual computation 
    # kp wie man das syms s aus matlab umsetzt
    #y1_m_sin=1/(s+1/s)*(sin(s)-cos(s)/s-exp(-s^2)); #händisch bestimmtes Integral vom Kernel mit Sinus Funktion als Input
    #y1_mv_sin=eval(subs(y1_m_sin, s, s1)); #Bestimme Funktionswerte im Intervall s1
    figNr = int(str(1)+str(N))
    plot = plt.figure(figNr)
    plt.subplot(411)
    plt.plot(s1,y1,'r')
    plt.plot(s1,X(t1),'g')
    #plt.plot(s1, y1_mv,'b')
    #plt.legend('Output Numerical','Input')
    #plt.legend('Output Numerical','Input','Output Analytical')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: jump')


    plt.subplot(412)
    plt.plot(s1,y1_shifted,'r')
    plt.plot(s1,X(t1-tau),'g')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: jump verschoben')
    plt.subplot(413)
    plt.plot(s1,y1_sin,'r')
    plt.plot(s1,X_sin(t1),'g')
    # plt.plot(s1, y1_mv_sin,'b')
    # plt.legend('Output Numerical','Input','Output Analytical')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: oscillatory')


    plt.subplot(414)
    plt.plot(s1,y1_sin_shifted,'r')
    plt.plot(s1,X_sin(t1-tau),'g')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: oscillatory shifted')
    #plt.show
    return plot


def Problem_two(N):
    a2=0 # Interval
    b2=12 # Interval
    tau = 1

    s2=np.linspace(start=a2, stop=b2, num=N)
    t2=np.linspace(start=a2, stop=b2, num=N)

    y2 = np.matmul(K_sin(t2, s2), X(t2))
    y2_sin = np.matmul(K_sin(t2, s2), X_sin(t2))
    y2_shifted = np.matmul(K_sin(t2, s2), X(t2-tau))
    y2_sin_shifted = np.matmul(K_sin(t2, s2), X_sin(t2-tau))

    #manual computation

    figNr = int(str(2)+str(N))
    plot = plt.figure(figNr)
    plt.subplot(511)
    plt.plot(s2,y2,'r')
    plt.plot(s2,X(t2),'g')
    #plt.plot(s1, y1_mv,'b')
    #plt.legend('Output Numerical','Input')
    #plt.legend('Output Numerical','Input','Output Analytical')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: jump')


    plt.subplot(512)
    plt.plot(s2,y2_shifted,'r')
    plt.plot(s2,X(t2-tau),'g')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: jump verschoben')
    plt.subplot(513)
    plt.plot(s2,y2_sin,'r')
    plt.plot(s2,X_sin(t2),'g')
    # plt.plot(s1, y1_mv_sin,'b')
    # plt.legend('Output Numerical','Input','Output Analytical')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: oscillatory')


    plt.subplot(514)
    plt.plot(s2,y2_sin_shifted,'r')
    plt.plot(s2,X_sin(t2-tau),'g')
    plt.ylabel('y(s)')
    plt.xlabel('s')
    plt.title('Problem 1: oscillatory shifted')
    #plt.show
    return plot

#Problem Sheet 1
plot1_40 = Problem_one(N=40)
plot1_40.show()
plt1_400 = Problem_one(N=400)
plt1_400.show()

print()


#Problem 2
plot2_40 = Problem_two(N=40)
plot2_40.show()
plt2_400 = Problem_two(N=400)
plt2_400.show()
print()








