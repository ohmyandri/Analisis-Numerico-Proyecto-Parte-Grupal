# RK4 Sistema Esfera - Riel
import numpy as np
import matplotlib.pyplot as plt

#Proponerlas

k1=1
k2=1
k3=1
k4=1
bb=1
bp=1
Tm=1

#Función
def f(t,x):
    y = x[0]
    y_der = x[1]
    theta = x[2]
    theta_der= x[3]

    dy=y_der
    ddy= (Tm-np.sin(theta)*np.cos(theta)*(2*k2*theta_der*y_der-k3*(y_der)**2*np.cos(theta)-((k3*k4)/k2))-k3*(theta_der**2)*np.sin(theta)-bb*y_der-((k3*bp)/k2)*theta_der*np.cos(theta))/(k1+k2*(np.sin(theta)**2)-((k3**2)/k2)*(np.cos(theta)**2))

    dtheta=theta_der
    ddtheta= (y_der**2*np.sin(theta)*np.cos(theta)+((k4/k2)*np.sin(theta))-((bp/k2)*theta_der)+((k3/k2)*np.cos(theta)*ddy))

    return np.array([dy, ddy, dtheta, ddtheta])

#Datos del problema
h = 0.001
t0 = 0
x0 = np.array([0,0,0.1,0]) #Proponerlas
tn = 10
n = int(np.round((tn-t0)/h,0))

#Guardar

for i in range(n):
    k_1 = h*f(t0,x0)
    k_2 = h*f(t0 + 0.5*h, x0 + 0.5*k1)
    k_3 = h*f(t0 + 0.5*h, x0 + 0.5*k2)
    k_4 = h*f(t0 + h, x0 + k3)
    x_1 = x0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    x0 = x_1
    t0 = t0+h

#Imprimir resultado

#Graficar
    