# RK4 Sistema Esfera - Riel
import numpy as np
import matplotlib.pyplot as plt

#Declaracion constantes Fisicas
r = 0.018 # Radio de la esfera en [m]
m = 0.27 # Masa de la esfera en [kg]
Ib = 4.32e-5 # Momento de inercia de la esfera [kg * m^2]
M = 1.122 # Masa del riel en [kg]
lw = 0.50 # Radio del riel en [m]
Iw = 14.025e-2 # Momento de inercia del riel en [kg * m^2]
l = 0.49 # Distancia de alicacion de la fuerza
b = 1.0 # Friccion del sistema conductor en [N * s / m]
K = 0.001 # Constante de resorte del sistema conductor en [N/m]
g = 9.81 # Constante de aceleracion gravitatoria en [m / s^2]

# Constantes auxiliares para el sistema de ecuaciones de movimiento
a1 = m + (Ib / r**2 )
a2 = m*r + (Ib / r)
a3 = m*g

b1 = Ib + Iw + m * r**2
b2 = 2 * m
b3 = b * l**2
b4 = K * l**2
b5 = m * r + (Ib / r)
b6 = m * g

# Definicion de variables
x1 = 0 # X Posicion de la esfera.
x2 = 0 # X Valocidad de la esfera.
x3 = 0 # X Angulo del riel.
x4 = 0 # X Velocidad angular del riel.

# Funcion del input

def u(t):
    return 0.0

# Derivadas
x1prima = x2

x2primaNum = a2 * ( ( b2*x1*x2 + b3 ) * x4 + b4 * x3 - b6 * x1 * np.cos(x3)) + (m * x1**2 + b1) * (a3*np.sin(x3) + m * x1 * x4**2) - a2 * l * np.cos(x3) * u(t)
x2primaDenom = a1*(m*x1**2 + b1) - a2 * b5
x2prima = x2primaNum / x2primaDenom

x3prima= x4

#Proponerlas
k1=1
k2=1
k3=1
k4=1
bb=1
bp=1
Tm=1

#Funcion
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
    