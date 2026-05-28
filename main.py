# RK4 Sistema Esfera - Riel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

#Declaracion constantes Fisicas
r = 0.018 # Radio de la esfera en [m]
m = 0.27 # Masa de la esfera en [kg]
Ib = 4.32e-5 # Momento de inercia de la esfera [kg * m^2]
M = 1.122 # Masa del riel en [kg]
lw = 0.50 # Radio del riel en [m]
Iw = 14.025e-2 # Momento de inercia del riel en [kg * m^2]
l = 0.49 # Distancia de alicacion de la fuerza en [m]
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

# Funcion del input
def u(t):
    return 0.000
    #return 0.0001 # 0.0001 Para notar como la fuerza induce a que la esfera se mueva debido a la inclinacion

#Datos del problema
h = 0.001
t0 = 0
tn = 50
x0 = np.array([0, 0, 0, 0]) # Esfera desplazada 10 cm del centro
n = int(np.round((tn-t0)/h,0))

# 1. Línea del tiempo (eje X) — n+1 puntos desde t0 hasta tn
t_valores = np.linspace(t0, tn, n + 1)

# 2. Matriz de estados (eje Y) — n+1 filas × 4 columnas (x1, x2, x3, x4)
x_valores = np.zeros((n + 1, 4))

# 3. Registrar el estado inicial en la primera fila
x_valores[0] = x0

#Funcion del sistema Esfera-Riel
def f(t, x):
    x1 = x[0]  # Posicion de la esfera
    x2 = x[1]  # Velocidad de la esfera
    x3 = x[2]  # Angulo del riel
    x4 = x[3]  # Velocidad angular del riel

    # Derivadas del sistema Esfera-Riel
    x1prima = x2

    x2primaNum = a2 * ( ( b2*x1*x2 + b3 ) * x4 + b4 * x3 - b6 * x1 * np.cos(x3)) + (m * x1**2 + b1) * (a3*np.sin(x3) + m * x1 * x4**2) - a2 * l * np.cos(x3) * u(t)
    x2primaDenom = a1*(m*x1**2 + b1) - a2 * b5
    x2prima = x2primaNum / x2primaDenom

    x3prima = x4

    x4pter1 =  (-(b2 * x1 * x2 + b3)*x4 - b4*x3 + b6 * x1 * np.cos(x3) ) / (m * x1**2 + b1)
    x4pter2 =  ( b5 * (a3 * np.sin(x3) + m * x1 * x4**2) ) / ( a1*(m * x1**2 + b1) - a2 * b5 )
    x4pter3 = (a2*b5 * ( (b2 * x1 * x2 + b3)*x4 + b4*x3 - b6 * x1 * np.cos(x3)))/( (m * x1**2 + b1) * (a1*(m * x1**2 + b1) - a2 * b5) )
    x4pter4 = (1 + (a2 * b5) / ( a1 * (m * x1**2 + b1) - a2 * b5)) * ( ( l * np.cos(x3) * u(t)) / ( m * x1**2 + b1 ) )

    x4prima = x4pter1 - x4pter2 - x4pter3 + x4pter4

    # Retorno del vector de cambio corregido
    return np.array([x1prima, x2prima, x3prima, x4prima])


#Ciclo RK4
for i in range(n):
    t_i = t_valores[i]
    x_i = x_valores[i]

    #Frontera física Riel de medida 1metro (2 veces el lw)
    if abs(x_i[0]) >= lw:
        x_valores[i+1:] = x_i      #Se rellena el vector de valores de x, tal que no se mueva de el punto actual (ya que es la frontera)
        x_valores[i+1:, 1] = 0.0   #La velocidad de la esfera cae a cero por llegar a la frontera fisica ideal que detiene el movimiento sin ningun rebote
        print(f"--> Simulación finalizada: la esfera alcanzó el límite físico del riel en t = {t_i:.3f} s")
        break

    k_1 = h * f(t_i, x_i)
    k_2 = h * f(t_i + 0.5 * h, x_i + 0.5 * k_1)
    k_3 = h * f(t_i + 0.5 * h, x_i + 0.5 * k_2)
    k_4 = h * f(t_i + h, x_i + k_3)

    x_valores[i + 1] = x_i + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)

#RESULTADOS DE LA SIMULACION
print("Resultados de la Simulación RK4 — Sistema Esfera-Riel")
print(f"Tiempo de simulación: [{t0},{tn}] segundos  |  Paso h = {h} [s]  |  Iteraciones: {n}")
print(f"\nEstado inicial  (t=0):   x1={x_valores[0,0]:.6f}, x2={x_valores[0,1]:.6f}, x3={x_valores[0,2]:.6f}, x4={x_valores[0,3]:.6f}")
print(f"Estado final (t={tn}):  x1={x_valores[-1,0]:.6f}, x2={x_valores[-1,1]:.6f}, x3={x_valores[-1,2]:.6f}, x4={x_valores[-1,3]:.6f}")


#VISUALIZAR EN EL NOTEBOOK LAS GRAFICAS

#VISUALIZACION FISICA DEL SISTEMA ESFERA RIEL 
def visual_sistema():
    t_fisico = 10.0 #Tiempo a mostrar
    idx_fin = int(t_fisico / h)
    paso_fisico = 25
    t_fis = t_valores[:idx_fin:paso_fisico]
    x_fis = x_valores[:idx_fin:paso_fisico]
    n_frames_fis = len(t_fis)

    #Escalar la posición de la esfera para visualización (clamp para que no se salga)
    L_riel = 0.5  #Longitud VISUAL del riel
    pos_esfera = x_fis[:, 0]
    pos_visual = np.clip(pos_esfera, -lw, lw)

    fig_fis, ax_fis = plt.subplots(figsize=(10, 8))
    ax_fis.set_xlim(-0.8, 0.8)
    ax_fis.set_ylim(-0.8, 0.8)
    ax_fis.set_aspect('equal')
    ax_fis.set_title('Animación del Sistema Esfera-Riel', fontsize=16, fontweight='bold')
    ax_fis.grid(True, alpha=0.2)
    ax_fis.set_xlabel('x [m]')
    ax_fis.set_ylabel('y [m]')

    #Punto central de la viga
    pivote = plt.Circle((0, 0), 0.015, color='black', zorder=5)
    ax_fis.add_patch(pivote)

    #Línea del riel
    linea_riel, = ax_fis.plot([], [], 'k-', linewidth=4, solid_capstyle='round', zorder=3)

    #Esfera
    esfera = plt.Circle((0, 0), 0.03, color='tab:blue', zorder=4, ec='darkblue', linewidth=1.5)
    ax_fis.add_patch(esfera)

    #Rastro de la esfera
    rastro, = ax_fis.plot([], [], 'tab:blue', alpha=0.3, linewidth=0.8, zorder=2)
    rastro_x = []
    rastro_y = []

    #Textos informativos
    info_tiempo = ax_fis.text(-0.75, 0.72, '', fontsize=11, fontfamily='monospace',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    info_estado = ax_fis.text(-0.75, 0.55, '', fontsize=9, fontfamily='monospace',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))

    def init_fisica():
        linea_riel.set_data([], [])
        esfera.center = (0, 0)
        rastro.set_data([], [])
        rastro_x.clear()
        rastro_y.clear()
        info_tiempo.set_text('')
        info_estado.set_text('')
        return [linea_riel, esfera, rastro, info_tiempo, info_estado]

    def update_fisica(frame):
        theta = -x_fis[frame, 2]  # ángulo del riel
        pos = pos_visual[frame]   # posición de la esfera (escalada)

        # Extremos del riel rotado
        riel_x1 = -L_riel * np.cos(theta)
        riel_y1 = -L_riel * np.sin(theta)
        riel_x2 =  L_riel * np.cos(theta)
        riel_y2 =  L_riel * np.sin(theta)
        linea_riel.set_data([riel_x1, riel_x2], [riel_y1, riel_y2])

        # Posición de la esfera sobre el riel
        esf_x = pos * np.cos(theta)
        esf_y = pos * np.sin(theta)
        esfera.center = (esf_x, esf_y)

        # Rastro
        rastro_x.append(esf_x)
        rastro_y.append(esf_y)
        rastro.set_data(rastro_x, rastro_y)

        # Info
        info_tiempo.set_text(f't = {t_fis[frame]:.3f} s')
        info_estado.set_text(
            f'pos = {x_fis[frame,0]:.4f} m\n'
            f'vel = {x_fis[frame,1]:.4f} m/s\n'
            f'θ   = {x_fis[frame,2]:.4f} rad\n'
            f'ω   = {x_fis[frame,3]:.4f} rad/s'
        )

        return [linea_riel, esfera, rastro, info_tiempo, info_estado]

    anim_fisica = FuncAnimation(fig_fis, update_fisica, init_func=init_fisica,
                                frames=n_frames_fis, interval=30, blit=True, repeat=True)

    plt.show()


visual_sistema()