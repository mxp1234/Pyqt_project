import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from scipy import signal
import airparameter 
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

z = airparameter.z
mu = airparameter.mu
Vv = airparameter.Vv
Pid = np.array([[7,0.4,4],
    [7.5, 0.4, 4.4],
    [7, 0.2, 5.4]])



x01=np.array([1/57.3,0,0])


# 定义函数hypersonic_gainsch，将MATLAB函数转换为Python
def hypersonic_gainsch(x1, x3):
    ast1 = 0.0
    ast3 = 0.0
    if x1 < -0.4:
        ast1 = -0.4
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < -0.3:
        ast1 = -0.3
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < -0.2:
        ast1 = -0.2
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < -0.1:
        ast1 = -0.1
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < 0:
        ast1 = 0
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < 0.1:
        ast1 = 0.1
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < 0.2:
        ast1 = 0.2
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < 0.3:
        ast1 = 0.3
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5
    elif x1 < 0.4:
        ast1 = 0.4
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5

        # ... （重复上面的elif块，直到x1 < 0.5）
    elif x1 < 0.5:
        ast1 = 0.5
        if x3 < -0.4:
            ast3 = -0.4
        elif x3 < -0.3:
            ast3 = 0.3
        elif x3 < -0.2:
            ast3 = 0.2
        elif x3 < -0.1:
            ast3 = -0.1
        elif x3 < 0:
            ast3 = 0
        elif x3 < 0.1:
            ast3 = 0.1
        elif x3 < 0.2:
            ast3 = 0.2
        elif x3 < 0.3:
            ast3 = 0.3
        elif x3 < 0.4:
            ast3 = 0.4
        elif x3 < 0.5:
            ast3 = 0.5

    # 计算kMxg1、kMxg2、kMxg3、kMyg1、kMyg2、kMyg3、kMzg1、kMzg2和kMzg3
    kMxg1 = 0
    kMxg2 = (x1 - (ast1 - 0.1)) / 0.1 * (np.sin(ast1) - np.sin(ast1 - 0.1)) + np.sin(ast1 - 0.1)
    kMxg3 = 1
    kMyg1 = (x3 - (ast3 - 0.1)) / 0.1 * (np.sin(ast3) - np.sin(ast3 - 0.1)) + np.sin(ast3 - 0.1)
    kMyg2 = ((x1 - (ast1 - 0.1)) / 0.1 * (np.cos(ast1) - np.cos(ast1 - 0.1)) + np.cos(ast1 - 0.1)) * ((x3 - (ast3 - 0.1)) / 0.1 * (np.cos(ast3) - np.cos(ast3 - 0.1)) + np.cos(ast3 - 0.1))
    kMyg3 = 0
    kMzg1 = (x3 - (ast3 - 0.1)) / 0.1 * (np.cos(ast3) - np.cos(ast3 - 0.1)) + np.cos(ast3 - 0.1)
    kMzg2 = -((x1 - (ast1 - 0.1)) / 0.1 * (np.cos(ast1) - np.cos(ast1 - 0.1)) + np.cos(ast1 - 0.1)) * ((x3 - (ast3 - 0.1)) / 0.1 * (np.sin(ast3) - np.sin(ast3 - 0.1)) + np.sin(ast3 - 0.1))
    kMzg3 = 0

    # 将结果存储在KMGainsch中并返回
    KMGainsch = np.array([kMxg1, kMxg2, kMxg3, kMyg1, kMyg2, kMyg3, kMzg1, kMzg2, kMzg3])

    return KMGainsch

# 然后在你的代码中的hypersonic_dobBased函数中调用该函数
# 示例：在hypersonic_dobBased函数中调用hypersonic_gainsch
# 示例：kMGainsch = hypersonic_gainsch(x1, x3)

def airpara0(h, z, Vv, mu):
# 从2000开始没改
    rho = 0.0
    g = 0.0
    Vvvoice = 0.0
    if h<11000:
        z_values = [z[i] + (z[i+1] - z[i]) * (h - i * 500) / 500 for i in range(23)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - i * 500) / 500 for i in range(23)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - i * 500) / 500 for i in range(23)]
        for i in range(23):
            if h < (i + 1) * 500:
                rho = z_values[i]
                g = mu_values[i]
                Vvvoice = Vv_values[i]
                break
        
    elif h<32000:
        z_values = [z[i] + (z[i+1] - z[i]) * (h - 11000 -(i-23) * 1000) / 1000 for i in range(23,44)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - 11000 -(i-23) * 1000) / 1000 for i in range(23,44)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - 11000 -(i-23) * 1000) / 1000 for i in range(23,44)]
        for i in range(23,44):
            if h < 11000+(i - 22) * 1000:
                rho = z_values[i-23]
                g = mu_values[i-23]
                Vvvoice = Vv_values[i-23]
                break
    
    elif h<48000:            
        z_values = [z[i] + (z[i+1] - z[i]) * (h - 32000 -(i-44) * 2000) / 2000 for i in range(44,52)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - 32000 -(i-44) * 2000) / 2000 for i in range(44,52)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - 32000 -(i-44) * 2000) / 2000 for i in range(44,52)]
        for i in range(44,52):    
            if h < 32000+(i - 43) * 2000:
                rho = z_values[i-44]
                g = mu_values[i-44]
                Vvvoice = Vv_values[i-44]
                break
    elif h<100000:
        z_values = [z[i] + (z[i+1] - z[i]) * (h - 48000 -(i-52) * 5000) / 5000 for i in range(52,63)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - 48000 -(i-52) * 5000) / 5000 for i in range(52,63)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - 48000 -(i-52) * 5000) / 5000 for i in range(52,63)]
        for i in range(52,63):
            if h < 48000+(i - 51) * 5000:
                rho = z_values[i-52]
                g = mu_values[i-52]
                Vvvoice = Vv_values[i-52]
                break
    elif h<300000:        
        z_values = [z[i] + (z[i+1] - z[i]) * (h - 100000 -(i-63) * 10000) / 10000 for i in range(63,83)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - 100000 -(i-63) * 10000) / 10000 for i in range(63,83)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - 100000 -(i-63) * 10000) / 10000 for i in range(63,83)]
        for i in range(63,83):
            if h < 100000+(i - 62) * 10000:
                rho = z_values[i-63]
                g = mu_values[i-63]
                Vvvoice = Vv_values[i-63]
                break
            
    elif h<500000:
        z_values = [z[i] + (z[i+1] - z[i]) * (h - 300000 -(i-63) * 20000) / 20000 for i in range(83,93)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - 300000 -(i-63) * 20000) / 20000 for i in range(83,93)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - 300000 -(i-63) * 20000) / 20000 for i in range(83,93)]
        for i in range(83,93):
            if h < 300000+(i - 82) * 20000:
                rho = z_values[i-83]
                g = mu_values[i-83]
                Vvvoice = Vv_values[i-83]
                break
        
    elif h<1000000:
    
        z_values = [z[i] + (z[i+1] - z[i]) * (h - 500000 -(i-93) * 50000) / 50000 for i in range(93,100)]
        mu_values = [mu[i] + (mu[i+1] - mu[i]) * (h - 500000 -(i-93) * 50000) / 50000 for i in range(93,100)]
        Vv_values = [Vv[i] + (Vv[i+1] - Vv[i]) * (h - 500000 -(i-93) * 50000) / 50000 for i in range(93,100)]
        for i in range(93,100):   
            if h < 500000+(i - 92) * 50000:
                rho = z_values[i]
                g = mu_values[i]
                Vvvoice = Vv_values[i]
                break  
    return rho, g, Vvvoice



def airpara(h, z, Vv, mu):
    z_values = np.zeros_like(h)
    mu_values = np.zeros_like(h)
    Vv_values = np.zeros_like(h)

    conditions = [
        (h < 11000),
        (h < 32000),
        (h < 48000),
        (h < 100000),
        (h < 300000),
        (h < 500000),
        (h < 1000000)
    ]



    z_ranges = [(0, 23), (23, 44), (44, 52), (52, 63), (63, 83), (83, 93), (93, 100)]

    for idx, (start, end) in enumerate(z_ranges):
        condition = conditions[idx]
        for i in range(start, end):
            z_values[condition] = z[i] + (z[i + 1] - z[i]) * (h[condition] - (i - start) * (end - start) * 1000) / ((end - start) * 1000)
            mu_values[condition] = mu[i] + (mu[i + 1] - mu[i]) * (h[condition] - (i - start) * (end - start) * 1000) / ((end - start) * 1000)
            Vv_values[condition] = Vv[i] + (Vv[i + 1] - Vv[i]) * (h[condition] - (i - start) * (end - start) * 1000) / ((end - start) * 1000)
        break

    return z_values, mu_values, Vv_values


def Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz):
    # Calculate derivatives
    dx1 = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dx2 = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / np.cos(x[0])
    dx3 = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))

    # Extract KMGainsch
    Px1, Px2, Px3, Py1, Py2, Py3, Pz1, Pz2, Pz3 = KMGainsch

    PPP = np.array([[Px1, Px2, Px3],
                    [Py1, Py2, Py3],
                    [Pz1, Pz2, Pz3]])

    # Extract Pid gains
    kpv1, kiv1, kdv1 = Pid[0]
    kpv2, kiv2, kdv2 = Pid[1]
    kpv3, kiv3, kdv3 = Pid[2]

    # Calculate control output
    Uu = np.array([kiv1 * x[9] + kpv1 * (x01[0] - x[0]) + kdv1 * (-dx1),
                   kiv2 * x[10] + kpv2 * (x01[1] - x[1]) + kdv2 * (-dx2),
                   kiv3 * x[11] + kpv3 * (x01[2] - x[2]) + kdv3 * (-dx3)])

    UM = np.dot(np.dot(J, PPP), Uu) - np.dot(np.dot(J, PPP), np.array([dhatx, dhaty, dhatz]))

    return UM



def duopianjisuan(x, UM, altha, beta, an, piang, piang1, piang2, piang3, Vvoice, rho, piang11, piang33):
    Sref = 334.73
    b = 18.288
    c = 24.384
    
    x4 = x[15]  # Velocity variable

    qbar = 0.5 * rho * x4**2  # Dynamic pressure

    wx = x[3]  # x-axis angular velocity
    wy = x[4]  # y-axis angular velocity
    wz = x[5]  # z-axis angular velocity

    Ma = x4 / Vvoice  # Mach number

    # Roll moment linear coefficients
    CMx_beta = -1.402 * 0.1 + 3.326 * 0.01 * Ma - 7.59e-4 * altha + 8.596e-6 * (altha * Ma) - 3.794e-3 * Ma**2 + 2.354e-6 * altha**2 - 1.044e-8 * (altha * Ma)**2 + 2.219e-4 * Ma**3 - 8.964e-18 * altha**3 - 6.462e-6 * Ma**4 + 3.803e-19 * altha**4 + 7.419e-8 * Ma**5 - 3.353e-21 * altha**5
    CMx_deltaaltha_con = 3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 4.95e-6 * altha**2 + 1.411e-6 * Ma**2
    CMx_deltaaltha_u1 = 1.17e-4 * piang1 + 2.794e-8 * altha * Ma * piang1

    CMx_deltae_con = -(3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 4.95e-6 * altha**2 + 1.411e-6 * Ma**2)
    CMx_deltae_u3 = -(1.17e-4 * piang3 + 2.794e-8 * altha * Ma * piang3)

    CMx_deltar_con = -5.0103e-19 + 6.2723e-20 * altha + 2.3418e-20 * Ma - 3.4201e-21 * altha * Ma
    CMx_deltar_u2 = -3.5496e-6 * Ma * piang2 + 5.5547e-8 * altha * Ma * piang2 + 1.1441e-4 * piang2 - 2.6824e-6 * altha * piang2

    CMx_r = 3.82 * 0.1 - 1.06 * 0.1 * Ma + 1.94e-3 * altha - 8.15e-5 * altha * Ma + 1.45 * 0.01 * Ma**2 - 9.76e-6 * altha**2 + 4.49e-8 * (altha * Ma)**2 - 1.02e-3 * Ma**3 - 2.7e-7 * altha**3 + 3.56e-5 * Ma**4 + 3.19e-8 * altha**4 - 4.81e-7 * Ma**5 - 1.06e-9 * altha**5
    CMx_p = -2.99 * 0.1 + 7.47 * 0.01 * Ma + 1.38e-3 * altha - 8.78e-5 * altha * Ma - 9.13e-3 * Ma**2 - 2.04e-4 * altha**2 - 1.52e-7 * (altha * Ma)**2 + 5.73e-4 * Ma**3 - 3.86e-5 * altha**3 - 1.79e-5 * Ma**4 + 4.21e-6 * altha**4 + 2.2e-7 * Ma**5 - 1.15e-7 * altha**5

    CMx_con = CMx_beta * beta + CMx_deltaaltha_con + CMx_deltae_con + CMx_deltar_con + CMx_r * wz * b / (2 * x4) + CMx_p * wx * b / (2 * x4)

    # Pitch moment linear coefficients
    CMz_altha = -2.192e-2 + 7.739e-3 * Ma - 2.26e-3 * altha + 1.808e-4 * Ma * altha - 8.849e-4 * Ma**2 + 2.616e-4 * altha**2 - 2.88e-7 * (Ma * altha)**2 + 4.617e-5 * Ma**3 - 7.887e-5 * altha**3 - 1.143e-6 * Ma**4 + 8.288e-6 * altha**4 + 1.082e-8 * Ma**5 - 2.789e-7 * altha**5
    CMz_deltaaltha_con = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma - 4.46e-6 * altha * Ma
    CMz_deltaaltha_u1 = 2.89e-4 * piang1 + 4.48e-6 * altha * piang1 - 5.87e-6 * Ma * piang1 + 9.72e-8 * altha * Ma * piang1

    CMz_deltae_con = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma - 4.46e-6 * altha * Ma
    CMz_deltae_u3 = 2.89e-4 * piang3 + 4.48e-6 * altha * piang3 - 5.87e-6 * Ma * piang3 + 9.72e-8 * altha * Ma * piang3

    CMz_deltar = -2.79e-5 * altha - 5.89e-8 * altha**2 + 1.58e-3 * Ma**2 + 6.42e-8 * altha**3 - 6.69e-4 * Ma**3 - 2.1e-8 * altha**4 + 1.05e-4 * Ma**4 + 3.14e-9 * altha**5 - 7.74e-6 * Ma**5 - 2.18e-10 * altha**6 + 2.7e-7 * Ma**6 + 5.74e-12 * altha**7 - 3.58e-9 * Ma**7
    CMz_deltac = 0
    CMz_q = -1.36 + 3.86e-1 * Ma + 7.85e-4 * altha + 1.4e-4 * altha * Ma - 5.42e-2 * Ma**2 + 2.36e-3 * altha**2 - 1.95e-6 * (altha * Ma)**2 + 3.8e-3 * Ma**3 - 1.48e-3 * altha**3 - 1.3e-4 * Ma**4 + 1.69e-4 * altha**4 + 1.71e-6 * Ma**5 - 5.93e-6 * altha**5
    CMz_con = CMz_altha + CMz_deltaaltha_con + CMz_deltae_con + CMz_deltar + CMz_deltac + CMz_q * wy * c / (2 * x4)
    
    # print(CMz_con)
    # Yaw moment linear coefficients
    CMy_beta = 6.998e-4 * altha + 5.9115 * 0.01 * Ma - 7.525e-5 * Ma * altha + 2.516e-4 * altha**2 - 1.4824 * 0.01 * Ma**2 - 2.1924e-7 * (Ma * altha)**2 - 1.0777e-4 * altha**3 + 1.2692e-3 * Ma**3 + 1.0707e-8 * (Ma * altha)**3 + 9.4989e-6 * altha**4 - 4.7098e-5 * Ma**4 - 5.5472e-11 * (Ma * altha)**4 - 2.5953e-7 * altha**5 + 6.4284e-7 * Ma**5 + 8.5863e-14 * (Ma * altha)**5
    CMy_deltaaltha_con = 2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 6.39e-7 * altha**2 + 8.16e-7 * Ma**2
    CMy_deltaaltha_u1 = -1.3e-5 * piang1 - 8.93e-8 * altha * Ma * piang1
    CMy_deltae_con = -(2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 6.39e-7 * altha**2 + 8.16e-7 * Ma**2)
    CMy_deltae_u3 = 1.3e-5 * piang3 + 8.93e-8 * altha * Ma * piang3
    CMy_deltar_con = 2.85e-18 - 3.59e-19 * altha - 1.26e-19 * Ma + 1.57e-20 * (altha * Ma)
    CMy_deltar_u2 = -5.28e-4 * piang2 + 1.39e-5 * altha * piang2 + 1.65e-5 * (Ma * piang2) - 3.13e-7 * (altha * Ma) * piang2

    CMy_p = 3.68 * 0.1 - 9.79e-2 * Ma + 7.61e-16 * altha + 1.24e-2 * Ma**2 - 4.64e-16 * altha**2 - 8.05e-4 * Ma**3 + 1.01e-16 * altha**3 + 2.57e-5 * Ma**4 - 9.18e-18 * altha**4 - 3.2e-7 * Ma**5 + 2.96e-19 * altha**5
    CMy_r = -2.41 + 5.96e-1 * Ma - 2.74e-3 * altha + 2.09e-4 * (altha * Ma) - 7.57e-2 * Ma**2 + 1.15e-3 * altha**2 - 6.53e-8 * (altha * Ma)**2 + 4.9e-3 * Ma**3 - 3.87e-4 * altha**3 - 1.57e-4 * Ma**4 + 3.6e-5 * altha**4 + 1.96e-6 * Ma**5 - 1.18e-6 * altha**5

    CMy_con = CMy_beta * beta + CMy_deltaaltha_con + CMy_deltae_con + CMy_deltar_con + CMy_p * wx * b / (2 * x4) + CMy_r * wz * b / (2 * x4)

    # Solving for control deflections
    ACM = np.array([[1 / (qbar * Sref), 0, 0],
                    [0, 1 / (qbar * b * Sref), 0],
                    [0, 0, 1 / (qbar * c * Sref)]])

    ACMcon = np.array([an * CMx_con, an * CMy_con, an * CMz_con])
    # print(ACMcon)
    ACMu = an * np.array([[CMx_deltaaltha_u1, CMx_deltar_u2, CMx_deltae_u3],
                        [CMy_deltaaltha_u1, CMy_deltar_u2, CMy_deltae_u3],
                        [CMz_deltaaltha_u1, 0, CMz_deltae_u3]])

    I = np.eye(3)

    u = np.linalg.solve(ACMu, np.dot(ACM, UM) - ACMcon)

    ubar = 25 / 57.3

    for i in range(3):
        if u[i] > ubar:
            u[i] = ubar
        elif u[i] < -ubar:
            u[i] = -ubar

    return u



def lijuqiujie(x, u, altha, beta, an, piang, piang1, piang2, piang3, rho, Vvoice, piang11, piang33):
    # Constants
    Sref = 334.73
    b = 18.288
    c = 24.384
    deltarho = -0.3
    Vfield = 0  #理解是场的速度，即风速

    # Extract variables from input arrays
    x4 = x[15]
    qbar = 0.5 * rho*(1+deltarho)  * (x4 + Vfield) ** 2  #次数为偏差设置处
    # print(u)
    wx = x[3]
    wy = x[4]
    wz = x[5]

    Ma = x4 / Vvoice

    # Calculate CMx (aerodynamic moment coefficient)
    CMx_beta = -1.402 * 0.1 + 3.326 * 0.01 * Ma - 7.59e-4 * altha + 8.596e-6 * (altha * Ma) - 3.794e-3 * Ma**2 + 2.354e-6 * altha**2 - 1.044e-8 * (altha * Ma)**2 + 2.219e-4 * Ma**3 - 8.964e-18 * altha**3 - 6.462e-6 * Ma**4 + 3.803e-19 * altha**4 + 7.419e-8 * Ma**5 - 3.353e-21 * altha**5
    CMx_deltaaltha = 3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 1.17e-4 * piang1 * u[0] + 2.794e-8 * altha * Ma * piang1 * u[0] + 4.95e-6 * altha**2 + 1.411e-6 * Ma**2 - 1.16e-6 * (piang1 * u[0])**2 - 4.641e-11 * (altha * Ma * piang1 * u[0])**2
    CMx_deltae = -(3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 1.17e-4 * piang3 * u[2] + 2.794e-8 * altha * Ma * piang3 * u[2] + 4.95e-6 * altha**2 + 1.411e-6 * Ma**2 - 1.16e-6 * (piang3 * u[2])**2 - 4.641e-11 * (altha * Ma * piang3 * u[2])**2)
    CMx_deltar = -5.0103e-19 + 6.2723e-20 * altha + 2.3418e-20 * Ma + 1.1441e-4 * piang2 * u[1] - 2.6824e-6 * altha * piang2 * u[1] - 3.4201e-21 * altha * Ma - 3.5496e-6 * Ma * piang2 * u[1] + 5.5547e-8 * altha * Ma * piang2 * u[1]
    CMx_r = 3.82 * 0.1 - 1.06 * 0.1 * Ma + 1.94e-3 * altha - 8.15e-5 * altha * Ma + 1.45 * 0.01 * Ma**2 - 9.76e-6 * altha**2 + 4.49e-8 * (altha * Ma)**2 - 1.02e-3 * Ma**3 - 2.7e-7 * altha**3 + 3.56e-5 * Ma**4 + 3.19e-8 * altha**4 - 4.81e-7 * Ma**5 - 1.06e-9 * altha**5
    CMx_p = -2.99 * 0.1 + 7.47 * 0.01 * Ma + 1.38e-3 * altha - 8.78e-5 * altha * Ma - 9.13e-3 * Ma**2 - 2.04e-4 * altha**2 - 1.52e-7 * (altha * Ma)**2 + 5.73e-4 * Ma**3 - 3.86e-5 * altha**3 - 1.79e-5 * Ma**4 + 4.21e-6 * altha**4 + 2.2e-7 * Ma**5 - 1.15e-7 * altha**5
    CMx = CMx_beta * beta + CMx_deltaaltha + CMx_deltae + CMx_deltar + CMx_r * wz * b / (2 * x4) + CMx_p * wx * b / (2 * x4)
    Mx = qbar * Sref * an * CMx

    # Calculate CMz (aerodynamic moment coefficient)
    CMz_altha = -2.192e-2 + 7.739e-3 * Ma - 2.26e-3 * altha + 1.808e-4 * Ma * altha - 8.849e-4 * Ma**2 + 2.616e-4 * altha**2 - 2.88e-7 * (Ma * altha)**2 + 4.617e-5 * Ma**3 - 7.887e-5 * altha**3 - 1.143e-6 * Ma**4 + 8.288e-6 * altha**4 + 1.082e-8 * Ma**5 - 2.789e-7 * altha**5
    CMz_deltaaltha = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma + 2.89e-4 * piang1 * u[0] + 4.48e-6 * altha  * piang1 * u[0] - 4.46e-6 * altha * Ma - 5.87e-6 * Ma * piang1 * u[0] + 9.72e-8 * altha * Ma * piang1 * u[0]
    CMz_deltae = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma + 2.89e-4 * piang3 * u[2] + 4.48e-6 * altha  * piang3 * u[2] - 4.46e-6 * altha * Ma - 5.87e-6 * Ma * piang3 * u[2] + 9.72e-8 * altha * Ma * piang3 * u[2]
    CMz_deltar = -2.79e-5 * altha - 5.89e-8 * altha**2 + 1.58e-3 * Ma**2 + 6.42e-8 * altha**3 - 6.69e-4 * Ma**3 - 2.1e-8 * altha**4 + 1.05e-4 * Ma**4 + 3.14e-9 * altha**5 - 7.74e-6 * Ma**5 - 2.18e-10 * altha**6 + 2.7e-7 * Ma**6 + 5.74e-12 * altha**7 - 3.58e-9 * Ma**7 + 1.43e-7 * (piang * u[1])**4 - 4.77e-22 * (piang * u[1])**5 - 3.38e-10 * (piang * u[1])**6 + 2.63e-24 * (piang * u[1])**7
    CMz_deltac = 0
    CMz_q = -1.36 + 3.86e-1 * Ma + 7.85e-4 * altha + 1.4e-4 * altha * Ma - 5.42e-2 * Ma**2 + 2.36e-3 * altha**2 - 1.95e-6 * (altha * Ma)**2 + 3.8e-3 * Ma**3 - 1.48e-3 * altha**3 - 1.3e-4 * Ma**4 + 1.69e-4 * altha**4 + 1.71e-6 * Ma**5 - 5.93e-6 * altha**5
    CMz = CMz_altha + CMz_deltaaltha + CMz_deltae + CMz_deltar + CMz_deltac + CMz_q * wy * c / (2 * x4)
    Mz = qbar * c * Sref * an * CMz

    # print(u)
    # Calculate CMy (aerodynamic moment coefficient)
    CMy_beta = 6.998e-4 * altha + 5.9115 * 0.01 * Ma - 7.525e-5 * Ma * altha + 2.516e-4 * altha**2 - 1.4824 * 0.01 * Ma**2 - 2.1924e-7 * (Ma * altha)**2 - 1.0777e-4 * altha**3 + 1.2692e-3 * Ma**3 + 1.0707e-8 * (Ma * altha)**3 + 9.4989e-6 * altha**4 - 4.7098e-5 * Ma**4 - 5.5472e-11 * (Ma * altha)**4 - 2.5953e-7 * altha**5 + 6.4284e-7 * Ma**5 + 8.5863e-14 * (Ma * altha)**5
    CMy_deltaaltha = 2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 1.3e-5 * piang1 * u[0] - 8.93e-8 * altha * Ma * piang1 * u[0] - 6.39e-7 * altha**2 + 8.16e-7 * Ma**2 + 1.97e-6 * (piang1 * u[0])**2
    CMy_deltae = -(2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 1.3e-5 * piang3 * u[2] - 8.93e-8 * altha * Ma * piang3 * u[2] - 6.39e-7 * altha**2 + 8.16e-7 * Ma**2 + 1.97e-6 * (piang3 * u[2])**2)
    CMy_deltar = 2.85e-18 - 3.59e-19 * altha - 1.26e-19 * Ma - 5.28e-4 * piang2 * u[1] + 1.39e-5 * altha * piang2 * u[1] + 1.57e-20 * (altha * Ma) + 1.65e-5 * (Ma * piang2 * u[1]) - 3.13e-7 * (altha * Ma) * piang2 * u[1]
    CMy_p = 3.68e-1 - 9.79e-2 * Ma + 7.61e-16 * altha + 1.24e-2 * Ma**2 - 4.64e-16 * altha**2 - 8.05e-4 * Ma**3 + 1.01e-16 * altha**3 + 2.57e-5 * Ma**4 - 9.18e-18 * altha**4 - 3.2e-7 * Ma**5 + 2.96e-19 * altha**5
    CMy_r = -2.41 + 5.96e-1 * Ma - 2.74e-3 * altha + 2.09e-4 * (altha * Ma) - 7.57e-2 * Ma**2 + 1.15e-3 * altha**2 - 6.53e-8 * (altha * Ma)**2 + 4.9e-3 * Ma**3 - 3.87e-4 * altha**3 - 1.57e-4 * Ma**4 + 3.6e-5 * altha**4 + 1.96e-6 * Ma**5 - 1.18e-6 * altha**5
    CMy = CMy_beta * beta + CMy_deltaaltha +  CMy_deltae + CMy_deltar + CMy_p*wx*b/(2*x4)+CMy_r*wz*b/(2*x4)
    My=qbar*b*Sref*an*CMy

    CD_altha = 8.717e-2 - 3.307 * 0.01 * Ma + 3.179 * 0.001 * altha - 1.25 * 0.0001 * altha * Ma + 5.036 * 0.001 * Ma**2 - 1.1 * 0.001 * altha**2 + 1.405e-7 * (altha * Ma)**2 - 3.658e-4 * Ma**3 + 3.175e-4 * altha**3 + 1.274e-5 * Ma**4 - 2.985e-5 * altha**4 - 1.705e-7 * Ma**5 + 9.766e-7 * altha**5
    CD_deltaaltha = 4.5548e-4 + 2.5411e-5 * altha - 1.1436e-4 * Ma - 3.6417e-5 * (piang1 * u[0]) - 5.3015e-7 * (altha * Ma) * (piang1 * u[0]) + 3.2187e-6 * altha**2 + 3.014e-6 * Ma**2 + 6.9629e-6 * (piang1 * u[0])**2 + 2.1026e-12 * (altha * Ma * (piang1 * u[0]))**2
    CD_deltae = 4.5548e-4 + 2.5411e-5 * altha - 1.1436e-4 * Ma - 3.6417e-5 * (piang3 * u[2]) - 5.3015e-7 * (altha * Ma) * (piang3 * u[2]) + 3.2187e-6 * altha**2 + 3.014e-6 * Ma**2 + 6.9629e-6 * (piang3 * u[2])**2 + 2.1026e-12 * (altha * Ma * (piang3 * u[2]))**2
    CD_deltar = 7.5e-4 - 2.29e-5 * altha - 9.69e-5 * Ma - 1.83e-6 * piang2 * u[1] + 9.13e-9 * (altha * Ma) * piang2 * u[1] + 8.76e-7 * altha**2 + 2.7e-6 * Ma**2 + 1.97e-6 * (piang2 * u[1])**2 - 1.7702e-11 * (altha * Ma * piang2 * u[1])**2
    CD_deltac = 0
    CD = CD_altha + CD_deltaaltha + CD_deltae + CD_deltar + CD_deltac
    D = qbar * Sref * CD

    
    
    CL_altha = -8.19e-2 + 4.7 * 0.01 * Ma + 1.86 * 0.01 * altha - 4.73e-4 * altha * Ma - 9.19e-3 * Ma**2 - 1.52e-4 * altha**2 + 5.99e-7 * (altha * Ma)**2 + 7.74e-4 * Ma**3 + 4.08e-6 * altha**3 - 2.93e-5 * Ma**4 - 3.91e-7 * altha**4 + 4.12e-7 * Ma**5 + 1.3e-8 * altha**5
    CL_deltaaltha = -1.45e-5 + 1.01e-4 * altha + 7.1e-6 * Ma - 4.14e-4 * (piang1 * u[0]) - 3.51e-6 * altha * (piang1 * u[0]) + 4.7e-6 * (altha * Ma) + 8.72e-6 * Ma * (piang1 * u[0]) - 1.7e-7 * altha * Ma * (piang1 * u[0])
    CL_deltae = -1.45e-5 + 1.01e-4 * altha + 7.1e-6 * Ma - 4.14e-4 * (piang3 * u[2]) - 3.51e-6 * altha * (piang3 * u[2]) + 4.7e-6 * (altha * Ma) + 8.72e-6 * Ma * (piang3 * u[2]) - 1.7e-7 * altha * Ma * (piang3 * u[2])
    CL_deltac = 0
    CL = CL_altha + CL_deltaaltha + CL_deltae + CL_deltac
    L = qbar * Sref * CL

    CY_deltaaltha = -1.02e-6 - 1.12e-7 * altha + 4.48e-7 * Ma + 2.27e-7 * (piang1 * u[0]) + 4.11e-9 * altha * Ma * (piang1 * u[0]) + 2.82e-9 * altha**2 - 2.36e-8 * Ma**2 - 5.04e-8 * (piang1 * u[0])**2 + 4.5e-14 * (altha * Ma * (piang1 * u[0]))**2
    CY_deltae = -(-1.02e-6 - 1.12e-7 * altha + 4.48e-7 * Ma + 2.27e-7 * (piang3 * u[2]) + 4.11e-9 * altha * Ma * (piang3 * u[2]) + 2.82e-9 * altha**2 - 2.36e-8 * Ma**2 - 5.04e-8 * (piang3 * u[2])**2 + 4.5e-14 * (altha * Ma * (piang3 * u[2]))**2)
    CY_deltar = -1.43e-18 + 4.86e-20 * altha + 1.86e-19 * Ma + 3.84e-4 * piang2 * u[1] - 1.17e-5 * altha * piang2 * u[1] - 1.07e-5 * Ma * piang2 * u[1] + 2.6e-7 * altha * Ma * piang2 * u[1]
    CY_beta = 2.8803e-3 * altha - 2.8943e-4 * altha * Ma + 5.4822e-2 * Ma**2 + 7.3535e-4 * altha**2 - 4.6409e-9 * ((altha * Ma**2)**2) - 2.0675e-8 * ((altha**2 * Ma)**2) + 4.6205e-6 * ((altha * Ma)**2) + 2.6144e-11 * (altha**2 * Ma**2)**2 - 4.3203e-3 * Ma**3 - 3.7405e-4 * altha**3 + Ma**4 * 1.5495e-4 + 2.8183e-5 * altha**4 + Ma**5 * -2.0829e-6 + altha**5 * -5.2083e-7
    CY = (CY_beta * beta + CY_deltaaltha + CY_deltae + CY_deltar)*1.1+0.03
    Y = qbar * Sref * CY

    Force = [D, L, Y]
    M = [Mx, My, Mz]

    # print(Mx,Mz,My)
    return Force,M


# 定义函数hypersonic_dobBased，注意输入参数的顺序！！！切换ode的时候，交换t，x顺序
def hypersonic_dobBased(x, t,  x01, Pid, z, mu, Vv):
    dx = np.zeros(21)
    # print(x)
    # 飞行器结构参数
    Sref = 334.73
    b = 18.288
    c = 24.384
    [rho, g, Vvoice]= airpara(x[14], z, Vv, mu)
    
    # 飞行器质量和转动惯量
    m = 136080
    Ix = 1355818
    Iy = 13558180
    Iz = 13558180
    J = np.array([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])
    
    # 干扰设置
    if t >= 40:
        disMx = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
        disMy = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
        disMz = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
    else:
        disMx = 0
        disMy = 0
        disMz = 0
    
    deltaMx = -0.25
    deltaMy = -0.25
    deltaMz = -0.25
    deltaL = -0.1
    deltaD = -0.1
    deltaY = -0.1
    
    # 自设调节参数
    aaa1 = 1
    aaa2 = 1
    aaa3 = 1
    piang1 = 180 / np.pi * aaa1
    piang2 = 180 / np.pi * aaa2
    piang3 = 180 / np.pi * aaa3
    piang = 1
    piang11 = 1
    piang33 = 1
    
    an = 1
    
    # 计算攻角、侧滑角、倾侧角等
    beta = np.arcsin(
        (np.sin(x[1] - x[17]) * np.cos(x[2]) + np.sin(x[0]) * np.sin(x[2]) * np.cos(x[1] - x[17])) * np.cos(x[16])
        - np.cos(x[0]) * np.sin(x[2]) * np.sin(x[16]))
    altha = np.arcsin(
        (np.cos(x[1] - x[17]) * np.sin(x[0]) * np.cos(x[2]) * np.cos(x[16]) - np.sin(x[1] - x[17]) * np.sin(x[2]) * np.cos(x[16])
         - np.cos(x[0]) * np.cos(x[2]) * np.sin(x[16]) / (np.cos(beta))))
    gammac = np.arcsin(
        (np.cos(x[1] - x[17]) * np.sin(x[0]) * np.sin(x[2]) * np.sin(x[16]) + np.sin(x[1] - x[17]) * np.cos(x[2]) * np.sin(x[16])
         + np.cos(x[0]) * np.sin(x[2]) * np.cos(x[16]) / np.cos(beta)))
    
    althadu = altha * 180 / np.pi
    
    # 控制器增益
    KMGainsch = hypersonic_gainsch(x[0], x[2])
    
    # 干扰观测器输出
    p10 = 10
    p20 = 10
    p30 = 10
    
    dphi = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dpesai = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / (np.cos(x[0]))
    dgamma = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    
    g10 = dphi - x[6]
    dhatx = p10 * g10
    
    g20 = dpesai - x[7]
    dhaty = p20 * g20
    
    g30 = dgamma - x[8]
    dhatz = p30 * g30
    
    # 期望力矩
    UM = Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz)
    
    # 舵偏控制输入
    u = duopianjisuan(x, UM, althadu, beta, an, piang, piang1, piang2, piang3, Vvoice, rho, piang11, piang33)
    
    # 计算气动力与气动力矩
    Force, M = lijuqiujie(x, u, althadu, beta, an, piang, piang1, piang2, piang3, rho, Vvoice, piang11, piang33)
    
    D = Force[0]
    L = Force[1]
    Y = Force[2]
    Mx = M[0]
    My = M[1]
    Mz = M[2]
    
    # 姿态环动态
    dx[0] = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dx[1] = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / (np.cos(x[0]))
    dx[2] = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    dx[3] = (Iy - Iz) / Ix * x[4] * x[5] + (Mx * (1 + deltaMx) + disMx) / Ix
    dx[4] = (Iz - Ix) / Iy * x[3] * x[5] + (My * (1 + deltaMy) + disMy) / Iy
    dx[5] = (Ix - Iy) / Iz * x[3] * x[4] + (Mz * (1 + deltaMz) + disMz) / Iz
    
    # 干扰观测器动态
    dx[6] = np.sin(x[2]) / Iy * UM[1] + np.cos(x[2]) / Iz * UM[2] + dhatx
    dx[7] = np.cos(x[2]) / (Iy * np.cos(x[0])) * UM[1] - np.sin(x[2]) / (Iz * np.cos(x[0])) * UM[2] + dhaty
    dx[8] = 1 / Ix * UM[0] - np.tan(x[0]) / Iy * np.cos(x[2]) * UM[1] + np.tan(x[0]) / Iz * np.sin(x[2]) * UM[2] + dhatz
    
    # 增益调度控制器积分环节
    dx[9] = x01[0] - x[0]
    dx[10] = x01[1] - x[1]
    dx[11] = x01[2] - x[2]
    
    # 位置动态
    dx[12] = x[15] * np.cos(x[16]) * np.cos(x[17])
    dx[13] = x[15] * np.sin(x[16])
    dx[14] = -x[15] * np.sin(x[17]) * np.cos(x[16])
    
    # 速度换动态
    dx[15] = -D * (1 + deltaD) / m - g * np.sin(x[16])
    dx[16] = 1 / (m * x[15]) * (L * (1 + deltaL) * np.cos(gammac) - Y * (1 + deltaY) * np.sin(gammac)) - g * np.cos(gammac) / x[15]
    dx[17] = 1 / (m * x[15] * np.cos(x[16])) * (L * (1 + deltaL) * np.sin(gammac) + Y * (1 + deltaY) * np.cos(gammac))
    
    # print(dx)
    return dx

def hypersonic_dobBased_neural(x, t,  x01, Pid, z, mu, Vv, list):
    dx = np.zeros(21)
    # print(x)
    # 飞行器结构参数
    Sref = 334.73
    b = 18.288
    c = 24.384
    [rho, g, Vvoice]= airpara(x[14], z, Vv, mu)
    
    # 飞行器质量和转动惯量
    m = 136080
    Ix = 1355818
    Iy = 13558180
    Iz = 13558180
    J = np.array([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])
    
    # 干扰设置
    if t >= 40:
        disMx = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
        disMy = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
        disMz = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
    else:
        disMx = 0
        disMy = 0
        disMz = 0
    
    deltaMx = -0.25
    deltaMy = -0.25
    deltaMz = -0.25
    deltaL = -0.1
    deltaD = -0.1
    deltaY = -0.1
    
    # 自设调节参数
    aaa1 = 1
    aaa2 = 1
    aaa3 = 1
    piang1 = 180 / np.pi * aaa1
    piang2 = 180 / np.pi * aaa2
    piang3 = 180 / np.pi * aaa3
    piang = 1
    piang11 = 1
    piang33 = 1
    
    an = 1
    
    # 计算攻角、侧滑角、倾侧角等
    beta = np.arcsin(
        (np.sin(x[1] - x[17]) * np.cos(x[2]) + np.sin(x[0]) * np.sin(x[2]) * np.cos(x[1] - x[17])) * np.cos(x[16])
        - np.cos(x[0]) * np.sin(x[2]) * np.sin(x[16]))
    altha = np.arcsin(
        (np.cos(x[1] - x[17]) * np.sin(x[0]) * np.cos(x[2]) * np.cos(x[16]) - np.sin(x[1] - x[17]) * np.sin(x[2]) * np.cos(x[16])
         - np.cos(x[0]) * np.cos(x[2]) * np.sin(x[16]) / (np.cos(beta))))
    gammac = np.arcsin(
        (np.cos(x[1] - x[17]) * np.sin(x[0]) * np.sin(x[2]) * np.sin(x[16]) + np.sin(x[1] - x[17]) * np.cos(x[2]) * np.sin(x[16])
         + np.cos(x[0]) * np.sin(x[2]) * np.cos(x[16]) / np.cos(beta)))
    
    althadu = altha * 180 / np.pi
    
    # 控制器增益
    KMGainsch = hypersonic_gainsch(x[0], x[2])
    
    # 干扰观测器输出
    p10 = 10
    p20 = 10
    p30 = 10
    
    dphi = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dpesai = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / (np.cos(x[0]))
    dgamma = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    
    g10 = dphi - x[6]
    dhatx = p10 * g10
    
    g20 = dpesai - x[7]
    dhaty = p20 * g20
    
    g30 = dgamma - x[8]
    dhatz = p30 * g30
    
    # 期望力矩
    UM = Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz)
    
    # 舵偏控制输入
    u = duopianjisuan(x, UM, althadu, beta, an, piang, piang1, piang2, piang3, Vvoice, rho, piang11, piang33)
    
    # 计算气动力与气动力矩
    Force, M = lijuqiujie(x, u, althadu, beta, an, piang, piang1, piang2, piang3, rho, Vvoice, piang11, piang33)
    
    #将神经网络的输出值w，b放入标准气动模型中，此模型输出轨迹即为修正后轨迹
    D = Force[0]*(1+list[0][0])+list[0][1]
    L = Force[1]*(1+list[1][0])+list[1][1]
    Y = Force[2]*(1+list[2][0])+list[2][1]
    Mx = M[0]
    My = M[1]
    Mz = M[2]
    
    
    
    # 姿态环动态
    dx[0] = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dx[1] = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / (np.cos(x[0]))
    dx[2] = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    dx[3] = (Iy - Iz) / Ix * x[4] * x[5] + (Mx * (1 + deltaMx) + disMx) / Ix
    dx[4] = (Iz - Ix) / Iy * x[3] * x[5] + (My * (1 + deltaMy) + disMy) / Iy
    dx[5] = (Ix - Iy) / Iz * x[3] * x[4] + (Mz * (1 + deltaMz) + disMz) / Iz
    
    # 干扰观测器动态
    dx[6] = np.sin(x[2]) / Iy * UM[1] + np.cos(x[2]) / Iz * UM[2] + dhatx
    dx[7] = np.cos(x[2]) / (Iy * np.cos(x[0])) * UM[1] - np.sin(x[2]) / (Iz * np.cos(x[0])) * UM[2] + dhaty
    dx[8] = 1 / Ix * UM[0] - np.tan(x[0]) / Iy * np.cos(x[2]) * UM[1] + np.tan(x[0]) / Iz * np.sin(x[2]) * UM[2] + dhatz
    
    # 增益调度控制器积分环节
    dx[9] = x01[0] - x[0]
    dx[10] = x01[1] - x[1]
    dx[11] = x01[2] - x[2]
    
    # 位置动态
    dx[12] = x[15] * np.cos(x[16]) * np.cos(x[17])
    dx[13] = x[15] * np.sin(x[16])
    dx[14] = -x[15] * np.sin(x[17]) * np.cos(x[16])
    
    #dx[:,0] = x[:,4]
    
    # 速度换动态
    dx[15] = -D * (1 + deltaD) / m - g * np.sin(x[16])
    dx[16] = 1 / (m * x[15]) * (L * (1 + deltaL) * np.cos(gammac) - Y * (1 + deltaY) * np.sin(gammac)) - g * np.cos(gammac) / x[15]
    dx[17] = 1 / (m * x[15] * np.cos(x[16])) * (L * (1 + deltaL) * np.sin(gammac) + Y * (1 + deltaY) * np.cos(gammac))
    
    # print(dx)
    return dx
def hypersonic_dobBased1_numpy_forDrawing(x, t,  x01, Pid, z, mu, Vv,biases,ro_bias):
    dx = np.zeros(21)
    # print(x)
    # 飞行器结构参数
    Sref = 334.73
    b = 18.288
    c = 24.384
    [rho, g, Vvoice]= airpara(x[14], z, Vv, mu)
    
    eror = np.random.randn()*ro_bias
    rho = rho*(1+eror)
    qbar = 0.5*rho*(1+eror)*x[15]**2
    
    # 飞行器质量和转动惯量
    m = 136080
    Ix = 1355818
    Iy = 13558180
    Iz = 13558180
    J = np.array([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])
    
    # 干扰设置
    if t >= 40:
        disMx = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
        disMy = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
        disMz = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t)
    else:
        disMx = 0
        disMy = 0
        disMz = 0
    
    deltaMx = -0.25
    deltaMy = -0.25
    deltaMz = -0.25
    deltaL = -0.1
    deltaD = -0.1
    deltaY = -0.1
    
    # 自设调节参数
    aaa1 = 1
    aaa2 = 1
    aaa3 = 1
    piang1 = 180 / np.pi * aaa1
    piang2 = 180 / np.pi * aaa2
    piang3 = 180 / np.pi * aaa3
    piang = 1
    piang11 = 1
    piang33 = 1
    
    an = 1
    
    # 计算攻角、侧滑角、倾侧角等
    beta = np.arcsin(
        (np.sin(x[1] - x[17]) * np.cos(x[2]) + np.sin(x[0]) * np.sin(x[2]) * np.cos(x[1] - x[17])) * np.cos(x[16])
        - np.cos(x[0]) * np.sin(x[2]) * np.sin(x[16]))
    altha = np.arcsin(
        (np.cos(x[1] - x[17]) * np.sin(x[0]) * np.cos(x[2]) * np.cos(x[16]) - np.sin(x[1] - x[17]) * np.sin(x[2]) * np.cos(x[16])
         - np.cos(x[0]) * np.cos(x[2]) * np.sin(x[16]) / (np.cos(beta))))
    gammac = np.arcsin(
        (np.cos(x[1] - x[17]) * np.sin(x[0]) * np.sin(x[2]) * np.sin(x[16]) + np.sin(x[1] - x[17]) * np.cos(x[2]) * np.sin(x[16])
         + np.cos(x[0]) * np.sin(x[2]) * np.cos(x[16]) / np.cos(beta)))
    
    althadu = altha * 180 / np.pi
    
    # 控制器增益
    KMGainsch = hypersonic_gainsch(x[0], x[2])
    
    # 干扰观测器输出
    p10 = 10
    p20 = 10
    p30 = 10
    
    dphi = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dpesai = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / (np.cos(x[0]))
    dgamma = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    
    g10 = dphi - x[6]
    dhatx = p10 * g10
    
    g20 = dpesai - x[7]
    dhaty = p20 * g20
    
    g30 = dgamma - x[8]
    dhatz = p30 * g30
    
    # 期望力矩
    UM = Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz)
    
    # 舵偏控制输入
    u = duopianjisuan(x, UM, althadu, beta, an, piang, piang1, piang2, piang3, Vvoice, rho, piang11, piang33)
    
    # 计算气动力与气动力矩
    Force, M = lijuqiujie(x, u, althadu, beta, an, piang, piang1, piang2, piang3, rho, Vvoice, piang11, piang33)
    
    D = (Force[0]*(1+biases[0][0])+biases[0][1]) 
    L = (Force[1]*(1+biases[1][0])+biases[1][1])
    Y = (Force[2]*(1+biases[2][0])+biases[2][1])
    Mx = M[0]
    My = M[1]
    Mz = M[2]
    
    # 姿态环动态
    dx[0] = x[4] * np.sin(x[2]) + x[5] * np.cos(x[2])
    dx[1] = (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2])) / (np.cos(x[0]))
    dx[2] = x[3] - np.tan(x[0]) * (x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
    dx[3] = (Iy - Iz) / Ix * x[4] * x[5] + (Mx * (1 + deltaMx) + disMx) / Ix
    dx[4] = (Iz - Ix) / Iy * x[3] * x[5] + (My * (1 + deltaMy) + disMy) / Iy
    dx[5] = (Ix - Iy) / Iz * x[3] * x[4] + (Mz * (1 + deltaMz) + disMz) / Iz
    
    # 干扰观测器动态
    dx[6] = np.sin(x[2]) / Iy * UM[1] + np.cos(x[2]) / Iz * UM[2] + dhatx
    dx[7] = np.cos(x[2]) / (Iy * np.cos(x[0])) * UM[1] - np.sin(x[2]) / (Iz * np.cos(x[0])) * UM[2] + dhaty
    dx[8] = 1 / Ix * UM[0] - np.tan(x[0]) / Iy * np.cos(x[2]) * UM[1] + np.tan(x[0]) / Iz * np.sin(x[2]) * UM[2] + dhatz
    
    # 增益调度控制器积分环节
    dx[9] = x01[0] - x[0]
    dx[10] = x01[1] - x[1]
    dx[11] = x01[2] - x[2]
    
    # 位置动态
    dx[12] = x[15] * np.cos(x[16]) * np.cos(x[17])
    dx[13] = x[15] * np.sin(x[16])
    dx[14] = -x[15] * np.sin(x[17]) * np.cos(x[16])
    
    # 速度换动态
    dx[15] = -D * (1 + deltaD) / m - g * np.sin(x[16])
    dx[16] = 1 / (m * x[15]) * (L * (1 + deltaL) * np.cos(gammac) - Y * (1 + deltaY) * np.sin(gammac)) - g * np.cos(gammac) / x[15]
    dx[17] = 1 / (m * x[15] * np.cos(x[16])) * (L * (1 + deltaL) * np.sin(gammac) + Y * (1 + deltaY) * np.cos(gammac))
    
    # print(dx)
    return dx
# 定义常数和初始状态
x0 = np.array([6/57.3, 1/57.3, 2/57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33589, 0, 5000, 0, 0, 0.03, 0, 0])
#1，
Wgd = 80
tf = 80
step = 0.1
tspan = np.arange(0, tf + step, step)

# atol = 1e-5
# rtol = 1e-4
# 1.调用odeint进行仿真
x = odeint(hypersonic_dobBased,  x0, tspan , args=( x01, Pid, z, mu, Vv),atol=1e-7,rtol=1e-6)
# 2.龙格库塔法
# solution = solve_ivp(hypersonic_dobBased,  (0,80), x0 , method='RK45', args=( x01, Pid, z, mu, Vv))
# x = solution.y
# x = x.T
# print(x)
# 3.欧拉法
# x = np.zeros([len(tspan),len(x0)])
# x[0,:] = x0
# for i in range(1,len(tspan)):
#     t = tspan[i]
#     dx = hypersonic_dobBased(t, x[i-1,:], x01, Pid, z, mu, Vv)
#     x[i,:] = x[i-1,:] + step * dx
    
    
deltaa = np.zeros(len(tspan))
deltae = np.zeros(len(tspan))
deltar = np.zeros(len(tspan))
althavector = np.zeros(len(tspan))
betavector = np.zeros(len(tspan))
gammacvector = np.zeros(len(tspan))
CD_list = np.zeros(len(tspan))
CY_list = np.zeros(len(tspan))
CL_list = np.zeros(len(tspan))

# deltaa_pid = np.zeros(len(tspan))
# deltae_pid = np.zeros(len(tspan))
# deltar_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))

# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))
# althavector_pid = np.zeros(len(tspan))


Ix = 1355818
Iy = 13558180
Iz = 13558180

J = np.array([[Ix, 0, 0],
              [0, Iy, 0],
              [0, 0, Iz]])



for i in range(len(tspan)):
    Sref = 334.73
    b = 18.288
    c = 24.384

    m = 136080
    an = 1
    aaa1 = 1
    aaa2 = 1
    aaa3 = 1
    piang1 = 180 / np.pi * aaa1
    piang2 = 180 / np.pi * aaa2
    piang3 = 180 / np.pi * aaa3
    piang = 1  # 180/pi;
    piang11 = 1  # 180/pi;
    piang33 = 1  # 180/pi;

    rho, g, Vvoice = airpara(x[i, 14], z, Vv, mu)
    
    # print(rho,g,Vvoice)
    KMGainsch = hypersonic_gainsch(x[i, 0], x[i, 2])

    betavector[i] = np.arcsin((np.sin(x[i, 1] - x[i, 17]) * np.cos(x[i, 2]) + np.sin(x[i, 0]) * np.sin(x[i, 2]) * np.cos(x[i, 1] - x[i, 17])) * np.cos(x[i, 16]) - np.cos(x[i, 0]) * np.sin(x[i, 2]) * np.sin(x[i, 16]))
    althavector[i] = np.arcsin((np.cos(x[i, 1] - x[i, 17]) * np.sin(x[i, 0]) * np.cos(x[i, 2]) * np.cos(x[i, 16]) - np.sin(x[i, 1] - x[i, 17]) * np.sin(x[i, 2]) * np.cos(x[i, 16]) - np.cos(x[i, 0]) * np.cos(x[i, 2]) * np.sin(x[i, 16])) / np.cos(betavector[i]))
    gammacvector[i] = np.arcsin((np.cos(x[i, 1] - x[i, 17]) * np.sin(x[i, 0]) * np.sin(x[i, 2]) * np.sin(x[i, 16]) + np.sin(x[i, 1] - x[i, 17]) * np.cos(x[i, 2]) * np.sin(x[i, 16]) + np.cos(x[i, 0]) * np.sin(x[i, 2]) * np.cos(x[i, 16])) / np.cos(betavector[i]))
    
    altha = althavector[i] * 180 / np.pi
    beta = betavector[i]
    gamma = gammacvector[i]
    # Rest of the code goes here...

    # You can continue to translate the rest of your MATLAB code to Python.

    dx1 = x[i, 4] * np.sin(x[i, 2]) + x[i, 5] * np.cos(x[i, 2])
    dx2 = (x[i, 4] * np.cos(x[i, 2]) - x[i, 5] * np.sin(x[i, 2])) / np.cos(x[i, 0])
    dx3 = x[i, 3] - np.tan(x[i, 0]) * (x[i, 4] * np.cos(x[i, 2]) - x[i, 5] * np.sin(x[i, 2]))

    p10 = 10
    p20 = 10
    p30 = 10
    g10 = dx1 - x[i, 6]
    dhatx = p10 * g10
    g20 = dx2 - x[i, 7]
    dhaty = p20 * g20
    g30 = dx3 - x[i, 8]
    dhatz = p30 * g30

    Px1 = KMGainsch[0]
    Px2 = KMGainsch[1]
    Px3 = KMGainsch[2]
    Py1 = KMGainsch[3]
    Py2 = KMGainsch[4]
    Py3 = KMGainsch[5]
    Pz1 = KMGainsch[6]
    Pz2 = KMGainsch[7]
    Pz3 = KMGainsch[8]

    PPP=np.array([[Px1,Px2,Px3],
    [Py1, Py2 ,Py3],
    [Pz1 ,Pz2, Pz3]])

    kpv1 = Pid[0, 0]
    kiv1 = Pid[0, 1]
    kdv1 = Pid[0, 2]
    kpv2 = Pid[1, 0]
    kiv2 = Pid[1, 1]
    kdv2 = Pid[1, 2]
    kpv3 = Pid[2, 0]
    kiv3 = Pid[2, 1]
    kdv3 = Pid[2, 2]

    Uu = np.array([Px1 * kiv1 * x[i, 9] + Px1 * kpv1 * (x01[0] - x[i, 0]) + Px1 * kdv1 * (-dx1) + Px2 * kiv2 * x[i, 10] + Px2 * kpv2 * (x01[1] - x[i, 1]) + Px2 * kdv2 * (-dx2) + Px3 * kiv3 * x[i, 11] + Px3 * kpv3 * (x01[2] - x[i, 2]) + Px3 * kdv3 * (-dx3),
                   Py1 * kiv1 * x[i, 9] + Py1 * kpv1 * (x01[0] - x[i, 0]) + Py1 * kdv1 * (-dx1) + Py2 * kiv2 * x[i, 10] + Py2 * kpv2 * (x01[1] - x[i, 1]) + Py2 * kdv2 * (-dx2) + Py3 * kiv3 * x[i, 11] + Py3 * kpv3 * (x01[2] - x[i, 2]) + Py3 * kdv3 * (-dx3),
                   Pz1 * kiv1 * x[i, 9] + Pz1 * kpv1 * (x01[0] - x[i, 0]) + Pz1 * kdv1 * (-dx1) + Pz2 * kiv2 * x[i, 10] + Pz2 * kpv2 * (x01[1] - x[i, 1]) + Pz2 * kdv2 * (-dx2) + Pz3 * kiv3 * x[i, 11] + Pz3 * kpv3 * (x01[2] - x[i, 2]) + Pz3 * kdv3 * (-dx3)])

    UM = np.dot(J, Uu) - np.dot(np.dot(J, PPP), np.array([dhatx, dhaty, dhatz]))

    x4 = x[i, 15]
    qbar = 0.5 * rho * x4 ** 2
    wx = x[i, 3]
    wy = x[i, 4]
    wz = x[i, 5]
    Ma = x4 / Vvoice

    pppi = 1

    # Rest of the code goes here...

    # You can continue to translate the rest of your MATLAB code to Python.
    CMx_beta = -1.402 * 0.1 + 3.326 * 0.01 * Ma - 7.59e-4 * altha + 8.596e-6 * (altha * Ma) - 3.794e-3 * Ma**2 + 2.354e-6 * altha**2 - 1.044e-8 * (altha * Ma)**2 + 2.219e-4 * Ma**3 - 8.964e-18 * altha**3 - 6.462e-6 * Ma**4 + 3.803e-19 * altha**4 + 7.419e-8 * Ma**5 - 3.353e-21 * altha**5
    CMx_deltaaltha_con = 3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 4.95e-6 * altha**2 + 1.411e-6 * Ma**2
    CMx_deltaaltha_u1 = 1.17e-4 * piang1 + 2.794e-8 * altha * Ma * piang1

    CMx_deltae_con = -(3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 4.95e-6 * altha**2 + 1.411e-6 * Ma**2)
    CMx_deltae_u3 = -(1.17e-4 * piang3 + 2.794e-8 * altha * Ma * piang3)

    CMx_deltar_con = -5.0103e-19 + 6.2723e-20 * altha + 2.3418e-20 * Ma - 3.4201e-21 * altha * Ma
    CMx_deltar_u2 = -3.5496e-6 * Ma * piang2 + 5.5547e-8 * altha * Ma * piang2 + 1.1441e-4 * piang2 - 2.6824e-6 * altha * piang2

    CMx_r = 3.82 * 0.1 - 1.06 * 0.1 * Ma + 1.94e-3 * altha - 8.15e-5 * altha * Ma + 1.45 * 0.01 * Ma**2 - 9.76e-6 * altha**2 + 4.49e-8 * (altha * Ma)**2 - 1.02e-3 * Ma**3 - 2.7e-7 * altha**3 + 3.56e-5 * Ma**4 + 3.19e-8 * altha**4 - 4.81e-7 * Ma**5 - 1.06e-9 * altha**5
    CMx_p = -2.99 * 0.1 + 7.47 * 0.01 * Ma + 1.38e-3 * altha - 8.78e-5 * altha * Ma - 9.13e-3 * Ma**2 - 2.04e-4 * altha**2 - 1.52e-7 * (altha * Ma)**2 + 5.73e-4 * Ma**3 - 3.86e-5 * altha**3 - 1.79e-5 * Ma**4 + 4.21e-6 * altha**4 + 2.2e-7 * Ma**5 - 1.15e-7 * altha**5

    CMx_con = CMx_beta * beta + CMx_deltaaltha_con + CMx_deltae_con + CMx_deltar_con + CMx_r * wz * b / (2 * x4) + CMx_p * wx * b / (2 * x4)

    # Pitch moment linear coefficients
    CMz_altha = -2.192e-2 + 7.739e-3 * Ma - 2.26e-3 * altha + 1.808e-4 * Ma * altha - 8.849e-4 * Ma**2 + 2.616e-4 * altha**2 - 2.88e-7 * (Ma * altha)**2 + 4.617e-5 * Ma**3 - 7.887e-5 * altha**3 - 1.143e-6 * Ma**4 + 8.288e-6 * altha**4 + 1.082e-8 * Ma**5 - 2.789e-7 * altha**5
    CMz_deltaaltha_con = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma - 4.46e-6 * altha * Ma
    CMz_deltaaltha_u1 = 2.89e-4 * piang1 + 4.48e-6 * altha * piang1 - 5.87e-6 * Ma * piang1 + 9.72e-8 * altha * Ma * piang1

    CMz_deltae_con = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma - 4.46e-6 * altha * Ma
    CMz_deltae_u3 = 2.89e-4 * piang3 + 4.48e-6 * altha * piang3 - 5.87e-6 * Ma * piang3 + 9.72e-8 * altha * Ma * piang3

    CMz_deltar = -2.79e-5 * altha - 5.89e-8 * altha**2 + 1.58e-3 * Ma**2 + 6.42e-8 * altha**3 - 6.69e-4 * Ma**3 - 2.1e-8 * altha**4 + 1.05e-4 * Ma**4 + 3.14e-9 * altha**5 - 7.74e-6 * Ma**5 - 2.18e-10 * altha**6 + 2.7e-7 * Ma**6 + 5.74e-12 * altha**7 - 3.58e-9 * Ma**7
    CMz_deltac = 0
    CMz_q = -1.36 + 3.86e-1 * Ma + 7.85e-4 * altha + 1.4e-4 * altha * Ma - 5.42e-2 * Ma**2 + 2.36e-3 * altha**2 - 1.95e-6 * (altha * Ma)**2 + 3.8e-3 * Ma**3 - 1.48e-3 * altha**3 - 1.3e-4 * Ma**4 + 1.69e-4 * altha**4 + 1.71e-6 * Ma**5 - 5.93e-6 * altha**5
    CMz_con = CMz_altha + CMz_deltaaltha_con + CMz_deltae_con + CMz_deltar + CMz_deltac + CMz_q * wy * c / (2 * x4)
    
    # print(CMz_con)
    # Yaw moment linear coefficients
    CMy_beta = 6.998e-4 * altha + 5.9115 * 0.01 * Ma - 7.525e-5 * Ma * altha + 2.516e-4 * altha**2 - 1.4824 * 0.01 * Ma**2 - 2.1924e-7 * (Ma * altha)**2 - 1.0777e-4 * altha**3 + 1.2692e-3 * Ma**3 + 1.0707e-8 * (Ma * altha)**3 + 9.4989e-6 * altha**4 - 4.7098e-5 * Ma**4 - 5.5472e-11 * (Ma * altha)**4 - 2.5953e-7 * altha**5 + 6.4284e-7 * Ma**5 + 8.5863e-14 * (Ma * altha)**5
    CMy_deltaaltha_con = 2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 6.39e-7 * altha**2 + 8.16e-7 * Ma**2
    CMy_deltaaltha_u1 = -1.3e-5 * piang1 - 8.93e-8 * altha * Ma * piang1
    CMy_deltae_con = -(2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 6.39e-7 * altha**2 + 8.16e-7 * Ma**2)
    CMy_deltae_u3 = 1.3e-5 * piang3 + 8.93e-8 * altha * Ma * piang3
    CMy_deltar_con = 2.85e-18 - 3.59e-19 * altha - 1.26e-19 * Ma + 1.57e-20 * (altha * Ma)
    CMy_deltar_u2 = -5.28e-4 * piang2 + 1.39e-5 * altha * piang2 + 1.65e-5 * (Ma * piang2) - 3.13e-7 * (altha * Ma) * piang2

    CMy_p = 3.68 * 0.1 - 9.79e-2 * Ma + 7.61e-16 * altha + 1.24e-2 * Ma**2 - 4.64e-16 * altha**2 - 8.05e-4 * Ma**3 + 1.01e-16 * altha**3 + 2.57e-5 * Ma**4 - 9.18e-18 * altha**4 - 3.2e-7 * Ma**5 + 2.96e-19 * altha**5
    CMy_r = -2.41 + 5.96e-1 * Ma - 2.74e-3 * altha + 2.09e-4 * (altha * Ma) - 7.57e-2 * Ma**2 + 1.15e-3 * altha**2 - 6.53e-8 * (altha * Ma)**2 + 4.9e-3 * Ma**3 - 3.87e-4 * altha**3 - 1.57e-4 * Ma**4 + 3.6e-5 * altha**4 + 1.96e-6 * Ma**5 - 1.18e-6 * altha**5

    CMy_con = CMy_beta * beta + CMy_deltaaltha_con + CMy_deltae_con + CMy_deltar_con + CMy_p * wx * b / (2 * x4) + CMy_r * wz * b / (2 * x4)

    
    
    # Solving for control deflections
    ACM = np.array([[1.0 / (qbar * Sref), 0.0, 0.0],
                    [0.0, 1.0 / (qbar * b * Sref), 0.0],
                    [0.0, 0.0, 1.0 / (qbar * c * Sref)]])

    ACMcon = np.array([an * CMx_con, an * CMy_con, an * CMz_con])

    ACMu = an * np.array([[CMx_deltaaltha_u1, CMx_deltar_u2, CMx_deltae_u3],
                        [CMy_deltaaltha_u1, CMy_deltar_u2, CMy_deltae_u3],
                        [CMz_deltaaltha_u1, 0, CMz_deltae_u3]])

    I = np.eye(3)

    u = np.linalg.solve(ACMu, np.dot(ACM, UM) - ACMcon)
    # u = np.linalg.inv(ACMu).dot(np.linalg.inv(ACM).dot(UM) - ACMcon)
    ubar = 25 / 57.3

    for ii in range(3):
        if u[ii] > ubar:
            u[ii] = ubar
        elif u[ii] < -ubar:
            u[ii] = -ubar

    # # 计算 u
    # I = np.identity(3)
    
    # ubar = 25 / 57.3  # 这是u的上限

    # # 将u截断为-u_bar和u_bar之间
    # u = np.clip(u, -ubar, ubar)

    # 存储控制输入到deltaa、deltae和deltar
    deltaa[i] = u[0]
    deltae[i] = u[2]
    deltar[i] = u[1]
    
    CD_altha = 8.717e-2 - 3.307 * 0.01 * Ma + 3.179 * 0.001 * altha - 1.25 * 0.0001 * altha * Ma + 5.036 * 0.001 * Ma**2 - 1.1 * 0.001 * altha**2 + 1.405e-7 * (altha * Ma)**2 - 3.658e-4 * Ma**3 + 3.175e-4 * altha**3 + 1.274e-5 * Ma**4 - 2.985e-5 * altha**4 - 1.705e-7 * Ma**5 + 9.766e-7 * altha**5
    CD_deltaaltha = 4.5548e-4 + 2.5411e-5 * altha - 1.1436e-4 * Ma - 3.6417e-5 * (piang1 * u[0]) - 5.3015e-7 * (altha * Ma) * (piang1 * u[0]) + 3.2187e-6 * altha**2 + 3.014e-6 * Ma**2 + 6.9629e-6 * (piang1 * u[0])**2 + 2.1026e-12 * (altha * Ma * (piang1 * u[0]))**2
    CD_deltae = 4.5548e-4 + 2.5411e-5 * altha - 1.1436e-4 * Ma - 3.6417e-5 * (piang3 * u[2]) - 5.3015e-7 * (altha * Ma) * (piang3 * u[2]) + 3.2187e-6 * altha**2 + 3.014e-6 * Ma**2 + 6.9629e-6 * (piang3 * u[2])**2 + 2.1026e-12 * (altha * Ma * (piang3 * u[2]))**2
    CD_deltar = 7.5e-4 - 2.29e-5 * altha - 9.69e-5 * Ma - 1.83e-6 * piang2 * u[1] + 9.13e-9 * (altha * Ma) * piang2 * u[1] + 8.76e-7 * altha**2 + 2.7e-6 * Ma**2 + 1.97e-6 * (piang2 * u[1])**2 - 1.7702e-11 * (altha * Ma * piang2 * u[1])**2
    CD_deltac = 0
    CD = CD_altha + CD_deltaaltha + CD_deltae + CD_deltar + CD_deltac
    D = qbar * Sref * CD

    
    
    CL_altha = -8.19e-2 + 4.7 * 0.01 * Ma + 1.86 * 0.01 * altha - 4.73e-4 * altha * Ma - 9.19e-3 * Ma**2 - 1.52e-4 * altha**2 + 5.99e-7 * (altha * Ma)**2 + 7.74e-4 * Ma**3 + 4.08e-6 * altha**3 - 2.93e-5 * Ma**4 - 3.91e-7 * altha**4 + 4.12e-7 * Ma**5 + 1.3e-8 * altha**5
    CL_deltaaltha = -1.45e-5 + 1.01e-4 * altha + 7.1e-6 * Ma - 4.14e-4 * (piang1 * u[0]) - 3.51e-6 * altha * (piang1 * u[0]) + 4.7e-6 * (altha * Ma) + 8.72e-6 * Ma * (piang1 * u[0]) - 1.7e-7 * altha * Ma * (piang1 * u[0])
    CL_deltae = -1.45e-5 + 1.01e-4 * altha + 7.1e-6 * Ma - 4.14e-4 * (piang3 * u[2]) - 3.51e-6 * altha * (piang3 * u[2]) + 4.7e-6 * (altha * Ma) + 8.72e-6 * Ma * (piang3 * u[2]) - 1.7e-7 * altha * Ma * (piang3 * u[2])
    CL_deltac = 0
    CL = CL_altha + CL_deltaaltha + CL_deltae + CL_deltac
    L = qbar * Sref * CL

    CY_deltaaltha = -1.02e-6 - 1.12e-7 * altha + 4.48e-7 * Ma + 2.27e-7 * (piang1 * u[0]) + 4.11e-9 * altha * Ma * (piang1 * u[0]) + 2.82e-9 * altha**2 - 2.36e-8 * Ma**2 - 5.04e-8 * (piang1 * u[0])**2 + 4.5e-14 * (altha * Ma * (piang1 * u[0]))**2
    CY_deltae = -(-1.02e-6 - 1.12e-7 * altha + 4.48e-7 * Ma + 2.27e-7 * (piang3 * u[2]) + 4.11e-9 * altha * Ma * (piang3 * u[2]) + 2.82e-9 * altha**2 - 2.36e-8 * Ma**2 - 5.04e-8 * (piang3 * u[2])**2 + 4.5e-14 * (altha * Ma * (piang3 * u[2]))**2)
    CY_deltar = -1.43e-18 + 4.86e-20 * altha + 1.86e-19 * Ma + 3.84e-4 * piang2 * u[1] - 1.17e-5 * altha * piang2 * u[1] - 1.07e-5 * Ma * piang2 * u[1] + 2.6e-7 * altha * Ma * piang2 * u[1]
    CY_beta = 2.8803e-3 * altha - 2.8943e-4 * altha * Ma + 5.4822e-2 * Ma**2 + 7.3535e-4 * altha**2 - 4.6409e-9 * ((altha * Ma**2)**2) - 2.0675e-8 * ((altha**2 * Ma)**2) + 4.6205e-6 * ((altha * Ma)**2) + 2.6144e-11 * (altha**2 * Ma**2)**2 - 4.3203e-3 * Ma**3 - 3.7405e-4 * altha**3 + Ma**4 * 1.5495e-4 + 2.8183e-5 * altha**4 + Ma**5 * -2.0829e-6 + altha**5 * -5.2083e-7
    CY = CY_beta * beta + CY_deltaaltha + CY_deltae + CY_deltar
    Y = qbar * Sref * CY
    
    CD_list[i] = CD
    CY_list[i] = CY  
    CL_list[i] = CL
    # print(deltaa)
    
    
    
    

t = np.arange(0, tf + step, step)


# plt.figure(1)
# # plt.plot(t, 180/np.pi*deltae, 'k', label='x', linewidth=2)
# # 
# #print(x[0:20,14])
# plt.plot(t,  x[:,12], 'b', label='X位置', linewidth=2)
# # plt.legend(['俯仰角'])
# plt.plot(t,  x[:,13], 'r', label='Y位置', linewidth=2)
# # plt.legend(['偏航角'])
# # plt.plot(t, 180 / np.pi * x[:,14], 'y', label='Z位置', linewidth=2)
# plt.plot(t, x[:,14], 'y', label='Z位置', linewidth=2)
# # plt.legend(['滚转角'])
# plt.xlabel('t (sec)')
# plt.ylabel('位置')
# plt.legend() 
# plt.grid()  # 添加网格线
# plt.show()