import numpy as np
from copy import copy
from torch.utils.data import Dataset, DataLoader
import cProfile
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import signal
import airparameter
import torchbnn as bnn
import sys
from hypersonic import *
# np.concatenate((data0, data1, data2, data3,data4), 0)  # ,data2,data3,data4,data4,data5,data5,data4,data3

Pid = torch.Tensor([[7, 0.4, 4],
                    [7.5, 0.4, 4.4],
                    [7, 0.2, 5.4]])
z = torch.Tensor(airparameter.z)
mu = torch.Tensor(airparameter.mu)
Vv = torch.Tensor(airparameter.Vv)
device = torch.device("cpu")


def euler_method(hypersonic_dobBased, x, tspan, x01, Pid, z, mu, Vv):
    num_time_points = len(tspan)

    for i in range(1, num_time_points):
        dt = tspan[i] - tspan[i - 1]
        x = x + dt * hypersonic_dobBased(x, tspan[i - 1], x01, Pid, z, mu, Vv)

    return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1407)

def hypersonic_dobBased1(x, t, x01, Pid, z, mu, Vv, biases,ro_bias):
    # dx = torch.zeros(21)
    # print(x)
    # 飞行器结构参数
    Sref = 334.73
    b = 18.288
    c = 24.384

    rho, g, Vvoice = airpara1(x[:, 14], z, Vv, mu)
    rho = rho.unsqueeze(-1)
    g = g.unsqueeze(-1)
    Vvoice = Vvoice.unsqueeze(-1)
    eror = torch.randn_like(rho)*ro_bias
    qbar = 0.5 * rho*(1+eror) * (x[:, 15]).unsqueeze(-1) ** 2
    # print(rho.shape)
    # print(qbar.shape)
    # 使用 (x[:, 15]).unsqueeze(-1) 可以确保结果是一个列向量，即使在 x[:, 15] 已经是列向量的情况下也不会有问题。

    m = 136080.
    Ix = 1355818.
    Iy = 13558180.
    Iz = 13558180.

    J = torch.tensor([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])  # J是一个二维矩阵

    disMx = torch.zeros((x.shape[0], 1))
    disMy = torch.zeros((x.shape[0], 1))
    disMz = torch.zeros((x.shape[0], 1))
    # 干扰设置
    for i in range(len(t)):
        if t[i] >= 40:
            disMx[i] = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t[i])
            disMy[i] = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t[i])
            disMz[i] = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t[i])
        else:
            disMx[i] = 0
            disMy[i] = 0
            disMz[i] = 0

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
    piang1 = 180 / torch.pi * aaa1
    piang2 = 180 / torch.pi * aaa2
    piang3 = 180 / torch.pi * aaa3
    piang = 1
    piang11 = 1
    piang33 = 1

    an = 1

    # 计算攻角、侧滑角、倾侧角等
    # beta = torch.arcsin(
    #     (torch.sin(x[1] - x[17]) * torch.cos(x[2]) + torch.sin(x[0]) * torch.sin(x[2]) * torch.cos(x[1] - x[17])) * torch.cos(x[16])
    #     - torch.cos(x[0]) * torch.sin(x[2]) * torch.sin(x[16]))
    # altha = torch.arcsin(
    #     (torch.cos(x[1] - x[17]) * torch.sin(x[0]) * torch.cos(x[2]) * torch.cos(x[16]) - torch.sin(x[1] - x[17]) * torch.sin(x[2]) * torch.cos(x[16])
    #      - torch.cos(x[0]) * torch.cos(x[2]) * torch.sin(x[16]) / (torch.cos(beta))))
    # gammac = torch.arcsin(
    #     (torch.cos(x[1] - x[17]) * torch.sin(x[0]) * torch.sin(x[2]) * torch.sin(x[16]) + torch.sin(x[1] - x[17]) * torch.cos(x[2]) * torch.sin(x[16])
    #      + torch.cos(x[0]) * torch.sin(x[2]) * torch.cos(x[16]) / torch.cos(beta)))

    beta = torch.arcsin(
        (torch.sin(x[:, 1] - x[:, 17]) * torch.cos(x[:, 2]) + torch.sin(x[:, 0]) * torch.sin(x[:, 2]) * torch.cos(
            x[:, 1] - x[:, 17])) * torch.cos(x[:, 16])
        - torch.cos(x[:, 0]) * torch.sin(x[:, 2]) * torch.sin(x[:, 16]))
    altha = torch.arcsin(
        (torch.cos(x[:, 1] - x[:, 17]) * torch.sin(x[:, 0]) * torch.cos(x[:, 2]) * torch.cos(x[:, 16]) - torch.sin(
            x[:, 1] - x[:, 17]) * torch.sin(x[:, 2]) * torch.cos(x[:, 16])
         - torch.cos(x[:, 0]) * torch.cos(x[:, 2]) * torch.sin(x[:, 16]) / (torch.cos(beta))))
    gammac = torch.arcsin(
        (torch.cos(x[:, 1] - x[:, 17]) * torch.sin(x[:, 0]) * torch.sin(x[:, 2]) * torch.sin(x[:, 16]) + torch.sin(
            x[:, 1] - x[:, 17]) * torch.cos(x[:, 2]) * torch.sin(x[:, 16])
         + torch.cos(x[:, 0]) * torch.sin(x[:, 2]) * torch.cos(x[:, 16]) / torch.cos(beta)))

    althadu = altha * 180 / torch.pi
    # print("beta==",beta.shape)   #(8000,)

    # 控制器增益
    KMGainsch = hypersonic_gainsch(x[:, 0], x[:, 2])

    # 干扰观测器输出
    p10 = 10
    p20 = 10
    p30 = 10

    # dphi = x[4] * torch.sin(x[2]) + x[5] * torch.cos(x[2])
    # dpesai = (x[4] * torch.cos(x[2]) - x[5] * torch.sin(x[2])) / (torch.cos(x[0]))
    # dgamma = x[3] - torch.tan(x[0]) * (x[4] * torch.cos(x[2]) - x[5] * torch.sin(x[2]))

    dphi = x[:, 4] * torch.sin(x[:, 2]) + x[:, 5] * torch.cos(x[:, 2])
    dpesai = (x[:, 4] * torch.cos(x[:, 2]) - x[:, 5] * torch.sin(x[:, 2])) / (torch.cos(x[:, 0]))
    dgamma = x[:, 3] - torch.tan(x[:, 0]) * (x[:, 4] * torch.cos(x[:, 2]) - x[:, 5] * torch.sin(x[:, 2]))

    # print('dphi=',dphi.shape)

    g10 = dphi - x[:, 6]
    dhatx = p10 * g10
    # print("dhatx",dhatx.shape)  #（8001，）

    g20 = dpesai - x[:, 7]
    dhaty = p20 * g20

    g30 = dgamma - x[:, 8]
    dhatz = p30 * g30

    # 期望力矩
    UM = Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz)

    # print("UM",UM.shape)  #UM (8001,3)
    # 舵偏控制输入，输出(8000，3)
    u = duopianjisuan(x, UM, althadu, beta, an, piang, piang1, piang2, piang3, Vvoice, rho, piang11, piang33)

    # 计算气动力与气动力矩,注意M 为torch.Size([8001, 3, 1])
    Force, M = lijuqiujie(x, u, althadu, beta, an, piang, piang1, piang2, piang3, rho, Vvoice, piang11, piang33)

    # print("M",M.shape)

    biases = torch.Tensor(biases)

    # print("force",Force.shape)

    D = qbar * Sref * (Force[:, 0].unsqueeze(-1) + Force[:, 0].unsqueeze(-1) * biases[0] + biases[1])  # 轴向力
    L = qbar * Sref * (Force[:, 1].unsqueeze(-1) + Force[:, 1].unsqueeze(-1) * biases[2] + biases[3])  # 法向力
    Y = qbar * Sref * (Force[:, 2].unsqueeze(-1) + Force[:, 2].unsqueeze(-1) * biases[4] + biases[5])  # 侧向力

    Mx = M[:, 0]
    My = M[:, 1]
    Mz = M[:, 2]

    # print("Mx",Mx.shape,"D",D.shape)  #Mx是（8001，1）

    dx0 = x[:, 4].unsqueeze(-1) * torch.sin(x[:, 2].unsqueeze(-1)) + x[:, 5].unsqueeze(-1) * torch.cos(
        x[:, 2].unsqueeze(-1))
    dx1 = (x[:, 4].unsqueeze(-1) * torch.cos(x[:, 2].unsqueeze(-1)) - x[:, 5].unsqueeze(-1) * torch.sin(
        x[:, 2].unsqueeze(-1))) / torch.cos(x[:, 0].unsqueeze(-1))
    dx2 = x[:, 3].unsqueeze(-1) - torch.tan(x[:, 0].unsqueeze(-1)) * (
                x[:, 4].unsqueeze(-1) * torch.cos(x[:, 2].unsqueeze(-1)) - x[:, 5].unsqueeze(-1) * torch.sin(
            x[:, 2].unsqueeze(-1)))
    dx3 = ((Iy - Iz) / Ix) * x[:, 4].unsqueeze(-1) * x[:, 5].unsqueeze(-1) + (Mx * (1 + deltaMx) + disMx) / Ix
    dx4 = ((Iz - Ix) / Iy) * x[:, 3].unsqueeze(-1) * x[:, 5].unsqueeze(-1) + (My * (1 + deltaMy) + disMy) / Iy
    dx5 = ((Ix - Iy) / Iz) * x[:, 3].unsqueeze(-1) * x[:, 4].unsqueeze(-1) + (Mz * (1 + deltaMz) + disMz) / Iz

    dx6 = torch.sin(x[:, 2].unsqueeze(-1)) / Iy * UM[:, 1].unsqueeze(-1) + torch.cos(x[:, 2].unsqueeze(-1)) / Iz * UM[:,
                                                                                                                   2].unsqueeze(
        -1) + dhatx.unsqueeze(-1)
    dx7 = torch.cos(x[:, 2].unsqueeze(-1)) / (Iy * torch.cos(x[:, 0].unsqueeze(-1))) * UM[:, 1].unsqueeze(
        -1) - torch.sin(x[:, 2].unsqueeze(-1)) / (Iz * torch.cos(x[:, 0].unsqueeze(-1))) * UM[:, 2].unsqueeze(
        -1) + dhaty.unsqueeze(-1)
    dx8 = 1 / Ix * UM[:, 0].unsqueeze(-1) - torch.tan(x[:, 0].unsqueeze(-1)) / Iy * torch.cos(
        x[:, 2].unsqueeze(-1)) * UM[:, 1].unsqueeze(-1) + torch.tan(x[:, 0].unsqueeze(-1)) / Iz * torch.sin(
        x[:, 2].unsqueeze(-1)) * UM[:, 2].unsqueeze(-1) + dhatz.unsqueeze(-1)

    dx9 = x01[0] - x[:, 0].unsqueeze(-1)
    dx10 = x01[1] - x[:, 1].unsqueeze(-1)
    dx11 = x01[2] - x[:, 2].unsqueeze(-1)

    dx12 = x[:, 15].unsqueeze(-1) * torch.cos(x[:, 16].unsqueeze(-1)) * torch.cos(x[:, 17].unsqueeze(-1))
    dx13 = x[:, 15].unsqueeze(-1) * torch.sin(x[:, 16].unsqueeze(-1))
    dx14 = -x[:, 15].unsqueeze(-1) * torch.sin(x[:, 17].unsqueeze(-1)) * torch.cos(x[:, 16].unsqueeze(-1))

    dx15 = -D * (1 + deltaD) / m - g * torch.sin(x[:, 16].unsqueeze(-1))
    dx16 = 1 / (m * x[:, 15].unsqueeze(-1)) * (
                L * (1 + deltaL) * torch.cos(gammac.unsqueeze(-1)) - Y * (1 + deltaY) * torch.sin(
            gammac.unsqueeze(-1))) - g * torch.cos(gammac.unsqueeze(-1)) / x[:, 15].unsqueeze(-1)
    dx17 = 1 / (m * x[:, 15].unsqueeze(-1) * torch.cos(x[:, 16].unsqueeze(-1))) * (
                L * (1 + deltaL) * torch.sin(gammac.unsqueeze(-1)) + Y * (1 + deltaY) * torch.cos(gammac.unsqueeze(-1)))

    # print("dx9",dx9.shape,"dx12",dx12.shape,"dx15",dx15.shape)

    dx = torch.stack([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12, dx13, dx14, dx15, dx16, dx17],
                     dim=1)
    # print("dx",dx.shape)   #dx的形状是 torch.Size([8001, 18, 1])
    return Force, M, dx

def hypersonic_dobBased2(x, t, x01, Pid, z, mu, Vv, Force, M):
    # 传进来的force是8000*3
    # dx = torch.zeros(21)
    # print(x)
    # 飞行器结构参数
    Sref = 334.73
    b = 18.288
    c = 24.384

    rho, g, Vvoice = airpara1(x[:, 14], z, Vv, mu)
    rho = rho.unsqueeze(-1)
    g = g.unsqueeze(-1)
    Vvoice = Vvoice.unsqueeze(-1)
    qbar = 0.5 * rho * (x[:, 15]).unsqueeze(-1) ** 2
    # 飞行器质量和转动惯量
    m = 136080.
    Ix = 1355818.
    Iy = 13558180.
    Iz = 13558180.
    J = torch.tensor([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])

    disMx = torch.zeros((x.shape[0], 1))
    disMy = torch.zeros((x.shape[0], 1))
    disMz = torch.zeros((x.shape[0], 1))
    # 干扰设置
    for i in range(len(t)):
        if t[i] >= 40:
            disMx[i] = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t[i])
            disMy[i] = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t[i])
            disMz[i] = 1e5 * signal.sawtooth(0.1 * 2 * np.pi * t[i])
        else:
            disMx[i] = 0
            disMy[i] = 0
            disMz[i] = 0

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
    piang1 = 180 / torch.pi * aaa1
    piang2 = 180 / torch.pi * aaa2
    piang3 = 180 / torch.pi * aaa3
    piang = 1
    piang11 = 1
    piang33 = 1

    an = 1

    # 计算攻角、侧滑角、倾侧角等
    beta = torch.arcsin(
        (torch.sin(x[:, 1] - x[:, 17]) * torch.cos(x[:, 2]) + torch.sin(x[:, 0]) * torch.sin(x[:, 2]) * torch.cos(
            x[:, 1] - x[:, 17])) * torch.cos(x[:, 16])
        - torch.cos(x[:, 0]) * torch.sin(x[:, 2]) * torch.sin(x[:, 16]))
    altha = torch.arcsin(
        (torch.cos(x[:, 1] - x[:, 17]) * torch.sin(x[:, 0]) * torch.cos(x[:, 2]) * torch.cos(x[:, 16]) - torch.sin(
            x[:, 1] - x[:, 17]) * torch.sin(x[:, 2]) * torch.cos(x[:, 16])
         - torch.cos(x[:, 0]) * torch.cos(x[:, 2]) * torch.sin(x[:, 16]) / (torch.cos(beta))))
    gammac = torch.arcsin(
        (torch.cos(x[:, 1] - x[:, 17]) * torch.sin(x[:, 0]) * torch.sin(x[:, 2]) * torch.sin(x[:, 16]) + torch.sin(
            x[:, 1] - x[:, 17]) * torch.cos(x[:, 2]) * torch.sin(x[:, 16])
         + torch.cos(x[:, 0]) * torch.sin(x[:, 2]) * torch.cos(x[:, 16]) / torch.cos(beta)))

    althadu = altha * 180 / torch.pi

    # 控制器增益
    KMGainsch = hypersonic_gainsch(x[:, 0], x[:, 2])

    # 干扰观测器输出
    p10 = 10
    p20 = 10
    p30 = 10

    dphi = x[:, 4] * torch.sin(x[:, 2]) + x[:, 5] * torch.cos(x[:, 2])
    dpesai = (x[:, 4] * torch.cos(x[:, 2]) - x[:, 5] * torch.sin(x[:, 2])) / (torch.cos(x[:, 0]))
    dgamma = x[:, 3] - torch.tan(x[:, 0]) * (x[:, 4] * torch.cos(x[:, 2]) - x[:, 5] * torch.sin(x[:, 2]))

    g10 = dphi - x[:, 6]
    dhatx = p10 * g10
    # print("dhatx",dhatx.shape)  #（8001，）

    g20 = dpesai - x[:, 7]
    dhaty = p20 * g20

    g30 = dgamma - x[:, 8]
    dhatz = p30 * g30

    # 期望力矩
    UM = Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz)

    # 输入的force是cof[0,:,:3]（也就是经过线性核处理过的单元）,输入的M是
    D = qbar * Sref * Force[:, 0].unsqueeze(-1)  # 轴向力
    L = qbar * Sref * Force[:, 1].unsqueeze(-1)  # 法向力
    Y = qbar * Sref * Force[:, 2].unsqueeze(-1)  # 侧向力
    # print("D",D.shape)
    # print("M",M.shape)
    Mx = M[:, 0]
    My = M[:, 1]
    Mz = M[:, 2]

    # 姿态环动态
    dx0 = x[:, 4].unsqueeze(-1) * torch.sin(x[:, 2].unsqueeze(-1)) + x[:, 5].unsqueeze(-1) * torch.cos(
        x[:, 2].unsqueeze(-1))
    dx1 = (x[:, 4].unsqueeze(-1) * torch.cos(x[:, 2].unsqueeze(-1)) - x[:, 5].unsqueeze(-1) * torch.sin(
        x[:, 2].unsqueeze(-1))) / torch.cos(x[:, 0].unsqueeze(-1))
    dx2 = x[:, 3].unsqueeze(-1) - torch.tan(x[:, 0].unsqueeze(-1)) * (
                x[:, 4].unsqueeze(-1) * torch.cos(x[:, 2].unsqueeze(-1)) - x[:, 5].unsqueeze(-1) * torch.sin(
            x[:, 2].unsqueeze(-1)))
    dx3 = ((Iy - Iz) / Ix) * x[:, 4].unsqueeze(-1) * x[:, 5].unsqueeze(-1) + (Mx * (1 + deltaMx) + disMx) / Ix
    dx4 = ((Iz - Ix) / Iy) * x[:, 3].unsqueeze(-1) * x[:, 5].unsqueeze(-1) + (My * (1 + deltaMy) + disMy) / Iy
    dx5 = ((Ix - Iy) / Iz) * x[:, 3].unsqueeze(-1) * x[:, 4].unsqueeze(-1) + (Mz * (1 + deltaMz) + disMz) / Iz

    dx6 = torch.sin(x[:, 2].unsqueeze(-1)) / Iy * UM[:, 1].unsqueeze(-1) + torch.cos(x[:, 2].unsqueeze(-1)) / Iz * UM[:,
                                                                                                                   2].unsqueeze(
        -1) + dhatx.unsqueeze(-1)
    dx7 = torch.cos(x[:, 2].unsqueeze(-1)) / (Iy * torch.cos(x[:, 0].unsqueeze(-1))) * UM[:, 1].unsqueeze(
        -1) - torch.sin(x[:, 2].unsqueeze(-1)) / (Iz * torch.cos(x[:, 0].unsqueeze(-1))) * UM[:, 2].unsqueeze(
        -1) + dhaty.unsqueeze(-1)
    dx8 = 1 / Ix * UM[:, 0].unsqueeze(-1) - torch.tan(x[:, 0].unsqueeze(-1)) / Iy * torch.cos(
        x[:, 2].unsqueeze(-1)) * UM[:, 1].unsqueeze(-1) + torch.tan(x[:, 0].unsqueeze(-1)) / Iz * torch.sin(
        x[:, 2].unsqueeze(-1)) * UM[:, 2].unsqueeze(-1) + dhatz.unsqueeze(-1)

    dx9 = x01[0] - x[:, 0].unsqueeze(-1)
    dx10 = x01[1] - x[:, 1].unsqueeze(-1)
    dx11 = x01[2] - x[:, 2].unsqueeze(-1)

    dx12 = x[:, 15].unsqueeze(-1) * torch.cos(x[:, 16].unsqueeze(-1)) * torch.cos(x[:, 17].unsqueeze(-1))
    dx13 = x[:, 15].unsqueeze(-1) * torch.sin(x[:, 16].unsqueeze(-1))
    dx14 = -x[:, 15].unsqueeze(-1) * torch.sin(x[:, 17].unsqueeze(-1)) * torch.cos(x[:, 16].unsqueeze(-1))

    dx15 = -D * (1 + deltaD) / m - g * torch.sin(x[:, 16].unsqueeze(-1))
    dx16 = 1 / (m * x[:, 15].unsqueeze(-1)) * (
                L * (1 + deltaL) * torch.cos(gammac.unsqueeze(-1)) - Y * (1 + deltaY) * torch.sin(
            gammac.unsqueeze(-1))) - g * torch.cos(gammac.unsqueeze(-1)) / x[:, 15].unsqueeze(-1)
    dx17 = 1 / (m * x[:, 15].unsqueeze(-1) * torch.cos(x[:, 16].unsqueeze(-1))) * (
                L * (1 + deltaL) * torch.sin(gammac.unsqueeze(-1)) + Y * (1 + deltaY) * torch.cos(gammac.unsqueeze(-1)))

    dx = torch.stack([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12, dx13, dx14, dx15, dx16, dx17],
                     dim=1)
    # print("dx",dx.shape)   #dx的形状是 torch.Size([8001, 18, 1])

    return dx.squeeze(-1)

def Hypersonic_Gainsch_Controller(KMGainsch, Pid, J, x, x01, dhatx, dhaty, dhatz):
    # Calculate derivatives
    dx1 = x[:, 4] * torch.sin(x[:, 2]) + x[:, 5] * torch.cos(x[:, 2])
    dx2 = (x[:, 4] * torch.cos(x[:, 2]) - x[:, 5] * torch.sin(x[:, 2])) / torch.cos(x[:, 0])
    dx3 = x[:, 3] - torch.tan(x[:, 0]) * (x[:, 4] * torch.cos(x[:, 2]) - x[:, 5] * torch.sin(x[:, 2]))
    # Extract KMGainsch
    Px1, Px2, Px3, Py1, Py2, Py3, Pz1, Pz2, Pz3 = KMGainsch

    PPP = torch.stack([torch.stack([Px1, Px2, Px3]),
                       torch.stack([Py1, Py2, Py3]),
                       torch.stack([Pz1, Pz2, Pz3])])

    # 3*8000*3
    # Extract Pid gains
    kpv1, kiv1, kdv1 = Pid[0]
    kpv2, kiv2, kdv2 = Pid[1]
    kpv3, kiv3, kdv3 = Pid[2]

    # Calculate control output
    Uu = torch.stack([kiv1 * x[:, 9] + kpv1 * (x01[0] - x[:, 0]) + kdv1 * (-dx1),
                      kiv2 * x[:, 10] + kpv2 * (x01[1] - x[:, 1]) + kdv2 * (-dx2),
                      kiv3 * x[:, 11] + kpv3 * (x01[2] - x[:, 2]) + kdv3 * (-dx3)],
                     dim=1)  # dim=1 表示在第二个维度进行堆叠，沿列的方向进行堆叠，就是一列一列堆叠的意思

    # UM = torch.mm(torch.mm(J, PPP), Uu) - torch.mm(torch.mm(J, PPP), torch.Tensor([dhatx, dhaty, dhatz]).unsqueeze(-1))
    UM = torch.matmul(torch.matmul(J, PPP).permute(2, 0, 1), Uu.unsqueeze(-1)).squeeze(-1) - torch.matmul(
        torch.matmul(J, PPP).permute(2, 0, 1), torch.stack([dhatx, dhaty, dhatz], dim=1).unsqueeze(-1)).squeeze(-1)
    # J, PPP都是3*3，dhatx是一个8000*1的向量

    # 8000*3
    return UM

def hypersonic_gainsch(x1, x3):

    ast1 = torch.zeros_like(x1)
    ast3 = torch.zeros_like(x3)

    ast1[x1 < -0.4] = -0.4
    ast3[x3 < -0.4] = -0.4
    ast3[(x3 >= -0.4) & (x3 < -0.3)] = -0.3
    ast3[(x3 >= -0.3) & (x3 < -0.2)] = -0.2
    ast3[(x3 >= -0.2) & (x3 < -0.1)] = -0.1
    ast3[(x3 >= -0.1) & (x3 < 0)] = 0
    ast3[(x3 >= 0) & (x3 < 0.1)] = 0.1
    ast3[(x3 >= 0.1) & (x3 < 0.2)] = 0.2
    ast3[(x3 >= 0.2) & (x3 < 0.3)] = 0.3
    ast3[(x3 >= 0.3) & (x3 < 0.4)] = 0.4
    ast3[(x3 >= 0.4) & (x3 < 0.5)] = 0.5

    ast1[(x1 >= -0.4) & (x1 < -0.3)] = -0.3
    ast1[(x1 >= -0.3) & (x1 < -0.2)] = -0.2
    ast1[(x1 >= -0.2) & (x1 < -0.1)] = -0.1
    ast1[(x1 >= -0.1) & (x1 < 0)] = 0
    ast1[(x1 >= 0) & (x1 < 0.1)] = 0.1
    ast1[(x1 >= 0.1) & (x1 < 0.2)] = 0.2
    ast1[(x1 >= 0.2) & (x1 < 0.3)] = 0.3
    ast1[(x1 >= 0.3) & (x1 < 0.4)] = 0.4
    ast1[(x1 >= 0.4) & (x1 < 0.5)] = 0.5
    # 计算kMxg1、kMxg2、kMxg3、kMyg1、kMyg2、kMyg3、kMzg1、kMzg2和kMzg3

    kMxg1 = torch.zeros_like(x1)
    kMxg2 = (x1 - (ast1 - 0.1)) / 0.1 * (torch.sin(ast1) - torch.sin(ast1 - 0.1)) + torch.sin(ast1 - 0.1)
    kMxg3 = torch.ones_like(x1)
    kMyg1 = (x3 - (ast3 - 0.1)) / 0.1 * (torch.sin(ast3) - torch.sin(ast3 - 0.1)) + torch.sin(ast3 - 0.1)
    kMyg2 = ((x1 - (ast1 - 0.1)) / 0.1 * (torch.cos(ast1) - torch.cos(ast1 - 0.1)) + torch.cos(ast1 - 0.1)) * (
                (x3 - (ast3 - 0.1)) / 0.1 * (torch.cos(ast3) - torch.cos(ast3 - 0.1)) + torch.cos(ast3 - 0.1))
    kMyg3 = torch.zeros_like(x3)
    kMzg1 = (x3 - (ast3 - 0.1)) / 0.1 * (torch.cos(ast3) - torch.cos(ast3 - 0.1)) + torch.cos(ast3 - 0.1)
    kMzg2 = -((x1 - (ast1 - 0.1)) / 0.1 * (torch.cos(ast1) - torch.cos(ast1 - 0.1)) + torch.cos(ast1 - 0.1)) * (
                (x3 - (ast3 - 0.1)) / 0.1 * (torch.sin(ast3) - torch.sin(ast3 - 0.1)) + torch.sin(ast3 - 0.1))
    kMzg3 = torch.zeros_like(x3)

    # 将结果存储在KMGainsch中并返回
    # KMGainsch = torch.Tensor([kMxg1, kMxg2, kMxg3, kMyg1, kMyg2, kMyg3, kMzg1, kMzg2, kMzg3])
    KMGainsch = torch.stack([kMxg1, kMxg2, kMxg3, kMyg1, kMyg2, kMyg3, kMzg1, kMzg2, kMzg3])
    return KMGainsch

def airpara0(h, z, Vv, mu):
    # 从2000开始没改
    rho = 0.0
    g = 0.0
    Vvvoice = 0.0
    if h < 11000:
        z_values = [z[i] + (z[i + 1] - z[i]) * (h - i * 500) / 500 for i in range(23)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - i * 500) / 500 for i in range(23)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - i * 500) / 500 for i in range(23)]
        for i in range(23):
            if h < (i + 1) * 500:
                rho = z_values[i]
                g = mu_values[i]
                Vvvoice = Vv_values[i]
                break

    elif h < 32000:
        z_values = [z[i] + (z[i + 1] - z[i]) * (h - 11000 - (i - 23) * 1000) / 1000 for i in range(23, 44)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - 11000 - (i - 23) * 1000) / 1000 for i in range(23, 44)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - 11000 - (i - 23) * 1000) / 1000 for i in range(23, 44)]
        for i in range(23, 44):
            if h < 11000 + (i - 22) * 1000:
                rho = z_values[i - 23]
                g = mu_values[i - 23]
                Vvvoice = Vv_values[i - 23]
                break

    elif h < 48000:
        z_values = [z[i] + (z[i + 1] - z[i]) * (h - 32000 - (i - 44) * 2000) / 2000 for i in range(44, 52)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - 32000 - (i - 44) * 2000) / 2000 for i in range(44, 52)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - 32000 - (i - 44) * 2000) / 2000 for i in range(44, 52)]
        for i in range(44, 52):
            if h < 32000 + (i - 43) * 2000:
                rho = z_values[i - 44]
                g = mu_values[i - 44]
                Vvvoice = Vv_values[i - 44]
                break
    elif h < 100000:
        z_values = [z[i] + (z[i + 1] - z[i]) * (h - 48000 - (i - 52) * 5000) / 5000 for i in range(52, 63)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - 48000 - (i - 52) * 5000) / 5000 for i in range(52, 63)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - 48000 - (i - 52) * 5000) / 5000 for i in range(52, 63)]
        for i in range(52, 63):
            if h < 48000 + (i - 51) * 5000:
                rho = z_values[i - 52]
                g = mu_values[i - 52]
                Vvvoice = Vv_values[i - 52]
                break
    elif h < 300000:
        z_values = [z[i] + (z[i + 1] - z[i]) * (h - 100000 - (i - 63) * 10000) / 10000 for i in range(63, 83)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - 100000 - (i - 63) * 10000) / 10000 for i in range(63, 83)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - 100000 - (i - 63) * 10000) / 10000 for i in range(63, 83)]
        for i in range(63, 83):
            if h < 100000 + (i - 62) * 10000:
                rho = z_values[i - 63]
                g = mu_values[i - 63]
                Vvvoice = Vv_values[i - 63]
                break

    elif h < 500000:
        z_values = [z[i] + (z[i + 1] - z[i]) * (h - 300000 - (i - 63) * 20000) / 20000 for i in range(83, 93)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - 300000 - (i - 63) * 20000) / 20000 for i in range(83, 93)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - 300000 - (i - 63) * 20000) / 20000 for i in range(83, 93)]
        for i in range(83, 93):
            if h < 300000 + (i - 82) * 20000:
                rho = z_values[i - 83]
                g = mu_values[i - 83]
                Vvvoice = Vv_values[i - 83]
                break

    elif h < 1000000:

        z_values = [z[i] + (z[i + 1] - z[i]) * (h - 500000 - (i - 93) * 50000) / 50000 for i in range(93, 100)]
        mu_values = [mu[i] + (mu[i + 1] - mu[i]) * (h - 500000 - (i - 93) * 50000) / 50000 for i in range(93, 100)]
        Vv_values = [Vv[i] + (Vv[i + 1] - Vv[i]) * (h - 500000 - (i - 93) * 50000) / 50000 for i in range(93, 100)]
        for i in range(93, 100):
            if h < 500000 + (i - 92) * 50000:
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
            z_values[condition] = z[i] + (z[i + 1] - z[i]) * (h[condition] - (i - start) * (end - start) * 1000) / (
                        (end - start) * 1000)
            mu_values[condition] = mu[i] + (mu[i + 1] - mu[i]) * (h[condition] - (i - start) * (end - start) * 1000) / (
                        (end - start) * 1000)
            Vv_values[condition] = Vv[i] + (Vv[i + 1] - Vv[i]) * (h[condition] - (i - start) * (end - start) * 1000) / (
                        (end - start) * 1000)
        break

    return z_values, mu_values, Vv_values

def airpara1(h, z, Vv, mu):
    z_values = torch.zeros_like(h)
    mu_values = torch.zeros_like(h)
    Vv_values = torch.zeros_like(h)

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
            z_values[condition] = z[i] + (z[i + 1] - z[i]) * (h[condition] - (i - start) * (end - start) * 1000) / (
                        (end - start) * 1000)
            mu_values[condition] = mu[i] + (mu[i + 1] - mu[i]) * (h[condition] - (i - start) * (end - start) * 1000) / (
                        (end - start) * 1000)
            Vv_values[condition] = Vv[i] + (Vv[i + 1] - Vv[i]) * (h[condition] - (i - start) * (end - start) * 1000) / (
                        (end - start) * 1000)
        break

    return z_values, mu_values, Vv_values

def duopianjisuan(x, UM, altha, beta, an, piang, piang1, piang2, piang3, Vvoice, rho, piang11, piang33):
    Sref = 334.73
    b = 18.288
    c = 24.384

    x4 = x[:, 15].unsqueeze(-1)  # Velocity variable
    beta = beta.unsqueeze(-1)
    qbar = 0.5 * rho * x4 ** 2  # 注意广播机制。x4×x4，x4（8000，），为了能够相乘会自动扩展到（8000，8000）。需要令x4先unsqueeze(-1)
    altha = altha.unsqueeze(-1)  # altha也要从（8000，），变为（8000，1）

    # print('beta',beta.shape)

    wx = x[:, 3].unsqueeze(-1)  # x-axis angular velocity
    wy = x[:, 4].unsqueeze(-1)  # y-axis angular velocity
    wz = x[:, 5].unsqueeze(-1)  # z-axis angular velocity

    # print('wx=',wx.shape)

    Ma = x4 / Vvoice  # Mach number,对应位置相除

    # Roll moment linear coefficients
    CMx_beta = -1.402 * 0.1 + 3.326 * 0.01 * Ma - 7.59e-4 * altha + 8.596e-6 * (
            altha * Ma) - 3.794e-3 * Ma ** 2 + 2.354e-6 * altha ** 2 - 1.044e-8 * (
                       altha * Ma) ** 2 + 2.219e-4 * Ma ** 3 - 8.964e-18 * altha ** 3 - 6.462e-6 * Ma ** 4 + 3.803e-19 * altha ** 4 + 7.419e-8 * Ma ** 5 - 3.353e-21 * altha ** 5
    CMx_deltaaltha_con = 3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 4.95e-6 * altha ** 2 + 1.411e-6 * Ma ** 2
    CMx_deltaaltha_u1 = 1.17e-4 * piang1 + 2.794e-8 * altha * Ma * piang1

    CMx_deltae_con = -(3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 4.95e-6 * altha ** 2 + 1.411e-6 * Ma ** 2)
    CMx_deltae_u3 = -(1.17e-4 * piang3 + 2.794e-8 * altha * Ma * piang3)

    CMx_deltar_con = -5.0103e-19 + 6.2723e-20 * altha + 2.3418e-20 * Ma - 3.4201e-21 * altha * Ma
    CMx_deltar_u2 = -3.5496e-6 * Ma * piang2 + 5.5547e-8 * altha * Ma * piang2 + 1.1441e-4 * piang2 - 2.6824e-6 * altha * piang2

    CMx_r = 3.82 * 0.1 - 1.06 * 0.1 * Ma + 1.94e-3 * altha - 8.15e-5 * altha * Ma + 1.45 * 0.01 * Ma ** 2 - 9.76e-6 * altha ** 2 + 4.49e-8 * (
            altha * Ma) ** 2 - 1.02e-3 * Ma ** 3 - 2.7e-7 * altha ** 3 + 3.56e-5 * Ma ** 4 + 3.19e-8 * altha ** 4 - 4.81e-7 * Ma ** 5 - 1.06e-9 * altha ** 5
    CMx_p = -2.99 * 0.1 + 7.47 * 0.01 * Ma + 1.38e-3 * altha - 8.78e-5 * altha * Ma - 9.13e-3 * Ma ** 2 - 2.04e-4 * altha ** 2 - 1.52e-7 * (
            altha * Ma) ** 2 + 5.73e-4 * Ma ** 3 - 3.86e-5 * altha ** 3 - 1.79e-5 * Ma ** 4 + 4.21e-6 * altha ** 4 + 2.2e-7 * Ma ** 5 - 1.15e-7 * altha ** 5

    CMx_con = CMx_beta * beta + CMx_deltaaltha_con + CMx_deltae_con + CMx_deltar_con + CMx_r * wz * b / (
            2 * x4) + CMx_p * wx * b / (2 * x4)

    # Pitch moment linear coefficients
    CMz_altha = -2.192e-2 + 7.739e-3 * Ma - 2.26e-3 * altha + 1.808e-4 * Ma * altha - 8.849e-4 * Ma ** 2 + 2.616e-4 * altha ** 2 - 2.88e-7 * (
            Ma * altha) ** 2 + 4.617e-5 * Ma ** 3 - 7.887e-5 * altha ** 3 - 1.143e-6 * Ma ** 4 + 8.288e-6 * altha ** 4 + 1.082e-8 * Ma ** 5 - 2.789e-7 * altha ** 5
    CMz_deltaaltha_con = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma - 4.46e-6 * altha * Ma
    CMz_deltaaltha_u1 = 2.89e-4 * piang1 + 4.48e-6 * altha * piang1 - 5.87e-6 * Ma * piang1 + 9.72e-8 * altha * Ma * piang1

    CMz_deltae_con = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma - 4.46e-6 * altha * Ma
    CMz_deltae_u3 = 2.89e-4 * piang3 + 4.48e-6 * altha * piang3 - 5.87e-6 * Ma * piang3 + 9.72e-8 * altha * Ma * piang3

    CMz_deltar = -2.79e-5 * altha - 5.89e-8 * altha ** 2 + 1.58e-3 * Ma ** 2 + 6.42e-8 * altha ** 3 - 6.69e-4 * Ma ** 3 - 2.1e-8 * altha ** 4 + 1.05e-4 * Ma ** 4 + 3.14e-9 * altha ** 5 - 7.74e-6 * Ma ** 5 - 2.18e-10 * altha ** 6 + 2.7e-7 * Ma ** 6 + 5.74e-12 * altha ** 7 - 3.58e-9 * Ma ** 7
    CMz_deltac = 0
    CMz_q = -1.36 + 3.86e-1 * Ma + 7.85e-4 * altha + 1.4e-4 * altha * Ma - 5.42e-2 * Ma ** 2 + 2.36e-3 * altha ** 2 - 1.95e-6 * (
            altha * Ma) ** 2 + 3.8e-3 * Ma ** 3 - 1.48e-3 * altha ** 3 - 1.3e-4 * Ma ** 4 + 1.69e-4 * altha ** 4 + 1.71e-6 * Ma ** 5 - 5.93e-6 * altha ** 5
    CMz_con = CMz_altha + CMz_deltaaltha_con + CMz_deltae_con + CMz_deltar + CMz_deltac + CMz_q * wy * c / (2 * x4)

    # Yaw moment linear coefficients
    CMy_beta = 6.998e-4 * altha + 5.9115 * 0.01 * Ma - 7.525e-5 * Ma * altha + 2.516e-4 * altha ** 2 - 1.4824 * 0.01 * Ma ** 2 - 2.1924e-7 * (
            Ma * altha) ** 2 - 1.0777e-4 * altha ** 3 + 1.2692e-3 * Ma ** 3 + 1.0707e-8 * (
                       Ma * altha) ** 3 + 9.4989e-6 * altha ** 4 - 4.7098e-5 * Ma ** 4 - 5.5472e-11 * (
                       Ma * altha) ** 4 - 2.5953e-7 * altha ** 5 + 6.4284e-7 * Ma ** 5 + 8.5863e-14 * (
                       Ma * altha) ** 5
    CMy_deltaaltha_con = 2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 6.39e-7 * altha ** 2 + 8.16e-7 * Ma ** 2
    CMy_deltaaltha_u1 = -1.3e-5 * piang1 - 8.93e-8 * altha * Ma * piang1
    CMy_deltae_con = -(2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 6.39e-7 * altha ** 2 + 8.16e-7 * Ma ** 2)
    CMy_deltae_u3 = 1.3e-5 * piang3 + 8.93e-8 * altha * Ma * piang3
    CMy_deltar_con = 2.85e-18 - 3.59e-19 * altha - 1.26e-19 * Ma + 1.57e-20 * (altha * Ma)
    CMy_deltar_u2 = -5.28e-4 * piang2 + 1.39e-5 * altha * piang2 + 1.65e-5 * (Ma * piang2) - 3.13e-7 * (
            altha * Ma) * piang2

    CMy_p = 3.68 * 0.1 - 9.79e-2 * Ma + 7.61e-16 * altha + 1.24e-2 * Ma ** 2 - 4.64e-16 * altha ** 2 - 8.05e-4 * Ma ** 3 + 1.01e-16 * altha ** 3 + 2.57e-5 * Ma ** 4 - 9.18e-18 * altha ** 4 - 3.2e-7 * Ma ** 5 + 2.96e-19 * altha ** 5
    CMy_r = -2.41 + 5.96e-1 * Ma - 2.74e-3 * altha + 2.09e-4 * (
            altha * Ma) - 7.57e-2 * Ma ** 2 + 1.15e-3 * altha ** 2 - 6.53e-8 * (
                    altha * Ma) ** 2 + 4.9e-3 * Ma ** 3 - 3.87e-4 * altha ** 3 - 1.57e-4 * Ma ** 4 + 3.6e-5 * altha ** 4 + 1.96e-6 * Ma ** 5 - 1.18e-6 * altha ** 5

    CMy_con = CMy_beta * beta + CMy_deltaaltha_con + CMy_deltae_con + CMy_deltar_con + CMy_p * wx * b / (
            2 * x4) + CMy_r * wz * b / (2 * x4)

    # Solving for control deflections
    # ACM = torch.Tensor([[1 / (qbar * Sref), 0, 0],
    #                 [0, 1 / (qbar * b * Sref), 0],
    #                 [0, 0, 1 / (qbar * c * Sref)]])
    # print(len(qbar))
    ACM = torch.zeros((len(qbar), 3, 3), dtype=torch.float32)
    ACM[:, 0, 0] = 1 / (qbar.squeeze(-1) * Sref)  # 注意！！！！前面qbar的shape是（8001，1）导致此处如果不加squeeze(-1)，会超出ACM的第一个维度
    ACM[:, 1, 1] = 1 / (qbar.squeeze(-1) * b * Sref)
    ACM[:, 2, 2] = 1 / (qbar.squeeze(-1) * c * Sref)

    # ACMcon = torch.Tensor([an * CMx_con, an * CMy_con, an * CMz_con])

    ACMcon = torch.stack([an * CMx_con, an * CMy_con, an * CMz_con], dim=1)

    # ACMu = an * torch.Tensor([[CMx_deltaaltha_u1, CMx_deltar_u2, CMx_deltae_u3],
    #                       [CMy_deltaaltha_u1, CMy_deltar_u2, CMy_deltae_u3],
    #                       [CMz_deltaaltha_u1, 0, CMz_deltae_u3]])

    ACMu = torch.zeros((len(qbar), 3, 3), dtype=torch.float32)

    ACMu[:, 0, :] = an * torch.cat([CMx_deltaaltha_u1, CMx_deltar_u2, CMx_deltae_u3], dim=1)  # （8000，3）
    ACMu[:, 1, :] = an * torch.cat([CMy_deltaaltha_u1, CMy_deltar_u2, CMy_deltae_u3], dim=1)
    ACMu[:, 2, :] = an * torch.cat([CMz_deltaaltha_u1, torch.zeros_like(CMz_deltaaltha_u1), CMz_deltae_u3], dim=1)

    # print(ACM.shape)
    # print(ACMu.shape)
    # print(UM.shape)
    # print(ACMcon.shape)
    u = torch.linalg.solve(ACMu, torch.bmm(ACM, UM.unsqueeze(-1)) - ACMcon).squeeze(-1)
    # 分析之后，u是（8000，3）
    ubar = 25 / 57.3

    # for i in range(3):
    #     if u[i] > ubar:
    #         u[i] = ubar
    #     elif u[i] < -ubar:
    #         u[i] = -ubar
    u = torch.clamp(u, -ubar, ubar)

    return u

def lijuqiujie(x, u, altha, beta, an, piang, piang1, piang2, piang3, rho, Vvoice, piang11, piang33):
    # Constants
    Sref = 334.73
    b = 18.288
    c = 24.384
    deltarho = -0.3
    Vfield = -100

    # Extract variables from input arrays
    x4 = x[:, 15].unsqueeze(-1)
    qbar = 0.5 * rho * (1 + deltarho) * (x4 + Vfield) ** 2
    # print(u)
    wx = x[:, 3].unsqueeze(-1)
    wy = x[:, 4].unsqueeze(-1)
    wz = x[:, 5].unsqueeze(-1)

    u0 = u[:, 0].unsqueeze(-1)
    u1 = u[:, 1].unsqueeze(-1)
    u2 = u[:, 2].unsqueeze(-1)
    # print("u[0]=",u[:,0].shape)

    # print("qbar===",qbar.shape)
    beta = beta.unsqueeze(-1)
    altha = altha.unsqueeze(-1)
    Ma = x4 / Vvoice  # (8000,1)

    # Calculate CMx (aerodynamic moment coefficient)
    CMx_beta = -1.402 * 0.1 + 3.326 * 0.01 * Ma - 7.59e-4 * altha + 8.596e-6 * (
            altha * Ma) - 3.794e-3 * Ma ** 2 + 2.354e-6 * altha ** 2 - 1.044e-8 * (
                       altha * Ma) ** 2 + 2.219e-4 * Ma ** 3 - 8.964e-18 * altha ** 3 - 6.462e-6 * Ma ** 4 + 3.803e-19 * altha ** 4 + 7.419e-8 * Ma ** 5 - 3.353e-21 * altha ** 5
    CMx_deltaaltha = 3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 1.17e-4 * piang1 * u0 + 2.794e-8 * altha * Ma * piang1 * u0 + 4.95e-6 * altha ** 2 + 1.411e-6 * Ma ** 2 - 1.16e-6 * (
            piang1 * u0) ** 2 - 4.641e-11 * (altha * Ma * piang1 * u0) ** 2
    CMx_deltae = -(
            3.57e-4 - 9.569e-5 * altha - 3.598e-5 * Ma + 1.17e-4 * piang3 * u2 + 2.794e-8 * altha * Ma * piang3 *
            u2 + 4.95e-6 * altha ** 2 + 1.411e-6 * Ma ** 2 - 1.16e-6 * (piang3 * u2) ** 2 - 4.641e-11 * (
                    altha * Ma * piang3 * u2) ** 2)
    CMx_deltar = -5.0103e-19 + 6.2723e-20 * altha + 2.3418e-20 * Ma + 1.1441e-4 * piang2 * u1 - 2.6824e-6 * altha * piang2 * u1 - 3.4201e-21 * altha * Ma - 3.5496e-6 * Ma * piang2 * u1 + 5.5547e-8 * altha * Ma * piang2 * u1
    CMx_r = 3.82 * 0.1 - 1.06 * 0.1 * Ma + 1.94e-3 * altha - 8.15e-5 * altha * Ma + 1.45 * 0.01 * Ma ** 2 - 9.76e-6 * altha ** 2 + 4.49e-8 * (
            altha * Ma) ** 2 - 1.02e-3 * Ma ** 3 - 2.7e-7 * altha ** 3 + 3.56e-5 * Ma ** 4 + 3.19e-8 * altha ** 4 - 4.81e-7 * Ma ** 5 - 1.06e-9 * altha ** 5
    CMx_p = -2.99 * 0.1 + 7.47 * 0.01 * Ma + 1.38e-3 * altha - 8.78e-5 * altha * Ma - 9.13e-3 * Ma ** 2 - 2.04e-4 * altha ** 2 - 1.52e-7 * (
            altha * Ma) ** 2 + 5.73e-4 * Ma ** 3 - 3.86e-5 * altha ** 3 - 1.79e-5 * Ma ** 4 + 4.21e-6 * altha ** 4 + 2.2e-7 * Ma ** 5 - 1.15e-7 * altha ** 5
    CMx = CMx_beta * beta + CMx_deltaaltha + CMx_deltae + CMx_deltar + CMx_r * wz * b / (2 * x4) + CMx_p * wx * b / (
            2 * x4)

    Mx = qbar * Sref * an * CMx

    # Calculate CMz (aerodynamic moment coefficient)
    CMz_altha = -2.192e-2 + 7.739e-3 * Ma - 2.26e-3 * altha + 1.808e-4 * Ma * altha - 8.849e-4 * Ma ** 2 + 2.616e-4 * altha ** 2 - 2.88e-7 * (
            Ma * altha) ** 2 + 4.617e-5 * Ma ** 3 - 7.887e-5 * altha ** 3 - 1.143e-6 * Ma ** 4 + 8.288e-6 * altha ** 4 + 1.082e-8 * Ma ** 5 - 2.789e-7 * altha ** 5
    CMz_deltaaltha = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma + 2.89e-4 * piang1 * u0 + 4.48e-6 * altha * piang1 * u0 - 4.46e-6 * altha * Ma - 5.87e-6 * Ma * piang1 * u0 + 9.72e-8 * altha * Ma * piang1 * u0
    CMz_deltae = -5.67e-5 - 6.59e-5 * altha - 1.51e-6 * Ma + 2.89e-4 * piang3 * u2 + 4.48e-6 * altha * piang3 * u2 - 4.46e-6 * altha * Ma - 5.87e-6 * Ma * piang3 * u2 + 9.72e-8 * altha * Ma * piang3 * u2
    CMz_deltar = -2.79e-5 * altha - 5.89e-8 * altha ** 2 + 1.58e-3 * Ma ** 2 + 6.42e-8 * altha ** 3 - 6.69e-4 * Ma ** 3 - 2.1e-8 * altha ** 4 + 1.05e-4 * Ma ** 4 + 3.14e-9 * altha ** 5 - 7.74e-6 * Ma ** 5 - 2.18e-10 * altha ** 6 + 2.7e-7 * Ma ** 6 + 5.74e-12 * altha ** 7 - 3.58e-9 * Ma ** 7 + 1.43e-7 * (
            piang * u1) ** 4 - 4.77e-22 * (piang * u1) ** 5 - 3.38e-10 * (piang * u1) ** 6 + 2.63e-24 * (
                         piang * u1) ** 7
    CMz_deltac = 0
    CMz_q = -1.36 + 3.86e-1 * Ma + 7.85e-4 * altha + 1.4e-4 * altha * Ma - 5.42e-2 * Ma ** 2 + 2.36e-3 * altha ** 2 - 1.95e-6 * (
            altha * Ma) ** 2 + 3.8e-3 * Ma ** 3 - 1.48e-3 * altha ** 3 - 1.3e-4 * Ma ** 4 + 1.69e-4 * altha ** 4 + 1.71e-6 * Ma ** 5 - 5.93e-6 * altha ** 5

    # print(CMz_altha.shape,CMz_deltaaltha.shape,CMz_deltae.shape,CMz_deltar.shape,CMz_q.shape,wy.shape)
    CMz = CMz_altha + CMz_deltaaltha + CMz_deltae + CMz_deltar + CMz_deltac + CMz_q * wy * c / (2 * x4)
    Mz = qbar * c * Sref * an * CMz

    # Calculate CMy (aerodynamic moment coefficient)
    CMy_beta = 6.998e-4 * altha + 5.9115 * 0.01 * Ma - 7.525e-5 * Ma * altha + 2.516e-4 * altha ** 2 - 1.4824 * 0.01 * Ma ** 2 - 2.1924e-7 * (
            Ma * altha) ** 2 - 1.0777e-4 * altha ** 3 + 1.2692e-3 * Ma ** 3 + 1.0707e-8 * (
                       Ma * altha) ** 3 + 9.4989e-6 * altha ** 4 - 4.7098e-5 * Ma ** 4 - 5.5472e-11 * (
                       Ma * altha) ** 4 - 2.5953e-7 * altha ** 5 + 6.4284e-7 * Ma ** 5 + 8.5863e-14 * (
                       Ma * altha) ** 5
    CMy_deltaaltha = 2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 1.3e-5 * piang1 * u0 - 8.93e-8 * altha * Ma * piang1 * \
                     u0 - 6.39e-7 * altha ** 2 + 8.16e-7 * Ma ** 2 + 1.97e-6 * (piang1 * u0) ** 2
    CMy_deltae = -(
                2.1e-4 + 1.83e-5 * altha - 3.56e-5 * Ma - 1.3e-5 * piang3 * u2 - 8.93e-8 * altha * Ma * piang3 * u2 - 6.39e-7 * altha ** 2 + 8.16e-7 * Ma ** 2 + 1.97e-6 * (
                    piang3 * u2) ** 2)
    CMy_deltar = 2.85e-18 - 3.59e-19 * altha - 1.26e-19 * Ma - 5.28e-4 * piang2 * u1 + 1.39e-5 * altha * piang2 * u1 + 1.57e-20 * (
                altha * Ma) + 1.65e-5 * (Ma * piang2 * u1) - 3.13e-7 * (altha * Ma) * piang2 * u1
    CMy_p = 3.68e-1 - 9.79e-2 * Ma + 7.61e-16 * altha + 1.24e-2 * Ma ** 2 - 4.64e-16 * altha ** 2 - 8.05e-4 * Ma ** 3 + 1.01e-16 * altha ** 3 + 2.57e-5 * Ma ** 4 - 9.18e-18 * altha ** 4 - 3.2e-7 * Ma ** 5 + 2.96e-19 * altha ** 5
    CMy_r = -2.41 + 5.96e-1 * Ma - 2.74e-3 * altha + 2.09e-4 * (
            altha * Ma) - 7.57e-2 * Ma ** 2 + 1.15e-3 * altha ** 2 - 6.53e-8 * (
                    altha * Ma) ** 2 + 4.9e-3 * Ma ** 3 - 3.87e-4 * altha ** 3 - 1.57e-4 * Ma ** 4 + 3.6e-5 * altha ** 4 + 1.96e-6 * Ma ** 5 - 1.18e-6 * altha ** 5
    CMy = CMy_beta * beta + CMy_deltaaltha + CMy_deltae + CMy_deltar + CMy_p * wx * b / (2 * x4) + CMy_r * wz * b / (
            2 * x4)
    My = qbar * b * Sref * an * CMy

    CD_altha = 8.717e-2 - 3.307 * 0.01 * Ma + 3.179 * 0.001 * altha - 1.25 * 0.0001 * altha * Ma + 5.036 * 0.001 * Ma ** 2 - 1.1 * 0.001 * altha ** 2 + 1.405e-7 * (
            altha * Ma) ** 2 - 3.658e-4 * Ma ** 3 + 3.175e-4 * altha ** 3 + 1.274e-5 * Ma ** 4 - 2.985e-5 * altha ** 4 - 1.705e-7 * Ma ** 5 + 9.766e-7 * altha ** 5
    CD_deltaaltha = 4.5548e-4 + 2.5411e-5 * altha - 1.1436e-4 * Ma - 3.6417e-5 * (piang1 * u0) - 5.3015e-7 * (
            altha * Ma) * (piang1 * u0) + 3.2187e-6 * altha ** 2 + 3.014e-6 * Ma ** 2 + 6.9629e-6 * (
                            piang1 * u0) ** 2 + 2.1026e-12 * (altha * Ma * (piang1 * u0)) ** 2
    CD_deltae = 4.5548e-4 + 2.5411e-5 * altha - 1.1436e-4 * Ma - 3.6417e-5 * (piang3 * u2) - 5.3015e-7 * (
            altha * Ma) * (piang3 * u2) + 3.2187e-6 * altha ** 2 + 3.014e-6 * Ma ** 2 + 6.9629e-6 * (
                        piang3 * u2) ** 2 + 2.1026e-12 * (altha * Ma * (piang3 * u2)) ** 2
    CD_deltar = 7.5e-4 - 2.29e-5 * altha - 9.69e-5 * Ma - 1.83e-6 * piang2 * u1 + 9.13e-9 * (
                altha * Ma) * piang2 * u1 + 8.76e-7 * altha ** 2 + 2.7e-6 * Ma ** 2 + 1.97e-6 * (
                            piang2 * u1) ** 2 - 1.7702e-11 * (
                        altha * Ma * piang2 * u1) ** 2
    CD_deltac = 0
    CD = CD_altha + CD_deltaaltha + CD_deltae + CD_deltar + CD_deltac
    # D = qbar * Sref * CD  # 轴向力

    CL_altha = -8.19e-2 + 4.7 * 0.01 * Ma + 1.86 * 0.01 * altha - 4.73e-4 * altha * Ma - 9.19e-3 * Ma ** 2 - 1.52e-4 * altha ** 2 + 5.99e-7 * (
            altha * Ma) ** 2 + 7.74e-4 * Ma ** 3 + 4.08e-6 * altha ** 3 - 2.93e-5 * Ma ** 4 - 3.91e-7 * altha ** 4 + 4.12e-7 * Ma ** 5 + 1.3e-8 * altha ** 5
    CL_deltaaltha = -1.45e-5 + 1.01e-4 * altha + 7.1e-6 * Ma - 4.14e-4 * (piang1 * u0) - 3.51e-6 * altha * (
            piang1 * u0) + 4.7e-6 * (altha * Ma) + 8.72e-6 * Ma * (piang1 * u0) - 1.7e-7 * altha * Ma * (
                            piang1 * u0)
    CL_deltae = -1.45e-5 + 1.01e-4 * altha + 7.1e-6 * Ma - 4.14e-4 * (piang3 * u2) - 3.51e-6 * altha * (
            piang3 * u2) + 4.7e-6 * (altha * Ma) + 8.72e-6 * Ma * (piang3 * u2) - 1.7e-7 * altha * Ma * (
                        piang3 * u2)
    CL_deltac = 0
    CL = CL_altha + CL_deltaaltha + CL_deltae + CL_deltac
    # L = qbar * Sref * CL  # 法向力

    CY_deltaaltha = -1.02e-6 - 1.12e-7 * altha + 4.48e-7 * Ma + 2.27e-7 * (piang1 * u0) + 4.11e-9 * altha * Ma * (
            piang1 * u0) + 2.82e-9 * altha ** 2 - 2.36e-8 * Ma ** 2 - 5.04e-8 * (piang1 * u0) ** 2 + 4.5e-14 * (
                            altha * Ma * (piang1 * u0)) ** 2
    CY_deltae = -(-1.02e-6 - 1.12e-7 * altha + 4.48e-7 * Ma + 2.27e-7 * (piang3 * u2) + 4.11e-9 * altha * Ma * (
            piang3 * u2) + 2.82e-9 * altha ** 2 - 2.36e-8 * Ma ** 2 - 5.04e-8 * (piang3 * u2) ** 2 + 4.5e-14 * (
                          altha * Ma * (piang3 * u2)) ** 2)
    CY_deltar = -1.43e-18 + 4.86e-20 * altha + 1.86e-19 * Ma + 3.84e-4 * piang2 * u1 - 1.17e-5 * altha * piang2 * u1 - 1.07e-5 * Ma * piang2 * u1 + 2.6e-7 * altha * Ma * piang2 * u1
    CY_beta = 2.8803e-3 * altha - 2.8943e-4 * altha * Ma + 5.4822e-2 * Ma ** 2 + 7.3535e-4 * altha ** 2 - 4.6409e-9 * (
            (altha * Ma ** 2) ** 2) - 2.0675e-8 * ((altha ** 2 * Ma) ** 2) + 4.6205e-6 * (
                      (altha * Ma) ** 2) + 2.6144e-11 * (
                      altha ** 2 * Ma ** 2) ** 2 - 4.3203e-3 * Ma ** 3 - 3.7405e-4 * altha ** 3 + Ma ** 4 * 1.5495e-4 + 2.8183e-5 * altha ** 4 + Ma ** 5 * -2.0829e-6 + altha ** 5 * -5.2083e-7
    CY = CY_beta * beta + CY_deltaaltha + CY_deltae + CY_deltar

    # print("cysaodaoi==",CY_deltaaltha.shape)
    # pre_out = 0.095*CY+0.00+output[0]*1e-8
    # Y = qbar * Sref *  CY  # qbar * Sref *(*CY+output[1])  # 侧向力
    Force_cof = torch.stack([CD, CL, CY], dim=1)

    M = torch.stack([Mx, My, Mz], dim=1)
    # print(Mx.shape,My.shape,Mz.shape)
    return Force_cof.squeeze(), M

class BayesianRegressor(nn.Module):
    def __init__(self, goal_data):
        # self.bias_list = bias_list
        self.goal_data = goal_data
        super().__init__()
        
        self.encoder11 = bnn.BayesLinear(prior_mu=0,prior_sigma=0.1,in_features=1,out_features=1,bias=None)
        self.encoder12 = bnn.BayesLinear(prior_mu=0,prior_sigma=0.1,in_features=1,out_features=1,bias=None)
        self.encoder13 = bnn.BayesLinear(prior_mu=0,prior_sigma=0.1,in_features=1,out_features=1,bias=None)

        self.bias1 = nn.Parameter(torch.empty(1).uniform_(-1e-2, 1e-2))
        self.bias2 = nn.Parameter(torch.empty(1).uniform_(-1e-2, 1e-2))
        self.bias3 = nn.Parameter(torch.empty(1).uniform_(-1e-2, 1e-2))
        torch.nn.init.zeros_(self.encoder12.weight_mu)
        torch.nn.init.zeros_(self.encoder13.weight_mu)
        torch.nn.init.zeros_(self.encoder11.weight_mu)
        # self.apply(init_weights)
        
        
    def forward(self, x,Force,M):

        cof = torch.zeros((x.shape[0], 6))

        self.bias1.data.clamp_(-0.03, 0.03)
        self.bias2.data.clamp_(-0.03, 0.03)
        self.bias3.data.clamp_(-0.03, 0.03)
        self.encoder11.weight_mu.data.clamp_(-0.1, 0.1)
        self.encoder12.weight_mu.data.clamp_(-0.1, 0.1)
        self.encoder13.weight_mu.data.clamp_(-0.1, 0.1)
#notice: 以下Force应该是风洞气动表的force，即所有偏差全部为零跑出来的。
        cof[:, 0:1] = self.encoder11(torch.tensor(Force[:, 0]).unsqueeze(-1)) + torch.tensor(Force[:, 0]).unsqueeze(
            -1)+self.bias1
        cof[:, 1:2] = self.encoder12(torch.tensor(Force[:, 1]).unsqueeze(-1)) + torch.tensor(Force[:, 1]).unsqueeze(
            -1)+self.bias2
        cof[:, 2:3] = self.encoder13(torch.tensor(Force[:, 2]).unsqueeze(-1)) + torch.tensor(Force[:, 2]).unsqueeze(
            -1)+self.bias3

        w1_mean = self.encoder11.weight_mu.data
        w1_var = self.encoder11.weight_log_sigma.data
        w2_mean = self.encoder12.weight_mu.data
        w2_var = self.encoder12.weight_log_sigma.data
        w3_mean = self.encoder13.weight_mu.data
        w3_var = self.encoder13.weight_log_sigma.data
        b1_mean = self.bias1.data
        b2_mean = self.bias2.data
        b3_mean = self.bias3.data
        # cof是（1+Ω）*F + b  是估计的force
        # dx是神经网络算的增量，输出dx[0]:(8001, 18)
        dx = hypersonic_dobBased2(x[:, :21], x[:, 20], self.goal_data, Pid, z, mu, Vv, cof[:, :3], M)


        return dx,(w1_mean,w1_var,b1_mean,0,w2_mean,w2_var,b2_mean,0,w3_mean,w3_var,b3_mean,0)

import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
# 设置字体
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# if __name__ == '__main__':
#     data = np.load("C:\WorkFile\Python_File\Formal_project\\visualize_learning\集成神经网络\data0.npy")
#     dataset = data

#     X_train = torch.Tensor(dataset[0:8001, :])
#     X_test = torch.Tensor(dataset[:, :])
#     # print(dataset.shape)
#     Force, M, ddx = hypersonic_dobBased1(X_train[:, :21], X_train[:, 20], [1 / 57.3, 0, 0], Pid, z, mu, Vv,
#                                          [0.2,0,0.08, 0.02,0.1,0.03],0.3)
#     # Force1, M1, ddx1 = hypersonic_dobBased1(X_train[:, :21], X_train[:, 20], [1 / 57.3, 0, 0], Pid, z, mu, Vv,
#     #                                      [0,0,0, 0,0,0],0.3)
#     # Force2, M2, ddx2 = hypersonic_dobBased1(X_train[:, :21], X_train[:, 20], [1 / 57.3, 0, 0], Pid, z, mu, Vv,
#     #                                      [0.1,0.0003,0.0814, 0.0201,0.08,0.0292],0.3)
#     x0 = np.array([6/57.3, 1/57.3, 2/57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33589, 0, 5000, 0, 0, 0.03, 0, 0])

#     Wgd = 80
#     tf = 80
#     step = 0.1
#     tspan = np.arange(0, tf + step, step)
    
#     states1 = odeint(hypersonic_dobBased1_numpy_forDrawing, x0, tspan , args=( [1 / 57.3, 0, 0], Pid, z, mu, Vv ,[0.2,0,0.08, 0.02,0.1,0.03],0.3))
#     states2 = odeint(hypersonic_dobBased1_numpy_forDrawing, x0, tspan , args=( [1 / 57.3, 0, 0], Pid, z, mu, Vv ,[0,0,0, 0,0,0],0.3))
#     states3 = odeint(hypersonic_dobBased1_numpy_forDrawing, x0, tspan , args=( [1 / 57.3, 0, 0], Pid, z, mu, Vv ,[0.1,0.0003,0.0814, 0.0201,0.08,0.0292],0.3))


#     _,nums = np.shape(ddx)
#     nums_to_show = [0,1,2,3,4,5,12,13,14,15,16,17]
#     nums_label = ['俯仰角','滚转角','偏航角','绕x轴角速度','绕y轴角速度','绕z轴角速度','x轴位置','y轴位置','海拔','速度','弹道倾角','弹道偏角']
#     figures = []
#     for i, label in zip(nums_to_show, nums_label):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(states1[:, i], label='真实值', color='blue', linestyle='-')
#         ax.plot(states2[:, i], label='气动表值', color='green', linestyle='--')
#         ax.plot(states3[:, i], label='修正值', color='red', linestyle='-.')
#         ax.set_xlabel('时间')
#         ax.set_ylabel(label)
#         ax.set_title(f'{label} 对比曲线')
#         ax.legend()
#         figures.append(fig)

#     # 显示所有图像
#     for fig in figures:
#         fig.show()

#     # 保持所有窗口打开
#     plt.show()


