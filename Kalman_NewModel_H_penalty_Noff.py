# -*- coding: utf-8 -*-
import gc
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import time
matplotlib.use('TkAgg')
from scipy.optimize import minimize

device = "cpu"


def callback(xk):
    print(xk)


"""
X(k) = FX(k-1) + BU(k) + w(k-1)
Z(k) = HX(k) + e(k)
p(w) = N(0, Q)
p(e) = N(0, R)
"""


def load_data(offset):
    path = r'./data/npydata/elek/SDK002'
    f = np.load(path + '/syn+force+S1_SDK_002_longrun.npy')
    force = f[1]

    elek = np.load(path + '/syn+flt+S1_SDK_002_longrun.npy')
    r = elek[1]
    r += np.ones(len(force)) * offset
    p = elek[2]
    # print('======force resistance piezo=====', np.shape(force), np.shape(r), np.shape(p))

    return force, r, p


def load_data_tensor(offset):
    f, r, p = load_data(offset)
    f_tensor = torch.tensor(f, dtype=torch.float, requires_grad=False)
    r_tensor = torch.tensor(r, dtype=torch.float, requires_grad=False)
    p_tensor = torch.tensor(p, dtype=torch.float, requires_grad=False)
    return f_tensor, r_tensor, p_tensor


def load_dataset(vs, vd, offset):
    vs = torch.ones_like(vs) * offset + vs.clone()
    return vs, vd


def kf_predict(X0, P0, A, Q, W):
    X10 = torch.matmul(A, X0) + W
    P10 = torch.matmul(torch.matmul(A, P0), A.transpose(-2, -1)) + Q
    return X10, P10


def kf_update(X10, P10, Z, H, R):
    batch_size, nx, _ = P10.shape
    V = Z - torch.matmul(H, X10)
    S = torch.matmul(torch.matmul(H, P10), H.transpose(-2, -1)) + R
    S = torch.where(torch.isnan(S), torch.full_like(S, 1e-6), S)  # 把 Inf 替换成 1e6

    K = torch.matmul(torch.matmul(P10, H.transpose(-2, -1)), torch.linalg.pinv(S))
    I = torch.eye(nx).repeat(batch_size, 1, 1)

    X1 = X10 + torch.matmul(K, V)

    P1 = torch.matmul(I - torch.matmul(K, H), P10)

    return X1, P1, K


l = 2000


def kalman_iter_batch(batch_size, xn_1, pn_1, vsn, vdn, kn, k2n, k3n, theta1, theta2, varf, varR, varP, offset, thetah):
    nx = 5
    thetah = thetah * 1e-4
    R = torch.diag_embed(torch.stack([varR.squeeze(-1), varP.squeeze(-1)], dim=-1))

    vs_obs, vd_obs = vsn + offset, vdn

    x_obs = torch.cat((vs_obs, vd_obs), dim=1)
    Zi = x_obs.unsqueeze(-1)
    # print(Zi)
    W = torch.zeros(batch_size, nx, 1, dtype=torch.float)

    W[:, -1, :] = thetah
    # W = [0,0,0,0,thetah]T

    k_comn = 1.0 / (kn + k2n + k3n)

    # row1: [1, 0, 0, 0, 0]
    row1 = torch.cat([torch.ones_like(k_comn),
                      torch.zeros_like(k_comn), torch.zeros_like(k_comn), torch.zeros_like(k_comn),
                      torch.zeros_like(k_comn)], dim=1)
    # row2: [k_comn, k_comn*(k2n+2*k3n), k_comn*k3n, 0, 0]
    row2 = torch.cat([k_comn,
                      k_comn * (k2n + 2 * k3n),
                      k_comn * k3n,
                      torch.zeros_like(k_comn), torch.zeros_like(k_comn)], dim=1)
    # row3: [0, 1, 0, 0, 0]
    row3 = torch.cat([torch.zeros(batch_size, 1, device=device),
                      torch.ones(batch_size, 1, device=device),
                      torch.zeros(batch_size, 3, device=device), ], dim=1)
    # row4: [0, 0, 1, 0, 0]
    row4 = torch.cat([torch.zeros(batch_size, 2, device=device),
                      torch.ones(batch_size, 1, device=device),
                      torch.zeros(batch_size, 2, device=device)], dim=1)
    row5 = torch.cat([torch.zeros(batch_size, 4, device=device),
                      torch.ones(batch_size, 1, device=device)], dim=1)
    A = torch.stack([row1, row2, row3, row4, row5], dim=1)  # shape [batch_size, 5, 5]

    # 构造过程噪声协方差矩阵 Q，结果 shape [batch_size, nx, nx]
    # row1: [1, k_comn, 0, 0, 0]
    q_row1 = torch.cat([torch.ones_like(k_comn), k_comn,
                        torch.zeros_like(k_comn), torch.zeros_like(k_comn),torch.zeros_like(k_comn)], dim=1)
    # row2: [k_comn, k_comn**2, 0, 0, 0]
    q_row2 = torch.cat([k_comn, k_comn ** 2,
                        torch.zeros_like(k_comn), torch.zeros_like(k_comn), torch.zeros_like(k_comn)], dim=1)
    # row3, row4 为全 0
    q_row3 = torch.zeros(batch_size, nx, device=device, dtype=torch.float32)
    q_row4 = torch.zeros(batch_size, nx, device=device, dtype=torch.float32)
    q_row5 = torch.zeros(batch_size, nx, device=device, dtype=torch.float32)
    Q = torch.stack([q_row1, q_row2, q_row3, q_row4, q_row5], dim=1) * varf  # shape [batch_size, 5, 5]

    # 构造观测矩阵 Hi，结果 shape [batch_size, 2, nx]
    # row1: [0, theta1, 0, 0, 0]
    hi_row1 = torch.cat([torch.zeros_like(k_comn),
                         theta1,
                         torch.zeros_like(k_comn),
                         torch.zeros_like(k_comn),
                         torch.zeros_like(k_comn)], dim=1)
    # row2: [0, theta2*(k2n+k3n), -theta2*k2n - 2*theta2*k3n, theta2*k3n, 0]
    hi_row2 = torch.cat([torch.zeros_like(k_comn),
                         theta2 * (k2n + k3n),
                         -theta2 * k2n - 2 * theta2 * k3n,
                         theta2 * k3n,
                         torch.zeros_like(k_comn)], dim=1)
    Hi = torch.stack([hi_row1, hi_row2], dim=1)  # shape [batch_size, 2, 5]

    # 重塑上一时刻状态与协方差（假设 xn_1 的形状为 [batch_size, nx, 1]，pn_1 为 [batch_size, nx, nx]）
    Xi = xn_1.view(batch_size, nx, 1)
    Pi = pn_1

    # 调用卡尔曼滤波预测与更新（请确保 kf_predict 与 kf_update 支持 batch 输入）
    X10, P10 = kf_predict(Xi, Pi, A, Q, W)
    X1, P1, K = kf_update(X10, P10, Zi, Hi, R)
    return X1, P1


class MyDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.f_true, self.vs, self.vd = load_data(0)

        # self.y = torch.flatten(self.y, start_dim=1, end_dim=2)
        self.stride = 2000
        self.num = 15

    def __getitem__(self, index):
        start = index * self.stride
        end = start + 2000
        vs_slice = self.vs[start:end]
        vd_slice = self.vd[start:end]
        f_true_s = self.f_true[start:end]
        return start, vs_slice, vd_slice, f_true_s

    def __len__(self):
        return self.num


# device = "cuda" if torch.cuda.is_available() else "cpu"



class KalmanNetwork(nn.Module):
    def __init__(self, size):
        """
        Class constructor that defines necessary parameters.
        :param size: Hidden layer dimension
        """
        super(KalmanNetwork, self).__init__()
        self.offset = nn.Parameter(torch.ones(1, dtype=torch.float) * 0.40)
        self.theta2 = nn.Parameter(torch.ones(1, dtype=torch.float) * 1.05)
        self.varR = nn.Parameter(torch.ones(1, dtype=torch.float) * 2.5)
        self.varP = nn.Parameter(torch.ones(1, dtype=torch.float) * 0.5)
        self.varf = nn.Parameter(torch.ones(1, dtype=torch.float) * 1.5)
        self.linear = nn.Linear(2, 64)
        self.linear2 = nn.Linear(64, 1)
        self.hlayer = nn.Linear(2, 32)
        self.hlayerout = nn.Linear(32, 1)

        self.linear3 = nn.Linear(1, 64)
        self.linear4 = nn.Linear(64, 1)
        self.linear5 = nn.Linear(1, 64)
        self.linear6 = nn.Linear(64, 1)
        self.linear7 = nn.Linear(1, 64)
        self.linear8 = nn.Linear(64, 1)

        nn.init.constant_(self.linear2.weight, -15.5 / 64)
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.linear4.weight, 2 / 64)
        nn.init.constant_(self.linear4.bias, 0)
        nn.init.constant_(self.linear6.weight, 1.5 / 64)
        nn.init.constant_(self.linear6.bias, 0)
        nn.init.constant_(self.linear8.weight, -1.5 / 64)
        nn.init.constant_(self.linear8.bias, 0)
        nn.init.constant_(self.hlayerout.weight, 0.001)
        nn.init.constant_(self.hlayerout.bias, 0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.tanh = nn.Tanh()

        # Layer 1 parameter
        self.to(device)

    def forward(self, batch_size, x_init, vs, vd):
        xn_1 = x_init

        P0 = torch.eye(5).repeat(batch_size, 1, 1) * 0.02
        force = x_init[:, 0, :]
        penalty_loss = torch.tensor([0], dtype=torch.float)
        vs = vs.float()
        vd = vd.float()
        length = len(vs[0])
        l_n_out = np.zeros([batch_size, length, 1])
        h_n_out = np.zeros([batch_size, length, 1])

        for i in range(1, length):
            vsn = vs[:, i].unsqueeze(1) * 100 + 30
            vdn = vd[:, i].unsqueeze(1) * 50
            l_n = xn_1[:, 1, :]
            h_n = xn_1[:, 4, :]
            combined_lh = torch.cat([l_n, h_n], dim=1)
            l_n_out[:, i] = l_n.detach().numpy()
            h_n_out[:, i] = h_n.detach().numpy()
            hiden_k = self.relu(self.linear(combined_lh))
            kn = self.linear2(hiden_k)
            hiden_theta = self.relu(self.linear3(l_n))
            theta1 = self.linear4(hiden_theta)
            # tanh as activation
            hidden_h = self.tanh(self.bn(self.hlayer(combined_lh)))
            theta_h = self.hlayerout(hidden_h)

            hiden_k = self.relu(self.linear5(l_n))
            k2 = self.linear6(hiden_k)
            hiden_k = self.relu(self.linear7(l_n))
            k3 = self.linear8(hiden_k)

            xn_1, P0 = kalman_iter_batch(batch_size, xn_1, P0, vsn, vdn, kn, k2, k3, theta1, self.theta2, self.varf,
                                         self.varR, self.varP, self.offset, theta_h)

            forcen = xn_1[:, 0, :]
            #forcen = forcen-torch.mul(kn, h_n)
            force = torch.cat([force, forcen], dim=1)
            epsilon = 0.1
            lambda_penalty = 0.5

            condition_mask = (torch.abs(l_n) < epsilon) | (l_n > 0)

            penalty = torch.where(condition_mask, torch.relu(theta_h) ** 2, torch.zeros_like(theta_h))
            penalty_lossn = penalty.mean()*lambda_penalty
            penalty_loss = penalty_lossn + penalty_loss

        # offset = torch.tensor(0)

        # out is prediction force sequence

        force_predict = force.squeeze()
        return force_predict, l_n_out, h_n_out, penalty_loss  # returns result without extra dimension


F = 0.795




def train(model, batch_size, dataloader, warmup=False):
    x_init = torch.zeros([batch_size, 5, 1], dtype=torch.float32)
    avg_loss = 0
    # Warm-up steps
    if warmup:
        model.eval()
        with torch.no_grad():
            _, vs_all, vd_all = load_data_tensor(0)
            vs_all, vd_all = vs_all.unsqueeze(0), vd_all.unsqueeze(0)
            xn_1_zeros = torch.zeros([1, 5, 1], dtype=torch.float)
            _, l_all, h_all, _ = model(1, xn_1_zeros, vs_all, vd_all)
            l_all = l_all[0].squeeze(-1)
            h_all = h_all[0].squeeze(-1)
    model.train()
    for start, vs, vd, f_true in dataloader:
        if warmup:
            index = 0
            start_points = start.detach().numpy()
            for s in start_points:
                x_init[index] = torch.tensor(
                    [f_true[index][0], l_all[s], l_all[s - 1] if s > 0 else 0, l_all[s - 2] if s > 0 else 0, h_all[s]],
                    requires_grad=False, dtype=torch.float).reshape(5, 1)
                index += 1
        # Train steps
        f_predict, _, _, penalty_loss = model(batch_size, x_init, vs, vd)

        loss_fn = torch.nn.L1Loss()
        f_true = f_true.float()
        loss = loss_fn(f_true, f_predict)
        loss = loss + penalty_loss
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name} doesn't have a gradient.")

        optimizer.step()
        print('batch ', start, 'loss=', loss.item(),'penalty_loss=', penalty_loss.item())
    avg_loss = avg_loss / len(dataloader)

    return avg_loss


def load_data_val(offset):
    path = r'./data/npydata/elek/SDK002'
    f = np.load(path + '/syn+force+S1_SDK_002_allinone.npy')
    force = f[1]

    elek = np.load(path + '/syn+flt+S1_SDK_002_allinone.npy')
    r = elek[1]
    r += np.ones(len(force)) * offset
    p = elek[2]
    # print('======force resistance piezo=====', np.shape(force), np.shape(r), np.shape(p))

    return force, r, p


def val(model, epoch):
    model.eval()
    x_init = torch.zeros([1, 5, 1], dtype=torch.float32)
    f_true, vs_numpy, vd_numpy = load_data_val(0.0044)

    vs_tensor = torch.tensor(vs_numpy).float().unsqueeze(0)
    vd_tensor = torch.tensor(vd_numpy).float().unsqueeze(0)
    start = time.time()
    f_predict, _, h, _ = model(1, x_init, vs_tensor, vd_tensor)
    f_p = f_predict.detach().numpy()
    loss_val = np.mean(np.abs(f_p - f_true))
    end = time.time()
    print('run time=', end-start)
    plt.figure()
    plt.plot(f_p)
    plt.plot(f_true)
    plt.plot(h[0]*50)
    plt.text(x=4,
             y=4,
             s='loss=' + str(loss_val))
    plt.savefig('pt_' + str(F) + 'Epoch' + str(epoch) + 'val_Fig' + '.jpg')
    plt.show()

    plt.close('all')
    return loss_val

if __name__ == '__main__':
    """Small model
    cn -> ac predict
    """
    with open('Lossdata' + str(F) + '.txt', 'w') as f:
        f.write('Start training')
    f.close()
    model = KalmanNetwork(20)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    lr = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)

    checkpoint_path = "checkpointKNet_batch_F{}.pt".format(F)
    if os.path.isfile(checkpoint_path):  # checks whether checkpoint exists
        # if checkpoint exists, load states and global parameters and overwrite them
        training_step, rng_state, training_loss, model_sd, optimizer_sd = torch.load(checkpoint_path)
        torch.set_rng_state(
            rng_state)  # loads state of RNG to have consistent results regardless of training interruptions
        model.load_state_dict(model_sd)  # overwrite parameters of GCN
        optimizer.load_state_dict(optimizer_sd)  # overwrite state of optimizer
    else:
        # otherwise, use initial parameters
        training_step = 0
        training_loss = []
    max_train_steps = 112
    dataset = MyDataset()
    batch_size = 5
    dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for training_step in range(training_step, max_train_steps):
        #print(val(model, training_step))
        if training_step < 5:
            loss = train(model, batch_size, dataload)
        else:
            loss = train(model, batch_size, dataload, warmup=True)

        lr.step()
        print('epoch', training_step, 'loss=', loss)
        training_loss.append(loss)

        with open('Lossdata' + str(F) + '.txt', 'a') as f:
            f.write('epoch' + str(training_step) + 'train loss: ' + str(loss) + '\n')

        torch.save([training_step + 1,
                    torch.get_rng_state(),  # rng state saved for consistent results
                    training_loss,  # list tracking training progress saved as well
                    model.state_dict(),
                    optimizer.state_dict()],
                   checkpoint_path)
        if training_step % 20 == 11:
            val_loss = val(model, training_step)
            with open('Lossdata' + str(F) + '.txt', 'a') as f:
                f.write('validation loss: ' + str(val_loss) + '\n')
            print('val loss=', val_loss)

    f.close()
