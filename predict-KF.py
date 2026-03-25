import csv
import os
import sys
from scipy import signal
import numpy as np
import matplotlib
import re
import torch
import torch.nn as nn
import time
from Kalman_NewModel_H_penalty_Noff import kalman_iter_batch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def lowpass_filter(data):
    b, a = signal.butter(8, 0.08, 'lowpass')
    filtered_Data = signal.filtfilt(b, a, data)  # low pass filter with order 8
    return filtered_Data


def tail_lines(filepath, n):
    with open(filepath, 'r') as f:
        data = f.readlines()
    # if data is shorter than n, just wait
    if len(data) < n:
        data = 0
    else:
        data = data[-n:]
    return data


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
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.tanh = nn.Tanh()

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

            force = torch.cat([force, forcen], dim=1)

        # offset = torch.tensor(0)

        # out is prediction force sequence

        force_predict = force.squeeze()
        return force_predict, l_n_out, h_n_out, xn_1  # returns result without extra dimension


def lineprocess(lines):
    result = [[], [], []]
    for line in lines:
        parts = line.strip().split(',')
        if not re.search('[a-zA-Z]', parts[0]):
            try:
                t = float(parts[0])
                r = float(parts[1])
                p = float(parts[2])
            except (ValueError, IndexError):
                continue
            result[0].append(t)
            result[1].append(r)
            result[2].append(p)
    vs = result[1]
    vd = result[2]
    result[1] = lowpass_filter(vs)
    result[2] = lowpass_filter(vd)
    return np.array(result)


def addoffset(data, n):
    l = len(data[0])
    ret = data.copy()
    tmp = data[1]
    ret[1] = tmp + np.ones(l) * n
    return ret


def predict(data, offset, model, x_init):
    model.eval()
    vs_vd = addoffset(data, offset)
    vs = vs_vd[1]
    vd = vs_vd[2]

    x1 = torch.tensor(vs, dtype=torch.float)
    x2 = torch.tensor(vd, dtype=torch.float)
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)

    f_predict, _, _, xn_1 = model(1, x_init, x1, x2)

    fp_numpy = f_predict.detach().numpy()
    # remove the first predict with too much noise
    fp_numpy[0] = fp_numpy[1]

    return fp_numpy, xn_1


def follow_and_save(path, filename, offset, model, delay, refresh):
    infile = os.path.join(path, filename + '.csv')
    outfile = os.path.join(path, f'{filename}-out.csv')
    currenttime = -1
    x_init = torch.zeros([1, 5, 1], dtype=torch.float32)
    try:
        while True:
            # read the last delay samples
            last_lines = tail_lines(infile, delay)
            while last_lines == 0:
                print('Waiting for more data.')
                time.sleep(2)
                last_lines = tail_lines(infile, delay)
            # low pass filter
            np_data = lineprocess(last_lines)
            starttime = np_data[0, 0]
            endtime = np_data[0, -1]
            if endtime <= currenttime:
                print('No further input data.')
                time.sleep(2)
            else:

                timeline = np.linspace(starttime, endtime, delay)
                index = (timeline > currenttime).argmax()
                newtime = timeline[index:]
                new_inputs = np_data[:, index:]
                f_p, x_out = predict(new_inputs, offset, model, x_init)
                newdata = np.transpose([newtime, f_p])
                currenttime = endtime
                x_init = x_out
                with open(outfile, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(newdata)
                print('Update success! Current time:', currenttime)
                time.sleep(refresh)
    except KeyboardInterrupt:
        print('Interrupt by keyboard.')

    return 0


def loadModel(checkpoint_path):
    model = KalmanNetwork(20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # if checkpoint exists, load states and global parameters and overwrite them
    training_step, rng_state, training_loss, model_sd, optimizer_sd = torch.load(checkpoint_path)
    torch.set_rng_state(
        rng_state)  # loads state
    model.load_state_dict(model_sd)  # overwrite parameters
    optimizer.load_state_dict(optimizer_sd)  # overwrite state of optimizer
    return model


def plot_result(filename):
    with open(filename + '-out.csv', 'r') as f:
        lines = f.readlines()
    result = [[], [], []]
    for line in lines:
        parts = line.strip().split(',')
        if not re.search('[a-zA-Z]', parts[0]):
            try:
                t = float(parts[0])
                r = float(parts[1])
            except (ValueError, IndexError):
                continue
            result[0].append(t)
            result[1].append(r)
    plt.plot(result[1])
    plt.show()


def main(argv):
    """
    Input:
    argv1(filename): name of the row data file, should be csv, this file should change over time.
    argv2(delay): For DL-KalmanNet, the warmup is not needed, so this parameter refers to how many points you would like to
    predict at once. 500 should be good enough
    argv3(refresh): Refresh every how many seconds, e.g. 0.5 This determines the true latency you can feel,
    can theoretically be set to 0.
    argv4(offset): key parameter, bias due to differences in environment and sensors.
    This parameter should result in a static measurement of approximately -0.302V at an applied force of 0

    Output:
    An output csv file contains the result after signal processing. This file can keep changing over time.
    """
    model = loadModel('./pt/checkpointKNet_batch_F0.795.pt')
    filename = argv[1]
    delay = int(argv[2])
    refresh = float(argv[3])
    offset = float(argv[4])
    # check if there is an existing file, clear it up.
    if os.path.isfile(filename + '-out.csv'):
        with open(filename + '-out.csv', 'w'):
            pass
    # data processing
    follow_and_save('', filename, offset, model, delay, refresh)

    # plot if needed
    plot_result(filename)


if __name__ == '__main__':
    main(sys.argv)
