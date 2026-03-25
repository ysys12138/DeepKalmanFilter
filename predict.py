import csv
import os
import sys
from scipy import signal
import numpy as np
import matplotlib
import re
import torch
from cnn_causal_newData_F import TCN
import time

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


def predict(data, offset, model):
    model.eval()
    vs_vd = addoffset(data, offset)
    vs = vs_vd[1]
    vd = vs_vd[2]

    x1 = torch.tensor(vs, dtype=torch.float)
    x2 = torch.tensor(vd, dtype=torch.float)
    x1 = x1.unsqueeze(0)
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    x2 = x2.unsqueeze(0)

    combined_input = torch.cat((x1, x2), dim=1)
    print(x1.shape)
    print(x2.shape)
    print(combined_input.shape)
    output = model(combined_input)
    fp_numpy = output.detach().numpy()

    return fp_numpy


def follow_and_save(path, filename, offset, model, delay, refresh):
    infile = os.path.join(path, filename + '.csv')
    outfile = os.path.join(path, f'{filename}-out.csv')
    currenttime = -1

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
                f_p = predict(np_data, offset, model)
                newdata = np.transpose([timeline, f_p])
                index = (timeline > currenttime).argmax()
                newdata = newdata[index:, :]
                currenttime = endtime
                print(newdata)
                with open(outfile, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(newdata)
                time.sleep(refresh)
    except KeyboardInterrupt:
        print('Interrupt by keyboard.')

    return 0


def loadModel(checkpoint_path):
    model_params = {
        # parameters of TCN
        'input_size': 2,
        'output_size': 1,
        'num_channels': [32] * 8,
        'kernel_size': 3,
        'dropout': 0
    }
    model = TCN(**model_params)
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
    argv2(delay): Warmup time before started, e.g. 20000 refers to 40 seconds, which means you need to wait 40
    seconds before started, This is just a single wait at the beginning. Subsequent outputs are in real time and will
    only be delayed by much less than 0.1s.(20000 is the best choice)
    argv3(refresh): Refresh every how many seconds, e.g. 0.5 This determines the true latency you can feel,
    can theoretically be set to 0.
    argv4(offset): key parameter, bias due to differences in environment and sensors.
    This parameter should result in a static measurement of approximately -0.302V at an applied force of 0

    Output:
    An output csv file contains the result after signal processing. This file can keep changing over time.
    """
    model = loadModel('./pt/checkpointTCNet_F10.04.pt')
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
