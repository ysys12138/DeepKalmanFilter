import sys

import scipy.io as io
import scipy
import os
# os.environ['MPLCONFIGDIR'] = './mtl'
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def getPath():
    paths = os.walk(r'./data/npydata/elek')
    files = []
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst:
            files.append(os.path.join(path, file_name))
    return files

    # print(np.shape(data_drift_P))


def zero_padding(data, width, length):
    tmp = np.zeros((width, length))
    tmp[:, 0:len(data[0])] = data
    return tmp



from torch.nn.utils import weight_norm


# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size  # 这个chomp_size就是padding的值

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# 最后就是TCN的主网络了
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x).transpose(1, 2)  # [N,C_out,L_out=L_in]
        y = self.linear(y)
        return y.squeeze()



def offsetcal(data, n):
    l = len(data[0])
    ret = data.copy()
    tmp = data[1]
    ret[1] = tmp - np.ones(l)*n
    return ret

def val_long(model, sensorname, of):
    model.eval()
    path = './data/cutdata/S2/'
    f_ture_long = np.load(path + 'Int+'+sensorname+'_ref.npy')
    vs_vd = offsetcal(np.load(path + sensorname+'_Rohdaten.npy'), of)
    print(np.shape(vs_vd))
    f_ture = f_ture_long[1]
    vs = vs_vd[1]
    vd = vs_vd[2]


    x1 = torch.tensor(vs, dtype=torch.float)
    x2 = torch.tensor(vd, dtype=torch.float)
    x1 = x1.unsqueeze(0)
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    y = torch.tensor(f_ture, dtype=torch.float)
    combined_input = torch.cat((x1, x2), dim=1)

    print(x1.shape)
    print(x2.shape)
    print(combined_input.shape)
    output = model(combined_input)
    min_len = min(len(output), len(y))
    output = output[:min_len]
    y = y[:min_len]
    fp_numpy = output.detach().numpy()
    np.savetxt("./result/fp_numpy_TCN_" + sensorname + '_Result.txt', fp_numpy)
    loss = loss_function(output, y)
    print('val loss='+str(loss))
    plt.figure()
    plt.plot(fp_numpy)
    plt.plot(f_ture)
    plt.text(x=4,
             y=4,
             s='loss=' + str(loss.item()))
    plt.ylabel('Force(N)')
    plt.legend(['Prediction', 'Reference'])
    plt.show()

    return loss

if __name__ == '__main__':
    '''
    argv[1]: sensor name e.g. S2_schime
    argv[2]: F pretrained model label e.g. 11.01
    argv[3]: offset e.g. 0.03
    input data should be in folder './data/cutdata/S2'
    output in in folder './result'
    '''
    sensorname = sys.argv[1]
    F = sys.argv[2]
    offset = float(sys.argv[3])
    print(torch.__version__)
    print(torch.cuda.is_available())

    model_params = {
        # 'input_size',C_in
        'input_size': 2,
        # 单步，预测未来一个时刻
        'output_size': 1,
        'num_channels': [32] * 8,
        'kernel_size': 3,
        'dropout': 0
    }
    model = TCN(**model_params)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sct = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.6, last_epoch=-1)
    checkpoint_path = "./pt/checkpointTCNet_F{}.pt".format(F)

    if os.path.isfile(checkpoint_path):  # checks whether checkpoint exists
        # if checkpoint exists, load states and global parameters and overwrite them
        training_step, rng_state, training_loss, model_sd, optimizer_sd = torch.load(checkpoint_path)
        torch.set_rng_state(
            rng_state)  # loads state of RNG to have consistent results regardless of training interruptions
        model.load_state_dict(model_sd)  # overwrite parameters of GCN
        optimizer.load_state_dict(optimizer_sd)  # overwrite state of optimizer
    else:
        print('Error: Cannot find the model.')

    loss = val_long(model, sensorname, offset)
    with open('./result/dataTCN-' + 'F' + str(F) + '.txt', 'w') as f:
        f.write('loss:' + str(loss))
    f.close()

