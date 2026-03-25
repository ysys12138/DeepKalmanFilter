import sys

import numpy as np
import matplotlib
import re

import scipy

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def read_raw_force(path, filename):
    with open(path + filename+'.txt', 'r') as f:
        data = f.readlines()
    list = [[], []]
    for line in data:
        tmp = line.split(',')
        # if tmp[0].isnumeric():
        if not bool(re.search("[a-zA-Z]", tmp[0])):
            time = float(tmp[0])
            force = float(tmp[1])
            list[0].append(time)
            list[1].append(force)
    f.close()
    nplist = np.array(list)
    print(np.shape(nplist))
    np.save(path+'raw_force_'+filename, nplist)
    return nplist
    # print(list,np.shape(list))


def read_raw_ele(path, filename):
    with open(path + filename+'.csv', 'r') as f:
        data = f.readlines()  # read txt file
    list = [[], [], []]
    for line in data:
        tmp = line.strip().split(',')  # split the csv data
        # if it is truly a number
        if not bool(re.search("[a-zA-Z]", tmp[0])):
            time = float(tmp[0])
            res = float(tmp[1])
            piezo = float(tmp[2])
            list[0].append(time)
            list[1].append(res)
            list[2].append(piezo)
    f.close()
    nplist = np.array(list)
    print(np.shape(nplist))
    np.save(path+'raw_ele_'+filename, nplist)
    return nplist


def cut(npydata, starttime,fs):
    ret = npydata[:, starttime*fs:]
    return ret


def cut2(npydata, starttime, endtime,fs):
    ret = npydata[:, starttime*fs: endtime*fs]
    return ret

def interp():
    newfile = []
    path = './data/cutdata/S2/'
    filename = 'S2'
    afile = np.load(path + filename + '.npy')
    bfile = np.load(path + 'S2_Rohdaten' + '.npy')
    time = afile[0]
    force = afile[1]
    start = time[0]
    end = time[-1]
    newtimeline = np.linspace(start, end, len(time)*10)
    f = scipy.interpolate.interp1d(time, force)
    ret = f(newtimeline)
    newfile.append(newtimeline)
    newfile.append(ret)
    print(newfile)
    np.save(path+'Int+S2_ref.npy', newfile)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(force)
    plt.subplot(2,1,2)
    plt.plot(ret)
    plt.show()


def main(argv):
    filename = argv[1]   # txt file name here
    filename1 = filename + '_ref'  # txt file name here
    filename2 = filename+'_Rohdaten'   # scv file name here
    path = './data/rawdata/s2/'
    read_raw_force(path, filename1)
    read_raw_ele(path, filename2)

    force = np.load(path + 'raw_force_'+filename1+'.npy', allow_pickle=True)
    ele = np.load(path + 'raw_ele_'+filename2+'.npy', allow_pickle=True)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(force[0], force[1])
    plt.subplot(3, 1, 2)
    plt.plot(ele[0], ele[1])
    plt.subplot(3, 1, 3)
    plt.plot(ele[0], ele[2])
    plt.show()

    # cut(data, start point, sample frequency)
    cutforce = cut2(force, 0, -1, 50)
    cutele = cut2(ele, 0, -1, 500)
    # plot the cut data figure
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(cutforce[0], cutforce[1])
    plt.subplot(3, 1, 2)
    plt.plot(cutele[0], cutele[1])
    plt.subplot(3, 1, 3)
    plt.plot(cutele[0], cutele[2])
    plt.show()
    # save the numpy data in another folder
    savepath = './data/cutdata/S2/'
    np.save(savepath+filename, cutforce)
    np.save(savepath+filename2, cutele)
    interp()


if __name__ == '__main__':
    main(sys.argv)

