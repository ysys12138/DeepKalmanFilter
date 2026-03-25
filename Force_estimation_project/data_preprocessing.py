import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

def cut_into_timepieces(n, data, period):
    # cut into time period
    cur = 0
    end = len(data[0]) - period
    # print(len(data[0]))
    parts = []
    n = min(n, int(end/period+2))
    for i in range(n):

        if cur + period >= len(data[0]):
            break
        tmp = data[:, cur:cur + period]
        parts.append(tmp)
        cur = cur + int(end / n)
    ret = np.asarray(parts)
    return ret

def standard_scalar(data):
    scaler_ss = StandardScaler()
    result_ss = scaler_ss.fit_transform(data)
    return result_ss

def offset(data, n):
    l = len(data[0])
    ret = data.copy()
    tmp = data[1]
    ret[1] = tmp - np.ones(l)*n
    return ret

def data_prepor(sensorname):
    data_length = 20000
    path = './data/cutdata/S2/'
    data_p = []
    data_r = []
    data_p.append(np.load(path + 'Int+'+ sensorname +'_ref.npy'))
    data_r.append(offset(np.load(path + sensorname + '_Rohdaten.npy'), 0))
    #data_p.append(np.load(path + '/syn+002_random.npy'))
    #data_r.append(np.load(path + '/syn+Flt+002_random.npy'))

    #  validation set
    data_random_P = np.load(path + 'Int+'+sensorname+'_ref.npy')
    data_random_R = np.load(path + sensorname + '_Rohdaten.npy')
    lists_force = []
    lists_resistance = []
    lists_piezo = []
    #  cut into timepieces
    for data in data_p:
        p_sets = cut_into_timepieces(40, data, 20000)
        print(np.shape(p_sets))
        lists_force.extend((p_sets[:, 1]))
    for data in data_r:
        r_sets = cut_into_timepieces(40, data, 20000)
        print(np.shape(r_sets))
        lists_resistance.extend((r_sets[:, 1]))
        lists_piezo.extend((r_sets[:, 2]))
    print(np.shape(lists_resistance))
    print(np.shape(lists_piezo))
    print(np.shape(lists_force))
    vd_P_sets = cut_into_timepieces(3, data_random_P, data_length)
    vd_R_sets = cut_into_timepieces(3, data_random_R, data_length)
    l_resistance = vd_R_sets[:, 1]
    l_piezo = vd_R_sets[:, 2]
    l_force = vd_P_sets[:, 1]
    path = './data/npydata/dataset/'+sensorname
    np.save(path + 'res_data_real.npy', lists_resistance)
    np.save(path + 'filter+piezo_data_real.npy', lists_piezo)
    np.save(path + 'force_data_real.npy', lists_force)
    np.save(path + 'res_data_val.npy', l_resistance)
    np.save(path + 'filter+piezo_data_val.npy', l_piezo)
    np.save(path + 'force_data_val.npy', l_force)

def main(argv):
    data_prepor(argv[1])

if __name__ == '__main__':
    main(sys.argv)
