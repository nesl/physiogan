import os
import numpy as np
import scipy
import scipy.io as sio
import pickle
import argparse

if __name__ == '__main__':
    x_fpath = os.path.join('ecg_crnn/07_29_13_10_long_round_2', 'samples_x.npy')
    y_fpath = os.path.join('ecg_crnn/07_29_13_10_long_round_2', 'samples_y.npy')

    x_data = np.load(x_fpath)
    y_data = np.load(y_fpath)   
    '''
    # PKL Processing
    dataPath = os.path.join('ecg_long_pkl/test_ecg_long', 'test_ecg_long.pkl')
    with open( dataPath, "rb" ) as fh: 
        x_data, y_data = pickle.load(fh)
    '''
    for i in range(0,len(x_data)):
        sio.savemat("sample"+str(i)+"_"+str(y_data[i])+".mat", {'val':scipy.array(x_data[i,:,0],dtype='double')})

'''        
import pickle
import scipy.io
name = 'probe_features'  #probe without 1
p = open('%s.pkl'%name,'rb')
data = pickle.load(p)
dict = {}
dict['data'] = data
scipy.io.savemat(name,dict)
    for i in range(0,len(x_data)):
        line = ""
        #print(x_data[i,:,0].shape)
        sio.savemat("./ecg_rganar/ecgrganar"+str(i)+"_"+str(y_data[i])+".mat", {'val':scipy.array(x_data[i,:,0],dtype='double')}) 
        #label = "N"
        #if train_dataset.labels[i] == 1:
           #label = "A"
'''




args = parser.parse_args()
print args.accumulate(args.integers)