import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

def calculate(data):
    
    data = np.load('3pass_1.npy')/100



    datacopy = data[0,:,:].copy()
    datacopy[730:,700:][datacopy[730:,700:]<170] = 0


    data[0,:,:] = data[0,:,:]/2
    data[0,:,0:1000][data[0,:,0:1000]<100]=0


    data[0,145:800,694:1000] = np.rot90(data[0,145:800,694:1000],2)

    #data[0,145:805,694:]

    data[0,5:665,244:550]=data[0,145:805,694:1000]

    data[0,145:805,694:1000]=0



    data[0,466:609,473:610]=data[0,466:609,353:490]
    data[0,466:609,353:490] = 0
    data[0,282:424,503:550] = 0

    data[0,730:,700:] = datacopy[730:,700:]











    data[2,500:800,:] = data[2,500:800,:]/100
    data[2,800:,:] = data[2,800:,:]/100
    data[2,230:600,0:200] = 0
    data[2,200:500,200:560] = data[2,200:500,200:560]/6

    data[1, :, :][data[2, :, :] < 1] = 0


    return data

