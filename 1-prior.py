import numpy as np
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import pickle

# geostat simulation and flow classes
from fluvial import FluvialSimulation
from tracer import TracerTest

import geone
import time

def sample_from_prior(i):
    #fluvial_simulation = FluvialSimulation(nthreads=2)
    fluvial_simulation = FluvialSimulation(nthreads=1)
    img_geone = fluvial_simulation.generate_with_hd(i)
    #with open('prior/img{}.pickle'.format(i), 'wb') as file_handle:
    #    pickle.dump(img_geone, file_handle)

def main():
    start=time.time()
    #Parallel(n_jobs=32)(delayed(sample_from_prior)(i) for i in range(1024))
    Parallel(n_jobs=63)(delayed(sample_from_prior)(i) for i in range(5*63))
    end = time.time()-start
    with open('time.pickle', 'wb') as fh:
        pickle.dump(end, fh)

if __name__=='__main__':
    main()
