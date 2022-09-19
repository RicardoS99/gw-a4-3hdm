from __future__ import division
from __future__ import print_function
#from signal import pause
from tabnanny import verbose
from tokenize import String
from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd
import numpy as np
import random
import os
import math
import sys
import matplotlib.pyplot as plt
from scipy import linalg
import sympy as sp
import csv
import multiprocessing as mp
import pandas as pd

from A4_model import A4_vev1
from A4_model_gauge import A4_gauge_vev1
from A4_model_reduced import A4_reduced_vev1
from A4_spectrum import A4_spectrum
from MyProcessManager import MyProcessManager
from gw_spectrum import gw_spectrum
from helperTools import createPars, parsL, rndpars

def findTrans(queue, lock, file_name):
    pars = queue.get()
    #sys.stdout = open(os.devnull, 'w')
    m = A4_spectrum(Mn1 = pars[0], Mn2 = pars[1], Mch1 = pars[2], Mch2 = pars[3], verbose = 2, forcetrans=True, T_eps=1e-3, path = './bin/', betamax=1E6)
    if m.spectrainfo == []:
        m.genspec()
    m.save()
    #sys.stdout = sys.__stdout__

    with lock:
        for line in m.getinfo():
            with open(file_name, 'a', newline='') as csvfile:
                fieldnames = ['Mn1', 'Mn2', 'Mch1', 'Mch2', 'M0', 'L0', 'L1', 'L2', 'L3', 'L4', 'dM0', 'dL0', 'dL1', 'dL2', 'dL3', 'dL4', 'NPhases', 'NTrans', 'VEVdif/T', 'alpha', 'beta', 'JougetVel', 'WallVel', 'RadCrit', 'KCol', 'KSW', 'KTurb', 'TempCrit', 'TempNuc', 'FreqPeakCol', 'FreqPeakSW', 'FreqPeakTurb', 'AmpPeakCol', 'AmpPeakSW', 'AmpPeakTurb', 'Action', 'Action/Tnuc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(line)


def main():
    if len(sys.argv) > 1:
        inputfile = pd.read_csv(sys.argv[1])
        for index, input in inputfile.iterrows():
            file_name = input[0]
            box = [[float(input[1]),float(input[2]),float(input[3]),float(input[4])],[float(input[5]),float(input[6]),float(input[7]),float(input[8])]]
            nsamples = int(float(input[9])/0.3) #We divide by 30% because this is the approximate percentage of random points generated that lie on the valid space -> We have to oversample
            
            pars_list = rndpars(box,nsamples)
            #print(np.asanyarray(pars_list))
            print('List size: ', len(pars_list))
            
            queue = mp.Queue()
            lock = mp.Lock()

            print(pars_list)

            for pars in pars_list:
                queue.put(pars)

            with open(file_name, 'w', newline='') as csvfile:
                fieldnames = ['Mn1', 'Mn2', 'Mch1', 'Mch2', 'M0', 'L0', 'L1', 'L2', 'L3', 'L4', 'dM0', 'dL0', 'dL1', 'dL2', 'dL3', 'dL4', 'NPhases', 'NTrans', 'VEVdif/T', 'alpha', 'beta', 'JougetVel', 'WallVel', 'RadCrit', 'KCol', 'KSW', 'KTurb', 'TempCrit', 'TempNuc', 'FreqPeakCol', 'FreqPeakSW', 'FreqPeakTurb', 'AmpPeakCol', 'AmpPeakSW', 'AmpPeakTurb', 'Action', 'Action/Tnuc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            p = MyProcessManager(findTrans, queue, 1, (queue,lock,file_name), 1200, 10)
            p.run()

            """
            #sys.stdout = open(os.devnull, 'w')
            pool = mp.Pool(processes=16)
            output_list = pool.map(findTrans, pars_list)
            output_flat = [item for sublist in output_list for item in sublist if sublist != []]
            output_data = pd.DataFrame(output_flat)
            output_data.to_csv(file_name)
            #sys.stdout = sys.__stdout__
            print(file_name, ' saved!')
            """
    
if __name__ == '__main__':
  main()