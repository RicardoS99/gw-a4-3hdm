from __future__ import division
from __future__ import print_function
from signal import pause
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
from gw_spectrum import gw_spectrum

def findTrans(pars):
    m = A4_spectrum(Mn1 = pars[0], Mn2 = pars[1], Mch1 = pars[2], Mch2 = pars[3], verbose = 1, forcetrans=False, T_eps=5e-4, path = './bin/')
    m.genspec()
    m.geninfo()
    m.save()
    for line in m.info:
        with open('output/log.csv', 'w', newline='') as csvfile:
            fieldnames = ['Mn1', 'Mn2', 'Mch1', 'Mch2', 'M0', 'L0', 'L1', 'L2', 'L3', 'L4', 'dM0', 'dL0', 'dL1', 'dL2', 'dL3', 'dL4', 'NPhases', 'NTrans', 'VEVdif/T', 'alpha', 'beta', 'JougetVel', 'WallVel', 'RadCrit', 'KCol', 'KSW', 'KTurb', 'TempCrit', 'TempNuc', 'FreqPeakCol', 'FreqPeakSW', 'FreqPeakTurb', 'AmpPeakCol', 'AmpPeakSW', 'AmpPeakTurb', 'Action', 'Action/Tnuc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(line)
    return m.info

def createPars(box, n, isMasses=True): #isMasses=True: Mn1, Mn2, Mch1, Mch2; isMasses=False: Mn1+Mn2, Mn1-Mn2, Mch1+Mch2, Mch1-Mch2
    def preunitfilter(ms, Mh=125.1, vh=246.22):
        M0 = np.sqrt(3)/2.*Mh**2
        L1 = -(ms[2]**2 + ms[3]**2)/(vh**2)
        L4 = 2.*np.sqrt(3)*(ms[2]**2 - ms[3]**2)/(vh**2)
        L0 = np.sqrt(3)*M0/(vh**2) - L1
        xc1 = 6.*(ms[0]**2 + ms[1]**2)/(vh**2) + 5*L1
        xc2 = np.sqrt((6.*(ms[0]**2 - ms[1]**2)/(vh**2))**2 - 12*L4**2)
        L2 = 1/6.*(xc1 + xc2 + L1)
        L3 = 1/4.*(xc1 - xc2 - L1)

        req1 = np.abs(L0) < np.pi/2.
        req2 = np.abs(L1) < np.pi/2.
        req3 = np.abs(L2) < np.pi/2.
        req4 = np.abs(L3) < np.pi/2.
        req5 = np.abs(L4) < np.pi/2.
  
        cond = req1 and req2 and req3 and req4 and req5

        return cond

    print('Creating Parameters List...')
    Mn1 = np.linspace(box[0][0], box[-1][0], n[0], dtype=float)
    Mn2 = np.linspace(box[0][1], box[-1][1], n[1], dtype=float)
    Mch1 = np.linspace(box[0][2], box[-1][2], n[2], dtype=float)
    Mch2 = np.linspace(box[0][3], box[-1][3], n[3], dtype=float)
    pars = np.array(np.meshgrid(Mn1, Mn2, Mch1, Mch2)).T.reshape(-1, 4)
    if(isMasses is False):
        ind_to_remove = np.where(pars[...,3]== 0.)
        pars = np.delete(pars, ind_to_remove, 0)
        ind_to_remove = np.where(pars[...,0]-np.abs(pars[...,1])<0.)
        pars = np.delete(pars, ind_to_remove, 0)
        ind_to_remove = np.where(pars[...,2]-np.abs(pars[...,3])<0.)
        pars = np.delete(pars, ind_to_remove, 0)

        pars_temp = np.zeros_like(pars)
        pars_temp[...,0] = (pars[...,0] + pars[...,1])/2.
        pars_temp[...,1] = (pars[...,0] - pars[...,1])/2.
        pars_temp[...,2] = (pars[...,2] + pars[...,3])/2.
        pars_temp[...,3] = (pars[...,2] - pars[...,3])/2.
        pars = pars_temp


    print('Raw parameters list: ', pars.shape)
    pars = np.unique(pars, axis=0)
    ind_to_remove = np.where(np.abs(pars[...,2] - pars[...,3]) < 1)
    pars = np.delete(pars, ind_to_remove, 0)
        
    print('Unique parameters list shape before removing invalid masses: ', pars.shape)
    ind_to_remove = np.where((6.*(pars[...,0]**2 - pars[...,1]**2))**2 - 12*(2.*np.sqrt(3)*(pars[...,2]**2 - pars[...,3]**2))**2 <= 0. )
    pars = np.delete(pars, ind_to_remove, 0)
    print('Unique parameters list shape after removing invalid masses: ', pars.shape)
    pars_list = []
    for i in range(pars.shape[0]):
        if preunitfilter(pars[i]):
            pars_list.append(pars[i])

    print('Final list size: {0}'.format(len(pars_list)))
    return pars_list

def main():
    if len(sys.argv) > 1:
        inputfile = pd.read_csv(sys.argv[1])
        for index, input in inputfile.iterrows():
            file_name = input[0]
            box = [[float(input[1]),float(input[2]),float(input[3]),float(input[4])],[float(input[5]),float(input[6]),float(input[7]),float(input[8])]]
            divs = [int(input[9]),int(input[10]),int(input[11]),int(input[12])]
            massFlag = input[13]
            
            pars_list = createPars(box,divs, isMasses=massFlag)
            #print(np.asanyarray(pars_list))
            
            with open('output/log.csv', 'w', newline='') as csvfile:
                fieldnames = ['Mn1', 'Mn2', 'Mch1', 'Mch2', 'M0', 'L0', 'L1', 'L2', 'L3', 'L4', 'dM0', 'dL0', 'dL1', 'dL2', 'dL3', 'dL4', 'NPhases', 'NTrans', 'VEVdif/T', 'alpha', 'beta', 'JougetVel', 'WallVel', 'RadCrit', 'KCol', 'KSW', 'KTurb', 'TempCrit', 'TempNuc', 'FreqPeakCol', 'FreqPeakSW', 'FreqPeakTurb', 'AmpPeakCol', 'AmpPeakSW', 'AmpPeakTurb', 'Action', 'Action/Tnuc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            #sys.stdout = open(os.devnull, 'w')
            pool = mp.Pool(processes=1)
            output_list = pool.map(findTrans, pars_list)
            output_flat = [item for sublist in output_list for item in sublist if sublist != []]
            output_data = pd.DataFrame(output_flat)
            output_data.to_csv(file_name)
            #sys.stdout = sys.__stdout__
            print(file_name, ' saved!')
    
if __name__ == '__main__':
  main()