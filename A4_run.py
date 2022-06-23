from __future__ import division
from __future__ import print_function
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
from gw_spectrum import gw_spectrum

def findTrans(pars):

    output = []
    inMn1 = pars[0]
    inMn2 = pars[1]
    inMch1 = pars[2]
    inMch2 = pars[3]

    m = A4_vev1(Mn1=inMn1,Mn2=inMn2,Mch1=inMch1,Mch2=inMch2)

    n_trans = 0
   
    try:
        m.findAllTransitions()
        n_phases = len(m.phases)
        n_trans = len(m.TnTrans)
        model_info = {
            "Mn1": inMn1,
            "Mn2": inMn2,
            "Mch1": inMch1,
            "Mch2": inMch2,
            "L1": m.L1,
            "L2": m.L2,
            "L3": m.L3,
            "L4": m.L4,
            "NPhases": n_phases,
            "NTrans": n_trans
        }
        if(len(m.TnTrans)>0):
            for ind in range(0,len(m.TnTrans)):
                if(m.TnTrans[0]['trantype']==1): 
                    gw = gw_spectrum(m, ind, turb_on=True)
                    if(10 < gw.beta < 100000):
                        output.append(model_info | gw.info)
    except:
        pass
                
    return output

def createPars(box, n, isMasses=True): #isMasses=True: Mn1, Mn2, Mch1, Mch2; isMasses=False: Mn1+Mn2, Mn1-Mn2, Mch1+Mch2, Mch1-Mch2
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
    ind_to_remove = np.where((pars[...,0] - pars[...,1]) < 0)
    pars = np.delete(pars, ind_to_remove, 0)
        
    print('Unique parameters list shape before removing invalid masses: ', pars.shape)
    ind_to_remove = np.where((6.*(pars[...,0]**2 - pars[...,1]**2))**2 - 12*(2.*np.sqrt(3)*(pars[...,2]**2 - pars[...,3]**2))**2 <= 0. )
    pars = np.delete(pars, ind_to_remove, 0)
    print('Unique parameters list shape after removing invalid masses: ', pars.shape)

    return pars

def main():
    if len(sys.argv) > 1:
        inputfile = pd.read_csv(sys.argv[1])
        print(inputfile.iterrows())

        for index, input in inputfile.iterrows():
            file_name = input[0]

            print('input[1]', float(input[1]))

            box = [[float(input[1]),float(input[2]),float(input[3]),float(input[4])],[float(input[5]),float(input[6]),float(input[7]),float(input[8])]]
            divs = [int(input[9]),int(input[10]),int(input[11]),int(input[12])]
            massFlag = True if input[13]=='True' else False
            
            pars_list = createPars(box,divs, isMasses=massFlag)
            print(pars_list)
            #print(np.where((6.*(pars_list[...,0]**2 - pars_list[...,1]**2))**2 - 12*(2.*np.sqrt(3)*(pars_list[...,2]**2 - pars_list[...,3]**2))**2 <= 0. ))
            
            sys.stdout = open(os.devnull, 'w')
            pool = mp.Pool()
            output_list = pool.map(findTrans, list(pars_list))
            output_flat = [item for sublist in output_list for item in sublist if sublist != []]
            output_data = pd.DataFrame(output_flat)
            output_data.to_csv(file_name)
            sys.stdout = sys.__stdout__
            print(file_name, " saved!")
    
if __name__ == "__main__":
  main()