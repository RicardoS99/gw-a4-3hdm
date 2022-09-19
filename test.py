from __future__ import division
from __future__ import print_function
from fileinput import filename
import time
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
from py import process
from scipy import linalg
import sympy as sp
import csv
import multiprocessing as mp
import pandas as pd

from A4_full import A4full_vev1
from A4_model import A4_vev1
from A4_model_gauge import A4_gauge_vev1
from A4_model_reduced import A4_reduced_vev1
from MyProcessManager import MyProcessManager
from gw_spectrum import gw_spectrum
from A4_spectrum import A4_spectrum

from progress.bar import Bar

#sys.path.append('/src')

def findTrans(queue, lock, file_name):
    pars = queue.get()
    #sys.stdout = open(os.devnull, 'w')
    m = A4_spectrum(Mn1 = pars[0], Mn2 = pars[1], Mch1 = pars[2], Mch2 = pars[3], verbose = 2, forcetrans=False, path = './bin/')
    if m.spectrainfo == []:
        m.genspec()
    #m.save()
    #sys.stdout = sys.__stdout__

    with lock:
        for line in m.getinfo():
            with open(file_name, 'a', newline='') as csvfile:
                fieldnames = ['Mn1', 'Mn2', 'Mch1', 'Mch2', 'M0', 'L0', 'L1', 'L2', 'L3', 'L4', 'dM0', 'dL0', 'dL1', 'dL2', 'dL3', 'dL4', 'NPhases', 'NTrans', 'VEVdif/T', 'alpha', 'beta', 'JougetVel', 'WallVel', 'RadCrit', 'KCol', 'KSW', 'KTurb', 'TempCrit', 'TempNuc', 'FreqPeakCol', 'FreqPeakSW', 'FreqPeakTurb', 'AmpPeakCol', 'AmpPeakSW', 'AmpPeakTurb', 'Action', 'Action/Tnuc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(line)


def main():
    vh = 246.22
    L1 = -1.5
    L2 = 1.1
    L3 = 0.3
    L4 = -0.1
    Mn1 = np.sqrt( vh**2/12 *(-5*L1+3*L2+2*L3 + np.sqrt((-L1+3*L2-2*L3)**2 + 12*L4**2)))
    Mn2 = np.sqrt( vh**2/12 *(-5*L1+3*L2+2*L3 - np.sqrt((-L1+3*L2-2*L3)**2 + 12*L4**2)))
    Mch1 = np.sqrt( vh**2 * (-L1/2 + L4/(4*np.sqrt(3))))
    Mch2 = np.sqrt( vh**2 * (-L1/2 - L4/(4*np.sqrt(3))))
    print(Mn1,Mn2,Mch1,Mch2)
    pars_list = [[363.31704105, 228.60945556,  74.39969885, 137.42219051]]
    m = A4_spectrum(Mn1 = pars_list[0][0], Mn2 = pars_list[0][1], Mch1 = pars_list[0][2], Mch2 = pars_list[0][3], verbose=2, x_eps=1e-3, T_eps=1e-3)
    llim = m.mgr.Vtot(X=[vh/np.sqrt(3),vh/np.sqrt(3),0,vh/np.sqrt(3),0],T=0)-1000
    m.genspec()
    print(m.getinfo())
    return


    r1 = np.linspace(-150.,150.,5)
    r2 = np.linspace(-150.,150.,5)
    i2 = np.linspace(-10.,10.,2)
    r3 = np.linspace(-150.,150.,5)
    i3 = np.linspace(-10.,10.,2)
    x = np.array(np.meshgrid(r1, r2, i2, r3, i3)).T.reshape(-1, 5)

    print(m.mgr.tree_lvl_conditions())

    from scipy import linalg, interpolate, optimize

    def redrem(list, tol):
        indtorem = []
        for i in range(len(list)-1):
            if(indtorem.count(i)==0):
                for j in range(i+1,len(list)):
                    if np.linalg.norm(list[i]-list[j])<tol:
                        if indtorem.count(j)==0:
                            indtorem.append(j)
        indtorem.sort(reverse=True)
        for it in indtorem:
            list.pop(it)

        return list

    def inlistQ(list, x, tol):
        if len(list)>0:
            for el in list:
                if np.linalg.norm(el-x)<tol:
                    return True
        return False

    """
    for T in np.linspace(0.,80.,3):
        print('----T = {0} -------'.format(T))
        xt = []
        #x=[[0.,0.,0.,0.,0.]]
        for xi in x:
            #xmin = optimize.fmin(m.mgr.Vtot, xi, args=(T,),maxiter=10000, maxfun=20000, xtol=1e-5, ftol=1.e-10, disp=True)
            #xmin = optimize.minimize(m.mgr.Vtot, xi, args=(T,), method='Newton-CG', jac=m.mgr.gradV, hess=m.mgr.d2V, tol=1e-5)
            xmin = optimize.minimize(m.mgr.Vtot, xi, args=(T,), method='L-BFGS-B', jac=m.mgr.gradV, options={'maxiter':20000, 'maxcor':20, 'gtol':1e-9})
            if not inlistQ(xt, xmin['x'], 50) and (np.linalg.eigvals(m.mgr.d2V(xmin['x'],T)) > -500.).all():# and m.mgr.Vtot(xmin['x'], 0)>llim :
                if T<1.:
                    if m.mgr.Vtot(xmin['x'], 0)>llim:
                        xt.append(xmin['x'])
                else:
                    xt.append(xmin['x'])
                #print(xmin)
                #print('X: {0}   Grad: {1}   Hess: {2}'.format(xt[-1],m.mgr.gradV(xt[-1],T),np.linalg.eigvals(m.mgr.d2V(xt[-1],T))))
        xt = redrem(xt, 50)
        np.set_printoptions(precision=4, linewidth=100)
        print('# of different minima found: ', len(xt))
        for el in xt:
            print(np.asanyarray(el), m.mgr.Vtot(el, T), m.mgr.gradV(el,T),np.linalg.eigvals(m.mgr.d2V(el,T)))
    """


    print(m.mgr.gradV(X=[0.,0.,0.,0.,0.],T=m.mgr.findT0()))
    print(np.linalg.eigvals(m.mgr.d2V(X=[0.,0.,0.,0.,0.],T=m.mgr.findT0())))
    print(m.mgr.gradV(X=[1.42155181e+02,1.42155152e+02,2.09115174e-05,1.42155213e+02,1.94920174e-05],T=0))
    print(m.mgr.gradV(X=[1.42151703e+02,1.42151530e+02,9.14629714e-05,1.42151821e+02,1.03796244e-04],T=5.822196617527053))
    
    single_trace_args_={'dtabsMax':20.0, 'dtfracMax':.25, 'dtmin':1e-3, 'deltaX_tol':1.0, 'minratio':1e-4}
    m.mgr.getPhases(tracingArgs={'single_trace_args': single_trace_args_})

    print('V at (0,0,0) = ', m.mgr.Vtot(X=[0,0,0,0,0], T=96.60142436746617))
    print('V at (v,v,v) = ', m.mgr.Vtot(X=[vh/np.sqrt(3),vh/np.sqrt(3),0,vh/np.sqrt(3),0], T=96.60142436746617))
    print('V at (0,0,0) = ', m.mgr.Vtot(X=[0,0,0,0,0], T=91))
    print('V at (v,v,v) = ', m.mgr.Vtot(X=[vh/np.sqrt(3),vh/np.sqrt(3),0,vh/np.sqrt(3),0], T=91))
    #print('V at (v,-v,-v) = ', m.mgr.Vtot(X=[vh/np.sqrt(3),-vh/np.sqrt(3),0,-vh/np.sqrt(3),0], T=96.60142436746617))
    #print('V at (v,v,-v) = ', m.mgr.Vtot(X=[vh/np.sqrt(3),vh/np.sqrt(3),0,-vh/np.sqrt(3),0], T=96.60142436746617))
    #print('V at (v,-v,v) = ', m.mgr.Vtot(X=[vh/np.sqrt(3),-vh/np.sqrt(3),0,vh/np.sqrt(3),0], T=96.60142436746617))
    lim = 300.
    m.plot1d([0,0,0,0,0],[lim/np.sqrt(3),lim/np.sqrt(3),0,lim/np.sqrt(3),0],T=[0,40.0346,71.9,96.6,100])
    #m.plot1d([lim/np.sqrt(3),-lim/np.sqrt(3),0,-lim/np.sqrt(3),0],[lim/np.sqrt(3),lim/np.sqrt(3),0,lim/np.sqrt(3),0],T=[0,71.9,96.6,100])
    #m.plot1d([lim/np.sqrt(3),-lim/np.sqrt(3),0,lim/np.sqrt(3),0],[lim/np.sqrt(3),lim/np.sqrt(3),0,lim/np.sqrt(3),0],T=[0,71.9,96.6,100])
    #m.plot1d([lim/np.sqrt(3),-lim/np.sqrt(3),0,lim/np.sqrt(3),0],[lim/np.sqrt(3),-lim/np.sqrt(3),0,-lim/np.sqrt(3),0],T=[0,71.9,96.6,100])
    plt.show()
    m.genspec()
    print(m.getinfo())

    file_name = 'logtest.csv'
    queue = mp.Queue()
    lock = mp.Lock()

    for pars in pars_list:
        queue.put(pars)
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['Mn1', 'Mn2', 'Mch1', 'Mch2', 'M0', 'L0', 'L1', 'L2', 'L3', 'L4', 'dM0', 'dL0', 'dL1', 'dL2', 'dL3', 'dL4', 'NPhases', 'NTrans', 'VEVdif/T', 'alpha', 'beta', 'JougetVel', 'WallVel', 'RadCrit', 'KCol', 'KSW', 'KTurb', 'TempCrit', 'TempNuc', 'FreqPeakCol', 'FreqPeakSW', 'FreqPeakTurb', 'AmpPeakCol', 'AmpPeakSW', 'AmpPeakTurb', 'Action', 'Action/Tnuc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    p = MyProcessManager(findTrans, queue, 1, (queue,lock,file_name), 1200, 10)
    p.run()

if __name__ == "__main__":
  main()