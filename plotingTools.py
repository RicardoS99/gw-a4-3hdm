from A4_reduced import model1

from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd

import numpy as np

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

def plotActionT(m, trans, Tmin=0.001, Tmax=500., n=50):
    def compute_action_my(m,x1,x0,T):

        res=None

        def V(x):   return(m.Vtot(x,T))
        def dV(x):  return(m.gradV(x,T))

        res = pd.fullTunneling(np.array([x1,x0]), V, dV).action

        if(T!=0):
            res=res/T
        else:
            res=res/(T+0.001)

        return res

    x1 = trans['low_vev']
    x0 = trans['high_vev']
    

    T_vec = np.linspace(Tmin, Tmax, n)
    S_vec = np.empty_like(T_vec)

    for i in range(0, len(S_vec)):
        S_vec[i] = compute_action_my(m,x1,x0,T_vec[i])

    return T_vec, S_vec

def main():
    m = model1(Mn1=400.,Mn2=300.,Mch1=220.,Mch2=180.)
    m.findAllTransitions()

    if(len(m.TnTrans)>0):
        trans = m.TnTrans[0]
        Tnuc = trans['Tnuc']
        Tcrit = m.TcTrans[0]['Tcrit']
        dT = Tcrit - Tnuc
        mfh = 0.1
        mfl = 0.1
        T_vec, S_vec = plotActionT(m, trans, Tnuc-dT*mfl, Tnuc+dT*mfh)

        plt.plot(T_vec, S_vec)
        plt.show()
    else:
        print('No Transitions')
    
if __name__ == "__main__":
  main()