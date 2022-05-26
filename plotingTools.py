from A4_reduced import model1
from gw_spectrum import gw_spectrum

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
    m = model1(Mn1=400.,Mn2=302.5,Mch1=1.,Mch2=134.)
    m.findAllTransitions()

    if(len(m.TnTrans)>0):
        trans = m.TnTrans[0]
        Tnuc = trans['Tnuc']
        Tcrit = m.TcTrans[0]['Tcrit']
        dT = Tcrit - Tnuc
        mfh = 0.1
        mfl = 0.1
        T_vec, S_vec = plotActionT(m, trans, Tnuc-dT*mfl, Tnuc+dT*mfh, n=100)

        fit1 = np.polyfit(T_vec, S_vec, deg=1)
        fit2 = np.polyfit(T_vec, S_vec, deg=2)
        fit3 = np.polyfit(T_vec, S_vec, deg=3)
        fit4 = np.polyfit(T_vec, S_vec, deg=4)
        fit5 = np.polyfit(T_vec, S_vec, deg=5)

        gw = gw_spectrum(m, 0, turb_on=True)

        def divpoly(p, x):
            div = 0.
            for i in range(0, p.size):
                div += (p.size-1-i)*p[i]*x**(p.size-i-2)
            return div
        print(gw.info)
        print('fit1', fit1)
        print('fit1', fit1)
        print('fit2', fit2)
        print('fit3', fit3)
        print('fit4', fit4)
        print('fit5', fit5)
        print('Beta from poly order 1: ', divpoly(fit1,Tnuc)*Tnuc)
        print('Beta from poly order 2: ', divpoly(fit2,Tnuc)*Tnuc)
        print('Beta from poly order 3: ', divpoly(fit3,Tnuc)*Tnuc)
        print('Beta from poly order 4: ', divpoly(fit4,Tnuc)*Tnuc)
        print('Beta from poly order 5: ', divpoly(fit5,Tnuc)*Tnuc)

        plt.plot(T_vec, S_vec, label = 'CosmoTransitions')
        plt.plot(T_vec, np.polyval(fit1,T_vec), label = 'Fit poly order: 1')
        plt.plot(T_vec, np.polyval(fit2,T_vec), label = 'Fit poly order: 2')
        plt.plot(T_vec, np.polyval(fit3,T_vec), label = 'Fit poly order: 3')
        plt.plot(T_vec, np.polyval(fit4,T_vec), label = 'Fit poly order: 4')
        plt.plot(T_vec, np.polyval(fit5,T_vec), label = 'Fit poly order: 5')
        plt.show()
    else:
        print('No Transitions')
    
if __name__ == "__main__":
  main()