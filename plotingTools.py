from A4_model import A4_vev1
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
    S_vec = np.zeros_like(T_vec)

    for i in range(0, len(S_vec)):
        try:
            print("Calculating Action for Temperature:", T_vec[i])
            S_vec[i] = compute_action_my(m,x1,x0,T_vec[i])
        except:
            print("Error calculating")

    ind_to_rem = np.where(np.abs(S_vec)<1e-8)
    T_vec = np.delete(T_vec, ind_to_rem)
    S_vec = np.delete(S_vec, ind_to_rem)

    return T_vec, S_vec

def main():  
    m = A4_vev1(Mn1=400.,Mn2=10.,Mch1=134.,Mch2=45.33)
    m.findAllTransitions()
    m.prettyPrintTnTrans()

    if(len(m.TnTrans)>0):
        trans = m.TnTrans[0]
        Tnuc = trans['Tnuc']
        Tcrit = m.TcTrans[0]['Tcrit']
        dT = (Tcrit - Tnuc)*1.0
        nsteps = int(10*2*dT) if dT>1.0 else 20

        print('Tnuc', Tnuc)
        print('Tcrit', Tcrit)

        T_vec = np.array([])
        S_vec = np.array([])
        searching = True
        while(searching):
            T_vec, S_vec = plotActionT(m, trans, Tnuc-dT, Tcrit, n=nsteps)
            searching=False


        fit1 = np.polyfit(T_vec, S_vec, deg=1)
        fit2 = np.polyfit(T_vec, S_vec, deg=2)
        fit3 = np.polyfit(T_vec, S_vec, deg=3)
        fit4 = np.polyfit(T_vec, S_vec, deg=4)
        fit5 = np.polyfit(T_vec, S_vec, deg=5)
        fit6 = np.polyfit(T_vec, S_vec, deg=6)
        fit7 = np.polyfit(T_vec, S_vec, deg=7)
        fit8 = np.polyfit(T_vec, S_vec, deg=8)
        fit9 = np.polyfit(T_vec, S_vec, deg=9)

        gw = gw_spectrum(m, 0, turb_on=True)

        def divpoly(p, x):
            div = 0.
            for i in range(0, p.size):
                div += (p.size-1-i)*p[i]*x**(p.size-i-2)
            return div
        print(gw.info)
        #print('fit1', fit1)
        #print('fit1', fit1)
        #print('fit2', fit2)
        #print('fit3', fit3)
        #print('fit4', fit4)
        #print('fit5', fit5)
        print('Beta from poly order 1: ', divpoly(fit1,Tnuc)*Tnuc)
        print('Beta from poly order 2: ', divpoly(fit2,Tnuc)*Tnuc)
        print('Beta from poly order 3: ', divpoly(fit3,Tnuc)*Tnuc)
        print('Beta from poly order 4: ', divpoly(fit4,Tnuc)*Tnuc)
        print('Beta from poly order 5: ', divpoly(fit5,Tnuc)*Tnuc)
        print('Beta from poly order 6: ', divpoly(fit6,Tnuc)*Tnuc)
        print('Beta from poly order 7: ', divpoly(fit7,Tnuc)*Tnuc)
        print('Beta from poly order 8: ', divpoly(fit8,Tnuc)*Tnuc)
        print('Beta from poly order 9: ', divpoly(fit9,Tnuc)*Tnuc)

        plt.plot(T_vec, S_vec, label = 'CosmoTransitions')
        #plt.plot(T_vec, np.polyval(fit1,T_vec), label = 'Fit poly order: 1')
        #plt.plot(T_vec, np.polyval(fit2,T_vec), label = 'Fit poly order: 2')
        #plt.plot(T_vec, np.polyval(fit3,T_vec), label = 'Fit poly order: 3')
        #plt.plot(T_vec, np.polyval(fit4,T_vec), label = 'Fit poly order: 4')
        plt.plot(T_vec, np.polyval(fit5,T_vec), label = 'Fit poly order: 5')
        plt.plot(T_vec, np.polyval(fit6,T_vec), label = 'Fit poly order: 6')
        plt.plot(T_vec, np.polyval(fit7,T_vec), label = 'Fit poly order: 7')
        plt.plot(T_vec, np.polyval(fit8,T_vec), label = 'Fit poly order: 8')
        plt.plot(T_vec, np.polyval(fit9,T_vec), label = 'Fit poly order: 9')
        plt.show()
    else:
        print('No Transitions')
    
if __name__ == "__main__":
  main()