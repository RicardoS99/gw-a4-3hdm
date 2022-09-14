from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd

import numpy as np

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

class gw_spectrum():
    def __init__(self, model, ind_trans, g=106.75, delta=0.0001, v=0.95, turb_on=False, epsilon=0.05):
        self.m = model
        self.id = ind_trans
        self.g = g
        self.delta = delta
        self.v = v
        self.Tcrit = self.m.TcTrans[self.id]['Tcrit']
        self.Tnuc = self.m.TnTrans[self.id]['Tnuc']
        self.Dvev = (np.linalg.norm(self.m.TcTrans[self.id]['low_vev']) - np.linalg.norm(self.m.TcTrans[self.id]['high_vev']))/self.Tcrit
        self.alpha = self.calc_alpha()
        try:
            self.beta = self.calc_beta()
        except:
            self.beta = np.nan
        self.vJ = (3.**(-1/2.)+(self.alpha**2+2*self.alpha/3.)**(1/2.))/(1.+self.alpha)
        self.r = (8.*np.pi)**(1/3.)*v
        self.kf = self.alpha/(0.73+0.083*self.alpha**(1/2.)+self.alpha)
        self.kcol = 1/(1.+0.715*self.alpha)*(0.715*self.alpha+4/27.*(3*self.alpha/2)**(1/2.))
        self.kturb = epsilon*self.kf
        if(self.beta!=np.nan):
            self.fpeak_col, self.peak_col = self.colPeak()
            self.fpeak_sw, self.peak_sw = self.swPeak()
            if turb_on:
                self.fpeak_turb, self.peak_turb = self.turbPeak() 
            else:
                self.fpeak_turb, self.peak_turb = 1, 0
        else:
            self.fpeak_col, self.peak_col, self.fpeak_sw, self.peak_sw, self.fpeak_turb, self.peak_turb = np.nan
        self.info ={
            "VEVdif/T": self.Dvev,
            "alpha": self.alpha,
            "beta": self.beta,
            "JougetVel": self.vJ,
            "WallVel": self.v,
            "RadCrit": self.r,
            "KCol": self.kcol,
            "KSW": self.kf,
            "KTurb": self.kturb,
            "TempCrit": self.Tcrit,
            "TempNuc": self.Tnuc,
            "FreqPeakCol": self.fpeak_col,
            "FreqPeakSW": self.fpeak_sw,
            "FreqPeakTurb": self.fpeak_turb,
            "AmpPeakCol": self.peak_col,
            "AmpPeakSW": self.peak_sw,
            "AmpPeakTurb": self.peak_turb,
            "Action": self.m.TnTrans[self.id]['action'],
            "Action/Tnuc": self.m.TnTrans[self.id]['action'] /self.Tnuc
        }

    def total(self,f): return self.col(f) + self.sw() + self.turb()
        
    def col(self,f): return self.peak_col * 3.8*(f/self.fpeak_col)**2.8 / (1.+2.8*((f/self.fpeak_col)**3.8))
    
    def sw(self,f): return self.peak_sw*(f/self.fpeak_sw)**3 *(7/(4.+3.*(f/self.fpeak_sw)**2))**(7/2.)

    def turb(self,f):
        kin = self.kturb*self.alpha/(1.+self.alpha)
        g=self.g/100.
        T=self.Tnuc/100.
        h=1.6e-7*T*g**(1/6.)
        return self.peak_turb*((2.)**(11/3.) * (1.+13.5*np.pi*self.beta/self.v))*(f/self.fpeak_turb)**3 /((1.+(f/self.fpeak_turb))**(11/3.) * (1.+8*np.pi*f/h))
    
    def colPeak(self):
        kin = self.kcol*self.alpha/(1.+self.alpha)
        g = self.g/100.
        fpeak = (1.65e-5)*self.beta*self.Tnuc*(g**(1/6.))*0.62/(1.8-0.1*self.v+self.v**2)
        peak = (1.67e-5)/self.beta**2 *(kin**2)/(g**(1/3.))*0.11*self.v**3 /(0.42+self.v**2)

        return fpeak, peak

    def swPeak(self):
        kin = self.kf*self.alpha/(1.+self.alpha)
        g = self.g/100.
        T = self.Tnuc/100.
        F = 3.57e-5/g**(1/3.)
        cs = 1/np.sqrt(3.)
        HR = (8.*np.pi)**(1/3.)*self.v/self.beta
        Ht = 2*HR/np.sqrt(3.*kin)
        fpeak = (26e-6)/HR*T*g**(1/6.)
        peak = 0.687e-2*F *(HR/np.sqrt(cs))**(2) * kin**(3/2.)
        #fpeak = 1.9e-5/self.v * self.beta * T * g**(1/6.)
        #peak = 2.65e-6 / self.beta * kin**2 * g**(1/3.)*self.v * (1-(1+2*Ht)**(-1/2))

        return fpeak, peak

    def turbPeak(self):
        kin = self.kturb*self.alpha/(1.+self.alpha)
        g=self.g/100.
        T=self.Tnuc/100.
        fpeak = (2.7e-5)*self.beta*(T)*(g**(1/6.))/self.v
        peak = (3.35e-4)/self.beta*(kin**2)/(g**(1/3.))*self.v /((2.)**(11/3.) * (1.+13.5*np.pi*self.beta/self.v))

        return fpeak, peak

    def calc_alpha(self):
        xf = self.m.TnTrans[self.id]['high_vev']
        xt = self.m.TnTrans[self.id]['low_vev']
        vf = self.m.Vtot(xf,self.Tnuc)
        vt = self.m.Vtot(xt,self.Tnuc)
        dvf = (self.m.Vtot(xf,self.Tnuc+0.5*self.delta) -self.m.Vtot(xf,self.Tnuc-0.5*self.delta))/self.delta
        dvt = (self.m.Vtot(xt,self.Tnuc+0.5*self.delta) -self.m.Vtot(xt,self.Tnuc-0.5*self.delta))/self.delta
        
        rho = (np.pi**2 *self.g*self.Tnuc**4) /30.

        return ((vf-self.Tnuc/4*dvf)-(vt-self.Tnuc/4*dvt))/rho

    def calc_beta(self):
        def divpoly(p, x):
            div = 0.
            for i in range(0, p.size):
                div += (p.size-1-i)*p[i]*x**(p.size-i-2)
            return div

        def compute_action_my(x1,x0,T):

            res=None

            def V(x):   return(self.m.Vtot(x,T))
            def dV(x):  return(self.m.gradV(x,T))

            res = pd.fullTunneling(np.array([x1,x0]), V, dV).action

            if(T!=0):
                res=res/T
            else:
                res=res/(T+0.001)

            return res

        #compute_action_vec = np.vectorize(compute_action_my, excluded=['x1','x0'])
        xf = self.m.TnTrans[self.id]['high_vev']
        xt = self.m.TnTrans[self.id]['low_vev']

        dT = np.minimum(self.Tcrit-self.Tnuc,10)
        nsteps = int(10*2*dT)+1 if dT>1.0 else 21

        #T_vec = np.linspace(self.Tnuc-dT, self.Tnuc+dT, nsteps)
        T_vec = np.linspace(self.Tnuc-dT, self.Tnuc+dT, 101)
        S_vec = np.zeros_like(T_vec)

        for i in range(0, len(S_vec)):
            try:
                print("Calculating Action for Temperature:", T_vec[i])
                S_vec[i] = compute_action_my(xt,xf,T_vec[i])
            except:
                print("Couldn't calculate action...")

        ind_to_rem = np.where(np.abs(S_vec)<1e-8)
        T_vec = np.delete(T_vec, ind_to_rem)
        S_vec = np.delete(S_vec, ind_to_rem)

        fit1 = np.polyfit(T_vec, S_vec, deg=9)
        fit2 = np.polyfit(T_vec, S_vec, deg=10)
        fit3 = np.polyfit(T_vec, S_vec, deg=11)

        output1 = divpoly(fit1,self.Tnuc)*self.Tnuc
        output2 = divpoly(fit2,self.Tnuc)*self.Tnuc
        output3 = divpoly(fit3,self.Tnuc)*self.Tnuc

        return (output1+output2+output3)/3
