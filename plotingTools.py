from cProfile import label
from A4_model import A4_vev1
from gw_spectrum import gw_spectrum

from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd

import numpy as np

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

def plot2d(m, box, T=[0], treelevel=False, offset=0, xaxis=[0], yaxis=[1], n=50, clevs=200, cfrac=.8, **contourParams):
    """
    Makes a countour plot of the potential.
    Parameters
    ----------
    box : tuple
        The bounding box for the plot, (xlow, xhigh, ylow, yhigh).
    T : float, optional
        The temperature
    offset : array_like
        A constant to add to all coordinates. Especially
        helpful if Ndim > 2.
    xaxis, yaxis : int, optional
        The integers of the axes that we want to plot.
    n : int
        Number of points evaluated in each direction.
    clevs : int
        Number of contour levels to draw.
    cfrac : float
        The lowest contour is always at ``min(V)``, while the highest is
        at ``min(V) + cfrac*(max(V)-min(V))``. If ``cfrac < 1``, only part
        of the plot will be covered. Useful when the minima are more
        important to resolve than the maximum.
    contourParams :
        Any extra parameters to be passed to :func:`plt.contour`.
    """
    xmin,xmax,ymin,ymax = box
    X = np.linspace(xmin, xmax, n).reshape(n,1)*np.ones((1,n))
    Y = np.linspace(ymin, ymax, n).reshape(1,n)*np.ones((n,1))
    XY = np.zeros((n,n,m.Ndim))
    for axis in xaxis:
        XY[...,np.abs(axis)] = np.sign(axis)*X
    for axis in yaxis:
        XY[...,np.abs(axis)] = np.sign(axis)*Y
    #XY[...,xaxis], XY[...,yaxis] = X,Y
    XY += offset

    nlines = int(np.floor(np.sqrt(len(T))))
    ncols = int(np.ceil(len(T)/nlines))
    #rest = int(np.mod(len(T),nlines))
    rest = int(nlines*ncols-len(T))
    fig, axs = plt.subplots(nlines, ncols, squeeze=False)
    k=0

    for i in range(nlines):
      if i < nlines-1:
        for j in range(ncols):
            Z = m.V0(XY) if treelevel else m.Vtot(XY,T[k])
            minZ, maxZ = min(Z.ravel()), max(Z.ravel())
            N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
            axs[i,j].contour(X,Y,Z, N, **contourParams)
            axs[i,j].set_title("T = " + str(T[k]) + " GeV")
            k+=1
      else:
        for j in range(ncols-rest):
            Z = m.V0(XY) if treelevel else m.Vtot(XY,T[k])
            minZ, maxZ = min(Z.ravel()), max(Z.ravel())
            N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
            axs[i,j].contour(X,Y,Z, N, **contourParams)
            axs[i,j].set_title("T = " + str(T[k]) + " GeV")
            k+=1

    plt.show()

def plot1d(m, x1, x2, T=[0], treelevel=False, subtract=True, n=500, **plotParams):
    if m.Ndim == 1:
        x = np.linspace(x1,x2,n)
        X = x[:,np.newaxis]
    else:
        dX = np.array(x2)-np.array(x1)
        X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
        x = np.linspace(0,1,n)*np.sum(dX**2)**.5

    print(X)


    for t in T:
        if treelevel:
            y = m.V0(X) - m.V0(X*0) if subtract else m.V0(X)
        else:
            y = m.DVtot(X,t) if subtract else m.Vtot(X, t)
        plt.plot(x,y, **plotParams, label = str(t))
    plt.xlabel(R"$\phi$")
    plt.ylabel(R"$V(\phi)$")
    plt.legend()
    plt.axhline(y=0)
    plt.show()

def plot1dtht(m, tmin, tmax, vabs, caxs=[0], saxs=[1], T=[0], treelevel=False, subtract=True, n=500, **plotParams):
    X = np.zeros((n,m.Ndim))
    tht = np.linspace(tmin,tmax,n)

    for i in range(n):
        for ax in caxs:
            X[i,np.abs(ax)] = np.sign(ax)*vabs*np.cos(tht[i])
        for ax in saxs:
            X[i,np.abs(ax)] = np.sign(ax)*vabs*np.sin(tht[i])


    for t in T:
        if treelevel:
            y = m.V0(X) - m.V0(X*0) if subtract else m.V0(X)
        else:
            y = m.DVtot(X,t) if subtract else m.Vtot(X, t)
        plt.plot(tht,y, **plotParams, label = str(t))
    plt.xlabel(R"$\phi$")
    plt.ylabel(R"$V(\phi)$")
    plt.legend()
    plt.axhline(y=0)
    plt.show()

    

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
    m = A4_vev1(Mn1=200.,Mn2=100.,Mch1=350.,Mch2=349.75)
    #m.findAllTransitions()
    #m.prettyPrintTnTrans()

    #plot2d(m,(-300,300,-300,300),T=[0,30,60,90,120,150],n=200, xaxis=[0,1,3], yaxis=[2,4], clevs=100,cfrac=0.01)
    T = np.linspace(0.,140.,15)
    print(T)
    #plot2d(m,(-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1,-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1),T=T,n=200, xaxis=[0,1,3], yaxis=[2,4])
    #plot2d(m,(-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1,-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1),T=T,n=200, xaxis=[0,1,3], yaxis=[2,-4])
    #plot1d(m,m.TnTrans[0]['high_vev'],m.TnTrans[0]['low_vev'],T=T)
    plot1d(m,[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,0,0,0,0],[0,0,246.22/np.sqrt(3),0,246.22/np.sqrt(3)],T=T)
    #plot1d(m,[0,0,0,0,0],[0,0,246.22/np.sqrt(3),0,-246.22/np.sqrt(3)],T=T)
    #plot1d(m,[0,0,246.22/np.sqrt(3),0,246.22/np.sqrt(3)],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,0,246.22/np.sqrt(3),0,-246.22/np.sqrt(3)],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,246.22/np.sqrt(3),0,246./np.sqrt(3),0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    plot1dtht(m,0,np.pi/2,136.015362,caxs=[0,1,3],saxs=[2,4],T=T)
    plot1dtht(m,0,np.pi/2,136.015362,caxs=[0,1,3],saxs=[2,-4],T=T)


    """
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

    """
    
if __name__ == "__main__":
  main()