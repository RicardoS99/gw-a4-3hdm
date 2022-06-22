from A4_model import A4_vev1
from gw_spectrum import gw_spectrum

from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def plot2d(m, box, T=[0], treelevel=False, offset=0, xaxis=[0], yaxis=[1], n=50, clevs=200, cfrac=.8, filled=False, **contourParams):
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
            axs[i,j].contourf(X,Y,Z, N, **contourParams) if filled else axs[i,j].contour(X,Y,Z, N, **contourParams)
            axs[i,j].set_title("T = " + str(T[k]) + " GeV")
            k+=1
      else:
        for j in range(ncols-rest):
            Z = m.V0(XY) if treelevel else m.Vtot(XY,T[k])
            minZ, maxZ = min(Z.ravel()), max(Z.ravel())
            N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
            axs[i,j].contourf(X,Y,Z, N, **contourParams) if filled else axs[i,j].contour(X,Y,Z, N, **contourParams)
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
        X[i,0] = vabs
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
    #Reading CSV Files to Data Frames
    df02 = pd.read_csv('output/run02.csv')
    df02a = pd.read_csv('output/run02a.csv')
    df02b = pd.read_csv('output/run02b.csv')
    df02c = pd.read_csv('output/run02c.csv')

    df = pd.concat([df02, df02a, df02b, df02c], ignore_index=True) #Concatenates all Data Frames
    df.drop(df.columns[[0]], inplace=True, axis=1) #Deleting first column which is meaningless info
    df.drop_duplicates(subset=['Mn1','Mn2','Mch1','Mch2'],inplace=True) #Deleting duplicated entries

    print(df.corr())

    df.plot(x='FreqPeakSW',y='AmpPeakSW', kind='scatter', xlabel='$f$ [Hz]', ylabel=r'$h^2 \Omega_\mathrm{GW}$')
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    df.plot(x='Mn1',y='AmpPeakSW', kind='scatter', xlabel=r'$m_\mathrm{H1}$ [GeV]', ylabel=r'$h^2 \Omega_\mathrm{GW}$')
    plt.grid()
    plt.yscale("log")
    plt.show()

    print(df['Mn1'][0])


    m = A4_vev1(Mn1=df['Mn1'][0],Mn2=df['Mn2'][0],Mch1=df['Mch1'][0],Mch2=df['Mch2'][0])
    #m.findAllTransitions()
    #m.prettyPrintTnTrans()
    #gw = gw_spectrum(m, 0, turb_on=True)
    #print(gw.info)
    T = np.linspace(0.,140.,15)
    #print(T)
    plot2d(m,(-300,300,-300,300),T=T,n=200, xaxis=[1], yaxis=[2], clevs=100,cfrac=0.01, filled=True)
    #plot2d(m,(-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1,-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1),T=T,n=200, xaxis=[0,1,3], yaxis=[2,4])
    #plot2d(m,(-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1,-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1),T=T,n=200, xaxis=[0,1,3], yaxis=[2,-4])
    #plot1d(m,m.TnTrans[0]['high_vev'],m.TnTrans[0]['low_vev'],T=T)

    plot1d(m,[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[df['TempNuc'][0],df['TempCrit'][0]])
    #plot1d(m,[0,0,0,0,0],[0,0,246.22/np.sqrt(3),0,246.22/np.sqrt(3)],T=T)
    #plot1d(m,[0,0,0,0,0],[0,0,246.22/np.sqrt(3),0,-246.22/np.sqrt(3)],T=T)
    #plot1d(m,[0,0,246.22/np.sqrt(3),0,246.22/np.sqrt(3)],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,0,246.22/np.sqrt(3),0,-246.22/np.sqrt(3)],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,246.22/np.sqrt(3),0,246./np.sqrt(3),0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    
    #plot1dtht(m,0,np.pi/2,140.755284,caxs=[1,3],saxs=[2,4],T=T)
    #plot1dtht(m,0,np.pi/2,20.,caxs=[0,1,3],saxs=[2,4],T=T)

    #plot1d(A4_vev1(Mn1=413.,Mn2=312.,Mch1=111,Mch2=110.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[83.315])
    #plot1d(A4_vev1(Mn1=413.,Mn2=312.,Mch1=111,Mch2=10.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[84.401])
    #plot1d(A4_vev1(Mn1=413.,Mn2=212.,Mch1=211,Mch2=110.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[88.881])
    #plot1d(A4_vev1(Mn1=313.,Mn2=112.,Mch1=311,Mch2=310.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[92.736])
    #plot1d(A4_vev1(Mn1=413.,Mn2=12.,Mch1=311,Mch2=110.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[51.16])
    #plt.show()
    
if __name__ == "__main__":
  main()