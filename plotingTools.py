from A4_model import A4_vev1
from A4_model_gauge import A4_gauge_vev1
from A4_full import A4full_vev1
from gw_spectrum import gw_spectrum

from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd

import numpy as np

import pandas as pd
from matplotlib import colors
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
    fig, axs = plt.subplots(nlines, ncols, squeeze=False, sharex='all', sharey='all')
    k=0

    for i in range(nlines):
      if i < nlines-1:
        for j in range(ncols):
            Z = m.V0(XY) if treelevel else m.Vtot(XY,T[k])
            minZ, maxZ = min(Z.ravel()), max(Z.ravel())
            N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
            axs[i,j].contourf(X,Y,Z, N, **contourParams) if filled else axs[i,j].contour(X,Y,Z, N, **contourParams)
            axs[i,j].set_title('T = {:.1f} [GeV]'.format(T[k]))
            axs[i,j].set_aspect(1)
            k+=1
      else:
        for j in range(ncols-rest):
            Z = m.V0(XY) if treelevel else m.Vtot(XY,T[k])
            minZ, maxZ = min(Z.ravel()), max(Z.ravel())
            N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
            axs[i,j].contourf(X,Y,Z, N, **contourParams) if filled else axs[i,j].contour(X,Y,Z, N, **contourParams)
            axs[i,j].set_title('T = {:.1f} [GeV]'.format(T[k]))
            axs[i,j].set_aspect(1)
            k+=1

def plot1d(m, x1, x2, T=[0], treelevel=False, subtract=True, n=500, **plotParams):
    plt.figure()
    if m.Ndim == 1:
        x = np.linspace(x1,x2,n)
        X = x[:,np.newaxis]
    else:
        dX = np.array(x2)-np.array(x1)
        X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
        x = np.linspace(0,1,n)*np.sum(dX**2)**.5

    for t in T:
        if treelevel:
            y = m.V0(X) - m.V0(X*0) if subtract else m.V0(X)
        else:
            y = m.DVtot(X,t) if subtract else m.Vtot(X, t)
        plt.plot(x,y, **plotParams, label = 'T = {:.1f} [GeV]'.format(t))
    plt.xlabel(R"$|\phi|$")
    plt.ylabel(R"$V(\phi)$")
    plt.legend()
    plt.axhline(y=0, color="grey", linestyle="--")

def plot1dtht(m, tmin, tmax, vabs, caxs=[0], saxs=[1], T=[0], treelevel=False, subtract=True, n=500, **plotParams):
    plt.figure()
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
        plt.plot(tht,y, **plotParams, label = 'T = {:.1f} [GeV]'.format(t))
    plt.xlabel(R"$\theta$")
    plt.ylabel(R"$V(\phi)$")
    plt.legend()
    plt.axhline(y=0, color="grey", linestyle="--")
    

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
    """
    #Reading CSV Files to Data Frames
    df02 = pd.read_csv('output/run02.csv')
    df02a = pd.read_csv('output/run02a.csv')
    df02b = pd.read_csv('output/run02b.csv')
    df02c = pd.read_csv('output/run02c.csv')
    df02d = pd.read_csv('output/run02d.csv')
    df02d = pd.read_csv('output/run02d.csv')
    df03b = pd.read_csv('output/run03b.csv')

    df = pd.concat([df02, df02a, df02b, df02c, df02d, df03b], ignore_index=True) #Concatenates all Data Frames
    df.drop(df.columns[[0]], inplace=True, axis=1) #Deleting first column which is meaningless info
    df.drop_duplicates(subset=['Mn1','Mn2','Mch1','Mch2'],inplace=True) #Deleting duplicated entries
    df.sort_values(by='AmpPeakSW')

    df['MnSum'] = df['Mn1'] + df['Mn2']
    df['MnDif'] = df['Mn1'] - df['Mn2']
    df['MchSum'] = df['Mch1'] + df['Mch2']
    df['MchDif'] = df['Mch1'] - df['Mch2']
    df['MSum'] = df['MnSum'] + df['MchSum']

    print(df.corr())

    plt.ioff()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='MnSum', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$m_{\mathrm{H}_1} + m_{\mathrm{H}_2}$ [GeV]')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='MnDif', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$m_{\mathrm{H}_1} - m_{\mathrm{H}_2}$ [GeV]')
    plt.savefig('plots/peakmndif.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='MchSum', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$m_{\mathrm{H}^+} + m_{\mathrm{H}^-}$ [GeV]')
    plt.savefig('plots/peakmchsum.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='MchDif', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$m_{\mathrm{H}^+} - m_{\mathrm{H}^-}$ [GeV]')
    plt.savefig('plots/peakmchdif.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='MSum', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$m_{\mathrm{H}_1} + m_{\mathrm{H}_2} + m_{\mathrm{H}^+} + m_{\mathrm{H}^-}$ [GeV]')
    plt.savefig('plots/peakmsum.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='alpha', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\alpha$')
    plt.savefig('plots/peakalpha.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='VEVdif/T', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakvevdif.eps')
    #plt.show()

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='beta', colormap='viridis', s=2, loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\beta/H$')
    plt.savefig('plots/peakbeta.eps')
    #plt.show()

    m = A4_vev1(Mn1=df['Mn1'][0],Mn2=df['Mn2'][0],Mch1=df['Mch1'][0],Mch2=df['Mch2'][0])
    #m.findAllTransitions()
    #m.prettyPrintTnTrans()
    #gw = gw_spectrum(m, 0, turb_on=True)
    
    #print(gw.info)
    T = np.linspace(0.,140.,15)
    #print(T)
    plot2d(m,(-150,150,-150,150),T=[0,df['TempNuc'][0],df['TempCrit'][0],100],n=200, xaxis=[0,1], yaxis=[3], clevs=100,cfrac=0.8, filled=False)
    plt.savefig('plots/vplot2dx01y3.eps')
    plot2d(m,(-150,150,-150,150),T=[0,df['TempNuc'][0],df['TempCrit'][0],100],n=200, xaxis=[0,1,3], yaxis=[2,4], clevs=100,cfrac=0.8, filled=False)
    plt.savefig('plots/vplot2dx013y24.eps')
    plot2d(m,(-150,150,-150,150),T=[0,df['TempNuc'][0],df['TempCrit'][0],100],n=200, xaxis=[1], yaxis=[2], clevs=100,cfrac=0.8, filled=False)
    plt.savefig('plots/vplot2dx1y2.eps')
    #plot2d(m,(-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1,-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1),T=T,n=200, xaxis=[0,1,3], yaxis=[2,4])
    #plot2d(m,(-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1,-246.22/np.sqrt(3)*1.1,246.22/np.sqrt(3)*1.1),T=T,n=200, xaxis=[0,1,3], yaxis=[2,-4])
    #plot1d(m,m.TnTrans[0]['high_vev'],m.TnTrans[0]['low_vev'],T=T)

    #plot1d(m,[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    plot1d(m,[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[0,df['TempNuc'][0],df['TempCrit'][0],100])
    plt.savefig('plots/vplot1dradreal.eps')
    #plot1d(m,[0,0,0,0,0],[0,0,246.22/np.sqrt(3),0,246.22/np.sqrt(3)],T=T)
    #plot1d(m,[0,0,0,0,0],[0,0,246.22/np.sqrt(3),0,-246.22/np.sqrt(3)],T=T)
    #plot1d(m,[0,0,246.22/np.sqrt(3),0,246.22/np.sqrt(3)],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,0,246.22/np.sqrt(3),0,-246.22/np.sqrt(3)],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    #plot1d(m,[0,246.22/np.sqrt(3),0,246./np.sqrt(3),0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=T)
    
    plot1dtht(m,0,2*np.pi,139.007485,caxs=[1,3],saxs=[2,4],T=[0,df['TempNuc'][0],df['TempCrit'][0],100])
    plt.savefig('plots/vplotarg.eps')
    #plot1dtht(m,0,np.pi/2,20.,caxs=[0,1,3],saxs=[2,4],T=T)

    #plot1d(A4_vev1(Mn1=413.,Mn2=312.,Mch1=111,Mch2=110.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[83.315])
    #plot1d(A4_vev1(Mn1=413.,Mn2=312.,Mch1=111,Mch2=10.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[84.401])
    #plot1d(A4_vev1(Mn1=413.,Mn2=212.,Mch1=211,Mch2=110.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[88.881])
    #plot1d(A4_vev1(Mn1=313.,Mn2=112.,Mch1=311,Mch2=310.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[92.736])
    #plot1d(A4_vev1(Mn1=413.,Mn2=12.,Mch1=311,Mch2=110.),[0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[51.16])

    plt.show()
    """

    m = A4_vev1(Mn1=280.,Mn2=200.,Mch1=190.,Mch2=150., verbose = 1)
    mg = A4_gauge_vev1(Mn1=300.,Mn2=200.,Mch1=190.,Mch2=160., verbose = 1)

    print(np.linalg.eigvalsh(mg.d2V(X=[mg.vh/np.sqrt(3), 0., mg.vh/np.sqrt(3), 0., mg.vh/np.sqrt(3), 0.], T=0.)))
    mg.getPhases()

    T=np.linspace(0.,400.,41)
    #for t in T:
    #    print("T= {1:3.0f}, X = {0}".format(mg.gradV(X=[0.,0.,0.,0.,0.,0.],T=t),t))


    mg.findAllTransitions()
    mg.prettyPrintTnTrans()

    mg.plotPhasesPhi()
    plt.show()
    
    print(mg.phases)
    
    


    return
    #mtl = A4_vev1(Mn1=f*410.,Mn2=f*390.,Mch1=f*160.,Mch2=f*140., counterterms = False)
    #plot1d(mtl,[0,0,0,0,0],[300./np.sqrt(3),300./np.sqrt(3),0,300./np.sqrt(3),0],T=[0], treelevel=True)
    #plot1d(mtl,[0,0,0,0,0,0,0,0,0,0,0,0],[0.,300.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],T=np.linspace(0.,120.,4), treelevel=True)
    #plot1d(mtl,[0,0,0,0,0,0],[0.,300.,0.,0.,0.,0.],T=np.linspace(0.,120.,4), treelevel=True)
    #plt.axvline(x=246.22, color="grey", linestyle="--")
    #plot1d(m,[0,0,0,0,0],[300./np.sqrt(3),300./np.sqrt(3),0,300./np.sqrt(3),0],T=np.linspace(0.,120.,4))
    #plot1d(m,[0,0,0,0,0,0,0,0,0,0,0,0],[0.,300.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],T=np.linspace(0.,120.,4))
    #plot1d(m,[0,0,0,0,0,0,0,0,0,0,0,0],[300./np.sqrt(3),0,300./np.sqrt(3),0,300./np.sqrt(3),0,0,0,0,0,0,0],T=[0])

    
    sqcm = np.sqrt(1-m.cosa)
    sqcp = np.sqrt(1+m.cosa)
    min0 = np.array([0.,0. ,0., 0., 0., 0.])
    min1 = np.array([0.,m.vh ,0., 0., 0., 0.])
    min2 = np.array([0.,m.vh/np.sqrt(3), m.vh/2. * sqcm, -m.vh/np.sqrt(3)/2. * sqcm, -m.vh/2. * sqcp, -m.vh/np.sqrt(3)/2. * sqcp])
    min3 =np.array([0.,m.vh *2/3., (m.vh/np.sqrt(3)* sqcm - m.vh*np.sqrt(3)* sqcp)/4., -m.vh/12.* sqcm + m.vh/4.* sqcp,  (-m.vh/np.sqrt(3)* sqcp - m.vh*np.sqrt(3)* sqcm)/4.,-m.vh/4.* sqcm - m.vh/12.* sqcp])
    
    v0 = m.V0(min0)
    bm = m.boson_massSq(min0, 0.)
    fm = m.fermion_massSq(min0)
    v1 = m.V1(bm, fm)
    v1t = m.V1T(bm, fm, np.array([0.]))

    print('V0 = {0}, V1 = {1}, V1T = {2}'.format(v0,v1,v1t))
    print('boson = {0}\nfermion = {1}'.format(bm,fm))

    v0 = m.V0(min1)
    bm = m.boson_massSq(min1, 0.)
    fm = m.fermion_massSq(min1)
    v1 = m.V1(bm, fm)
    v1t = m.V1T(bm, fm, np.array([0.]))

    print('V0 = {0}, V1 = {1}, V1T = {2}'.format(v0,v1,v1t))
    print('boson = {0}\nfermion = {1}'.format(bm,fm))

    dmin0 =  m.gradV(X=min0, T=0.)
    dmin1 =  m.gradV(X=min1, T=0.)
    dmin2 =  m.gradV(X=min2, T=0.)
    dmin3 =  m.gradV(X=min3, T=0.)

    d2min0 =  np.linalg.eigvalsh(m.d2V(X=min0, T=0.))
    d2min1 =  np.linalg.eigvalsh(m.d2V(X=min1, T=0.))
    d2min2 =  np.linalg.eigvalsh(m.d2V(X=min2, T=0.))
    d2min3 =  np.linalg.eigvalsh(m.d2V(X=min3, T=0.))

    print("min 0: V = {0:13.1f}, dV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}), d2V = ({7:7.2f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=min0, T=0.), dmin0[0], dmin0[1], dmin0[2], dmin0[3], dmin0[4], dmin0[5], d2min0[0], d2min0[1], d2min0[2], d2min0[3], d2min0[4], d2min0[5]))
    print("min 1: V = {0:13.1f}, dV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}), d2V = ({7:7.2f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=min1, T=0.), dmin1[0], dmin1[1], dmin1[2], dmin1[3], dmin1[4], dmin1[5], d2min1[0], d2min1[1], d2min1[2], d2min1[3], d2min1[4], d2min1[5]))
    print("min 2: V = {0:13.1f}, dV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}), d2V = ({7:7.2f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=min2, T=0.), dmin2[0], dmin2[1], dmin2[2], dmin2[3], dmin2[4], dmin2[5], d2min2[0], d2min2[1], d2min2[2], d2min2[3], d2min2[4], d2min2[5]))
    print("min 3: V = {0:13.1f}, dV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}), d2V = ({7:7.2f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=min3, T=0.), dmin3[0], dmin3[1], dmin3[2], dmin3[3], dmin3[4], dmin3[5], d2min3[0], d2min3[1], d2min3[2], d2min3[3], d2min3[4], d2min3[5]))
    #for t in np.linspace(80., 90., 21):
    #    dmin1 =  m.gradV(X=min0, T=t)
    #    d2min1 =  np.linalg.eigvalsh(m.d2V(X=min0, T=t))
    #    print(" T = {13:5.3f}: V = {0:13.1f}, dV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}), d2V = ({7:7.2f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=min0, T=t), dmin1[0], dmin1[1], dmin1[2], dmin1[3], dmin1[4], dmin1[5], d2min1[0], d2min1[1], d2min1[2], d2min1[3], d2min1[4], d2min1[5], t))

    eps=1
    xtemp = min1
    sign = 1.
    tol = 1e-3
    xlist = []
    xdif =(np.pi/2.)**6
    tlist = np.linspace(0., 150., 151)
    for t in tlist:

        dmin1 =  m.gradV(X=xtemp, T=t)
        eps=1e-5
        sign = 1.
        while np.linalg.norm(dmin1)>tol:
            if xlist:
                if np.linalg.norm(xtemp-xlist[-1])>xdif:
                    break
            dprev = dmin1
            xtemp = xtemp - dmin1*eps
            dmin1 =  m.gradV(X=xtemp, T=t)
            if(np.dot(dprev,dmin1)<0): 
                eps = eps/2
            #print("X={0}\ndV={1}\ndVprev={2},\ndV.dVprev={3:10.5f}".format(xtemp,dmin1,dprev,np.dot(dprev,dmin1)))
        d2min1 =  np.linalg.eigvals(m.d2V(X=xtemp, T=t))
        if xlist:
            if np.linalg.norm(xtemp-xlist[-1])<xdif:
                xlist.append(xtemp)
                #if len(xlist)==2:
                #    xdif = xdif*np.linalg.norm(xlist[-1]-xlist[-2])
                #    print('xdif: {0:10.5f}'.format(xdif))
                #if len(xlist)>3:
                #    xdif = xdif*np.linalg.norm(xlist[-1]-xlist[-2])/np.linalg.norm(xlist[-2]-xlist[-3])
                #    print('xdif: {0:10.5f}'.format(xdif))
                print(" T = {13:5.3f}:\nX= {14},\nV = {0:13.1f},\ndV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}),\nmd2V = ({7:8.5f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=xtemp, T=t), dmin1[0], dmin1[1], dmin1[2], dmin1[3], dmin1[4], dmin1[5], d2min1[0], d2min1[1], d2min1[2], d2min1[3], d2min1[4], d2min1[5], t,xtemp))
                    
        else:
            xlist.append(xtemp)
            print(" T = {13:5.3f}:\nX= {14},\nV = {0:13.1f},\ndV = ({1:7.2f}, {2:7.2f}, {3:7.2f}, {4:7.2f}, {5:7.2f}, {6:7.2f}),\nmd2V = ({7:8.5f}, {8:7.2f}, {9:7.2f}, {10:7.2f}, {11:7.2f}, {12:7.2f})".format(m.Vtot(X=xtemp, T=t), dmin1[0], dmin1[1], dmin1[2], dmin1[3], dmin1[4], dmin1[5], d2min1[0], d2min1[1], d2min1[2], d2min1[3], d2min1[4], d2min1[5], t,xtemp))
    
    xlist = np.asanyarray(xlist)

    print(xlist)
    plt.figure()
    plt.plot(tlist,xlist[...,0],label='0')
    plt.plot(tlist,xlist[...,1],label='1')
    plt.plot(tlist,xlist[...,2],label='2')
    plt.plot(tlist,xlist[...,3],label='3')
    plt.plot(tlist,xlist[...,4],label='4')
    plt.plot(tlist,xlist[...,5],label='5')
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(tlist,xlist[...,0])
    plt.show()


    """
    x = 1e-3
    plot1d(m,[-x,0.,0.,0.,0.,0.],[x,0.,0.,0.,0.,0.],T=[300.])
    plot1d(m,[0.,-x,0.,0.,0.,0.],[0.,x,0.,0.,0.,0.],T=[300.])
    plot1d(m,[0.,0.,-x,0.,0.,0.],[0.,0.,x,0.,0.,0.],T=[300.])
    plot1d(m,[0.,0.,0.,-x,0.,0.],[0.,0.,0.,x,0.,0.],T=[300.])
    plot1d(m,[0.,0.,0.,0.,-x,0.],[0.,0.,0.,0.,x,0.],T=[300.])
    plot1d(m,[0.,0.,0.,0.,0.,-x],[0.,0.,0.,0.,0.,x],T=[300.])
    """
    """
    plot1d(m,-1.5*min1,1.5*min1,T=np.linspace(0., 200., 21))
    plot1d(m,-1.5*min2,1.5*min2,T=np.linspace(0., 200., 21))
    plot1d(m,-1.5*min3,1.5*min3,T=np.linspace(0., 200., 21))
    """

    """
    x = 500.

    xl = np.array([-2.16396918e-01, 8.20749973e+01, 1.74618491e-01, -2.22495268e+02, 5.86625824e-01, -6.62291277e+01])
    xh = np.array([-0.28221083, 82.1015842, 0.22772661, -222.56734099, 0.76503974, -66.25058194])
    xm = (xh + xl)/2.
    xd = (xh - xl)/2.
    print("begin: ", xm-x*xd)
    print("end  : ", xm+x*xd)

    plot1d(m,xm-x*xd,xm+x*xd,T=np.linspace(25., 26., 11))
    """

    """
    plot1d(m,[0,0,0,0,0,0],[0.,300.,0.,0.,0.,0.],T=np.linspace(0.,110.,12))
    plt.axvline(x=246.22, color="grey", linestyle="--")

    plot1d(m,[-300.,246.22,0,0,0,0],[300.,246.22,0.,0.,0.,0.],T=np.linspace(0.,110.,12))
    plt.axvline(x=246.22, color="grey", linestyle="--")

    plot1d(m,[0,246.22,-300.,0,0,0],[0.,246.22,300.,0.,0.,0.],T=np.linspace(0.,110.,12))
    plt.axvline(x=246.22, color="grey", linestyle="--")

    plot1d(m,[0,246.22,0,-300.,0,0],[0.,246.22,0.,300.,0.,0.],T=np.linspace(0.,110.,12))
    plt.axvline(x=246.22, color="grey", linestyle="--")

    plot1d(m,[0,246.22,0,0,-300.,0],[0.,246.22,0.,0.,300.,0.],T=np.linspace(0.,110.,12))
    plt.axvline(x=246.22, color="grey", linestyle="--")

    plot1d(m,[0,246.22,0,0,0,-300],[0.,246.22,300.,0.,0.,300.],T=np.linspace(0.,110.,12))
    plt.axvline(x=246.22, color="grey", linestyle="--")

    plot1dtht(m,0,2*np.pi,246.22,caxs=[1],saxs=[0],T=np.linspace(0.,110.,12))
    plot1dtht(m,0,2*np.pi,246.22,caxs=[1],saxs=[2],T=np.linspace(0.,110.,12))
    plot1dtht(m,0,2*np.pi,246.22,caxs=[1],saxs=[3],T=np.linspace(0.,110.,12))
    plot1dtht(m,0,2*np.pi,246.22,caxs=[1],saxs=[4],T=np.linspace(0.,110.,12))
    plot1dtht(m,0,2*np.pi,246.22,caxs=[1],saxs=[5],T=np.linspace(0.,110.,12))
    """
    """
    plot2d(m,(-300,300,-300,300),T=np.linspace(0.,110.,12),n=100, xaxis=[1], yaxis=[0], clevs=150,cfrac=0.5, filled=False)
    plot2d(m,(-300,300,-300,300),T=np.linspace(0.,110.,12),n=100, xaxis=[1], yaxis=[2], clevs=150,cfrac=0.5, filled=False)
    plot2d(m,(-300,300,-300,300),T=np.linspace(0.,110.,12),n=100, xaxis=[1], yaxis=[4], clevs=150,cfrac=0.5, filled=False)
    plot2d(m,(-300,300,-300,300),T=np.linspace(0.,110.,12),n=100, xaxis=[2], yaxis=[3], clevs=150,cfrac=0.5, filled=False)
    plot2d(m,(-300,300,-300,300),T=np.linspace(0.,110.,12),n=100, xaxis=[4], yaxis=[5], clevs=150,cfrac=0.5, filled=False)
    plot2d(m,(-300,300,-300,300),T=np.linspace(0.,110.,12),n=100, xaxis=[2], yaxis=[4], clevs=150,cfrac=0.5, filled=False)
    """
    #plot1d(m,[0,0,0,0,0],[300./np.sqrt(3),0,0,0,0],T=np.linspace(0.,140.,8))
    #plt.axvline(x=246.22, color="grey", linestyle="--")
    #plot1d(m,[0,0,0,0,0],[300./np.sqrt(3),300./np.sqrt(3)*np.cos(np.pi/3),300./np.sqrt(3)*np.sin(np.pi/3),300./np.sqrt(3)*np.cos(-np.pi/3),300./np.sqrt(3)*np.sin(-np.pi/3)],T=np.linspace(0.,140.,8))
    #plt.axvline(x=246.22, color="grey", linestyle="--")

    #print(np.sqrt(np.linalg.eigvals(mtl.massSqMatrix(X=[246.22/np.sqrt(3),246.22/np.sqrt(3),0.,246.22/np.sqrt(3),0.]))))
    #print(np.sqrt(np.linalg.eigvals(m.d2V(X=[246.22/np.sqrt(3),246.22/np.sqrt(3),0.,246.22/np.sqrt(3),0.],T=0))))
    #M, g1, g2 = m.boson_massSq(X=[246.22/np.sqrt(3),246.22/np.sqrt(3),0.,246.22/np.sqrt(3),0.],T=0)
    #print(np.sqrt(M))
    #print(np.sqrt(np.linalg.eigvals(m.d2V(X=[246.22/np.sqrt(3),246.22/np.sqrt(3),0.,246.22/np.sqrt(3),0.],T=200))))
    #M, g1, g2 = m.boson_massSq(X=[246.22/np.sqrt(3),246.22/np.sqrt(3),0.,246.22/np.sqrt(3),0.],T=200)
    #print(np.sqrt(M))
    
    #msm = mtl.massSqMatrix(X=[0.,246.22,0.,0.,0.,0.])
    """
    msm2 = m.d2V(X=[0.,246.22,0.,0.,0.,0.],T=0)
    msm2[np.abs(msm2) < 1] = 0
    for i in range(6):
        print("{0:2d}th scalar mass: {1:8.2f}".format(i,np.sqrt(msm2[i][i])))

    msm, g1, g2 = m.boson_massSq(X=[0.,246.22,0.,0.,0.,0.],T=0)
    msm[np.abs(msm) < 1] = 0
    for i in range(len(msm)):
        print("{0:2d}th boson: {1:8.2f}".format(i,np.sqrt(msm[i])))

    msm, g1 = m.fermion_massSq(X=[0.,246.22,0.,0.,0.,0.])
    for i in range(len(msm)):
        print("{0:2d}th fermion: {1:11.6f}".format(i,np.sqrt(msm[i])))
    """
    

    plt.show()
    
if __name__ == "__main__":
  main()


  