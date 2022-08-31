from A4_model import A4_vev1
from A4_model_gauge import A4_gauge_vev1
from A4_full import A4full_vev1
from A4_model_reduced import A4_reduced_vev1
from gw_spectrum import gw_spectrum
from A4_spectrum import A4_spectrum

from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder as tf
from cosmoTransitions import pathDeformation as pd

import numpy as np
from scipy import stats
from scipy import optimize

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
    
    #Reading CSV Files to Data Frames
    df01c = pd.read_csv('output/test1c.csv')
    df01d = pd.read_csv('output/test1d.csv')
    df02 = pd.read_csv('output/test2.csv')
    df03 = pd.read_csv('output/test3.csv')
    df04 = pd.read_csv('output/test4.csv')
    df05 = pd.read_csv('output/test5.csv')
    df06 = pd.read_csv('output/test6.csv')
    df06a = pd.read_csv('output/test6a.csv')
    dfL = pd.read_csv('output/Ltest01.csv')
    dfL2 = pd.read_csv('output/Ltest02.csv')
    dfL3 = pd.read_csv('output/Ltest03.csv')
    
    df = pd.concat([df01c, df01d, df02, df03, df04, df05, df06, df06a, dfL, dfL2, dfL3], ignore_index=True) #Concatenates all Data Frames
    #df = pd.concat([dfL, dfL2, dfL3], ignore_index=True) #Concatenates all Data Frames
    #df.drop(df.columns[[0]], inplace=True, axis=1) #Deleting first column which is meaningless info
    df.drop_duplicates(subset=['Mn1','Mn2','Mch1','Mch2'],inplace=True) #Deleting duplicated entries
    df.dropna(subset=['AmpPeakSW'],inplace=True)

    df['MnSum'] = df['Mn1'] + df['Mn2']
    df['MnDif'] = df['Mn1'] - df['Mn2']
    df['MchSum'] = df['Mch1'] + df['Mch2']
    df['MchDif'] = df['Mch1'] - df['Mch2']
    df['MSum'] = df['MnSum'] + df['MchSum']
    df['b/f'] = df['beta'].div(df['FreqPeakSW'])
    df['fp'] = np.log(df['AmpPeakSW'])*np.log(df['FreqPeakSW'])
    df['logf'] = np.log(df['FreqPeakSW'])
    df['loga'] = np.log(df['AmpPeakSW'])

    #df = df[df['AmpPeakSW']>1E-13]
    df = df[df['M0']>0.0]
    df = df[df['L0']+df['L1']>0.0]
    df = df[df['L4']**2 < 12.0*df['L1']**2]
    df = df[df['L4']**2 < 2.0*(df['L3']-df['L1'])*(df['L2']-df['L1'])]

    df.sort_values(by='FreqPeakSW')
    print('Number of total points: ', df['AmpPeakSW'].count())
    freq = np.log(df['FreqPeakSW'].to_numpy())
    amp =np.log(df['AmpPeakSW'].to_numpy())

    fit = np.poly1d(np.polyfit(freq,amp, deg=12))
    fax = np.sort(freq)

    #plt.plot(freq, amp, 'o')
    #plt.plot(fax, fit(fax))

    from scipy.optimize import fmin
    def outlierQ(function, x, y, threshold):
        def D2(z, function, x, y):
            return (x-z)**2 + (y-function(z))**2

        xopt = fmin(D2, x, args=(function, x, y), disp=False)

        return True if D2(xopt, function,x,y) > threshold else False

    dropit = []
    for index, row in df.iterrows():
        if outlierQ(fit, row['logf'], row['loga'],0.7):
            dropit.append(index)
    print('Removing {0} outliers.'.format(len(dropit)))
    df.drop(index=dropit, inplace=True)

    df1 = df.copy(deep=True)
    df = df[abs(df['L0'])<2.0]
    df = df[abs(df['L1'])<2.0]
    df = df[abs(df['L2'])<2.0]
    df = df[abs(df['L3'])<2.0]

    df1 = pd.concat([df1,df,df]).drop_duplicates(keep=False)
    print('Number of unitary valid points: ', df['AmpPeakSW'].count())
    print('Number of unitary invalid points: ', df1['AmpPeakSW'].count())

    plt.ioff()

    """

    ax = df1.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', s=0.5, marker='x', color='grey', label='non-unitary', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    df.plot.scatter(ax=ax, x='FreqPeakSW',y='AmpPeakSW', s=0.5, marker='o', color='blue', label='unitary', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.legend()
    plt.savefig('plots/peak.eps')

    df.plot.scatter(x='beta',y='alpha', s=0.5, marker='o', c='VEVdif/T', colormap='viridis', logx=True, grid=True, xlabel=r'$\beta/H$', ylabel=r'$\alpha$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/alphbeta.eps')

    df.plot.scatter(x='beta',y='alpha', s=0.5, marker='o', c='TempNuc', colormap='viridis', logx=True, grid=True, xlabel=r'$\beta/H$', ylabel=r'$\alpha$')
    plt.gcf().get_axes()[1].set_ylabel(r'$T_{\mathrm{n}}$ [GeV]')
    plt.savefig('plots/alphbetaT.eps')

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='beta', colormap='viridis', s=0.5, marker='o', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\beta/H$')
    plt.savefig('plots/peakbeta.eps')

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='alpha', colormap='viridis', s=0.5, marker='o', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\alpha$')
    plt.savefig('plots/peakalpha.eps')

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='VEVdif/T', colormap='viridis', s=0.5, marker='o', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakvevdif.eps')

    df.plot.scatter(x='L0',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$\Lambda_{0}$', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakL0.eps')

    df.plot.scatter(x='L1',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$\Lambda_{1}$', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakL1.eps')

    df.plot.scatter(x='L2',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$\Lambda_{2}$', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakL2.eps')

    df.plot.scatter(x='L3',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$\Lambda_{3}$', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakL3.eps')

    df.plot.scatter(x='L4',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$\Lambda_{4}$', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakL4.eps')

    df.plot.scatter(x='Mn1',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$m_{\mathrm{H}_1}$ [GeV]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakMn1.eps')

    df.plot.scatter(x='Mn2',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakMn2.eps')

    df.plot.scatter(x='Mch1',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$m_{\mathrm{H}_+}$ [GeV]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakMch1.eps')

    df.plot.scatter(x='Mch2',y='AmpPeakSW', color='blue', s=0.5, marker='o', logy=True, grid=True, xlabel=r'$m_{\mathrm{H}_-}$ [GeV]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakMch2.eps')

    df.plot.scatter(x='FreqPeakSW',y='AmpPeakSW', c='MSum', colormap='viridis', s=0.5, marker='o', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.gcf().get_axes()[1].set_ylabel(r'$m_{\mathrm{H}_1}+m_{\mathrm{H}_2}+m_{\mathrm{H}_+}m_{\mathrm{H}_-}$ [GeV]')
    plt.savefig('plots/peakMSum.eps')

    df.plot.scatter(x='MnSum',y='MnDif', c='VEVdif/T', colormap='viridis', s=0.5, marker='o', grid=True, xlabel=r'$m_{\mathrm{H}_1} + m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$m_{\mathrm{H}_1} - m_{\mathrm{H}_2}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/MnSumMnDif.eps')

    df.plot.scatter(x='MchSum',y='MchDif', c='VEVdif/T', colormap='viridis', s=0.5, marker='o', grid=True, xlabel=r'$m_{\mathrm{H}_+} + m_{\mathrm{H}_-}$ [GeV]', ylabel=r'$m_{\mathrm{H}_+} - m_{\mathrm{H}_-}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/MchSumMchDif.eps')

    df.plot.scatter(x='MnSum',y='MchDif', c='VEVdif/T', colormap='viridis', s=0.5, marker='o', grid=True, xlabel=r'$m_{\mathrm{H}_1} + m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$m_{\mathrm{H}_+} - m_{\mathrm{H}_-}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/MnSumMchDif.eps')

    df.plot.scatter(x='MchSum',y='MnDif', c='VEVdif/T', colormap='viridis', s=0.5, marker='o', grid=True, xlabel=r'$m_{\mathrm{H}_+} + m_{\mathrm{H}_-}$ [GeV]', ylabel=r'$m_{\mathrm{H}_1} - m_{\mathrm{H}_2}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/MchSumMnDif.eps')

    """
    m = A4_spectrum(Mn1 = df['Mn1'][100], Mn2 = df['Mn2'][100], Mch1 = df['Mch1'][100], Mch2 = df['Mch2'][100], verbose = 1, forcetrans=True, T_eps=5e-4, path = './bin/', betamax=1E6)
    if m.spectrainfo == []:
        m.genspec()

    m.plot1d([0,0,0,0,0],[246.22/np.sqrt(3),246.22/np.sqrt(3),0,246.22/np.sqrt(3),0],T=[0,df['TempNuc'][100],df['TempCrit'][100],100])
    plt.savefig('plots/vplot1dradreal.eps')

    m.plot1dtht(0,2*np.pi,139.007485,caxs=[1,3],saxs=[2,4],T=[0,df['TempNuc'][100],df['TempCrit'][100],100])
    plt.savefig('plots/vplotarg.eps')

    m.plot2d((-150,150,-150,150),T=[0,df['TempNuc'][100],df['TempCrit'][100],100],n=200, xaxis=[0,1], yaxis=[3], clevs=100,cfrac=0.8, filled=False)
    plt.savefig('plots/vplot2dx01y3.eps')

    m.plot2d((-150,150,-150,150),T=[0,df['TempNuc'][100],df['TempCrit'][100],100],n=200, xaxis=[0,1,3], yaxis=[2,4], clevs=100,cfrac=0.8, filled=False)
    plt.savefig('plots/vplot2dx013y24.eps')

    m.plot2d((-150,150,-150,150),T=[0,df['TempNuc'][100],df['TempCrit'][100],100],n=200, xaxis=[1], yaxis=[2], clevs=100,cfrac=0.8, filled=False)
    plt.savefig('plots/vplot2dx1y2.eps')



    return


    df.plot.scatter(x='FreqPeakSW',y='beta', s=2, marker='o', color='green', loglog=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$\beta$')
    df.plot.scatter(x='FreqPeakSW',y='TempNuc', s=2, marker='o', color='green', logx=True, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$T^{\mathrm{nuc}}$ [GeV]')
    df.plot.scatter(x='AmpPeakSW',y='beta', s=2, marker='o', color='green', loglog=True, grid=True, xlabel=r'$\Omega^{\mathrm{peak}}$', ylabel=r'$\beta$')
    df.plot.scatter(x='AmpPeakSW',y='TempNuc', s=2, marker='o', color='green', logx=True, grid=True, xlabel=r'$\Omega^{\mathrm{peak}}$', ylabel=r'$T^{\mathrm{nuc}}$ [GeV]')
    df.plot.scatter(x='TempNuc',y='b/f', s=2, marker='o', color='green', loglog=False, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$\beta/f^{\mathrm{peak}}$')
    df.plot.scatter(x='TempNuc',y='fp', s=2, marker='o', color='green', loglog=False, grid=True, xlabel=r'$f^{\mathrm{peak}}$ [Hz]', ylabel=r'$\beta/f^{\mathrm{peak}}$')
    df.plot.scatter(x='TempNuc',y='beta', s=2, marker='o', color='green', loglog=True, grid=True, xlabel=r'$T^{\mathrm{nuc}}$ [GeV]', ylabel=r'$\beta$')
    df.plot.scatter(x='TempNuc',y='alpha', s=2, marker='o', color='green', loglog=False, grid=True, xlabel=r'$T^{\mathrm{nuc}}$ [GeV]', ylabel=r'$\alpha$')
    df.plot.scatter(x='VEVdif/T',y='alpha', s=2, marker='o', color='green', loglog=False, grid=True, xlabel=r'$\Delta |\phi| / T_{\mathrm{n}}$', ylabel=r'$\alpha$')
    df.plot.scatter(x='VEVdif/T',y='beta', s=2, marker='o', color='green', loglog=True, grid=True, xlabel=r'$\Delta |\phi| / T_{\mathrm{n}}$', ylabel=r'$\beta$')
    plt.show()
    
    
    #print(df.corr())


    

    ax = df1.plot.scatter(x='L0',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$\Lambda_{0}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    #df.plot.scatter(ax=ax, x='L0',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$\Lambda_{0}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    plt.show()

    df.plot.scatter(x='L1',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$\Lambda_{1}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L2',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$\Lambda_{2}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L3',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$\Lambda_{3}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L4',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$\Lambda_{4}$ [Hz]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='MnSum',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$m_{\mathrm{H}_1} + m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='MnDif',y='AmpPeakSW', s=2, grid=True, logy=True, xlabel=r'$m_{\mathrm{H}_1} - m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$h^2 \Omega^{\mathrm{peak}}_{\mathrm{GW}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='Mn1',y='Mn2', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$m_{\mathrm{H}_1}$ [GeV]', ylabel=r'$m_{\mathrm{H}_2}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='Mn1',y='Mch1', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$m_{\mathrm{H}_1}$ [GeV]', ylabel=r'$m_{\mathrm{H}_+}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='Mn2',y='Mch1', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$m_{\mathrm{H}_+}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='MnSum',y='MnDif', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$m_{\mathrm{H}_1} + m_{\mathrm{H}_2}$ [GeV]', ylabel=r'$m_{\mathrm{H}_1} - m_{\mathrm{H}_2}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='MchSum',y='MchDif', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$m_{\mathrm{H}_+} + m_{\mathrm{H}_-}$ [GeV]', ylabel=r'$m_{\mathrm{H}_+} - m_{\mathrm{H}_-}$ [GeV]')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L0',y='L1', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$\Lambda_0$', ylabel=r'$\Lambda_1$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L1',y='L2', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$\Lambda_1$', ylabel=r'$\Lambda_2$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L3',y='L4', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$\Lambda_3$', ylabel=r'$\Lambda_4$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L1',y='L3', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$\Lambda_1$', ylabel=r'$\Lambda_3$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

    df.plot.scatter(x='L1',y='L4', c='VEVdif/T', colormap='viridis', s=2, grid=True, xlabel=r'$\Lambda_1$', ylabel=r'$\Lambda_4$')
    plt.gcf().get_axes()[1].set_ylabel(r'$\Delta |\phi| / T_{\mathrm{n}}$')
    plt.savefig('plots/peakmnsum.eps')
    #plt.show()

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

    

    
    plt.show()

    return
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
    

    msp = A4_spectrum(Mn1=200.,Mn2=100.,Mch1=120.,Mch2=140., verbose = 1, forcetrans=False, T_eps=1e-3, path='./bin/')
    print(vars(msp))
    if msp.spectra == []:
        msp.genspec()
    msp.geninfo()
    msp.printinfo()
    msp.save()
    return

    m = A4_vev1(Mn1=185.,Mn2=40.,Mch1=60.,Mch2=118., verbose = 1)
    mg = A4_gauge_vev1(Mn1=185.,Mn2=40.,Mch1=60.,Mch2=118., verbose = 1)
    mr = A4_reduced_vev1(Mn1=185.,Mn2=40.,Mch1=60.,Mch2=118., dM0 = mg.dM0, dL0 = mg.dL0, dL1 = mg.dL1, dL2 = mg.dL2, dL3 = mg.dL3, dL4 = mg.dL4, verbose = 1)
    plot1d(m,[0,0,0,0,0,0],[300.,0.,0.,0.,0.,0.],T=[0,135,200])
    plt.axvline(x=246.22, color="grey", linestyle="--")
    plt.savefig('potphys.png')

    print(np.linalg.eigvalsh(mg.d2V(X=[mg.vh/np.sqrt(3), 0., mg.vh/np.sqrt(3), 0., mg.vh/np.sqrt(3), 0.], T=0.)))
    print('BFB: {0}, Unitary: {1}'.format(mr.tree_lvl_conditions(), mr.unitary()))
    mr.getPhases()

    T=np.linspace(0.,400.,41)
    #for t in T:
    #    print("T= {1:3.0f}, X = {0}".format(mg.gradV(X=[0.,0.,0.,0.,0.,0.],T=t),t))


    mr.findAllTransitions()
    mr.prettyPrintTnTrans()

    mr.plotPhasesPhi()
    plt.savefig('phases.png')
    
    print(mr.phases)

    Tcrit = mr.TcTrans[0]['Tcrit']
    Tnuc = mr.TnTrans[0]['Tnuc']
    plot1d(mr,[0,0,0,0,0],[300./np.sqrt(3),300./np.sqrt(3),0.,300./np.sqrt(3),0.],T=[0,Tcrit,Tnuc,200])
    plt.axvline(x=246.22, color="grey", linestyle="--")
    plt.savefig('pot.png')
    
    


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


  