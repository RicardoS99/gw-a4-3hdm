from logging import raiseExceptions
import pickle
from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder
import importlib

from matplotlib import pyplot as plt

from A4_model import A4_vev1
from A4_model_reduced import A4_reduced_vev1
importlib.reload(transitionFinder)
from cosmoTransitions import pathDeformation as pd
import numpy as np
from gw_spectrum import gw_spectrum

class A4_spectrum():
    def __init__(self, Mn1=265.95, Mn2=174.10, Mch1=197.64, Mch2=146.84, verbose=1, x_eps = 1e-3, T_eps = 1e-3, deriv_order = 4, betamin = 10, betamax = 100000, forcetrans = False, maxit = 4, itTonly = False, path = './cache/', cachefile = None):
        self.nameid = str(int(Mn1*100)) + '_' + str(int(Mn2*100)) + '_' + str(int(Mch1*100)) + '_' + str(int(Mch2*100))
        self.path = path
        self.forcetrans = forcetrans
        self.maxit = maxit
        self.itTonly = itTonly
        self.betamin = betamin
        self.betamax = betamax
        self.Mn1 = Mn1
        self.Mn2 = Mn2
        self.Mch1 = Mch1
        self.Mch2 = Mch2
        self.x_eps = x_eps
        self.T_eps = T_eps
        self.deriv_order = deriv_order
        self.verbose = verbose
        self.modelinfo = {}
        self.spectrainfo = []
        self.mphys = A4_vev1(self.Mn1, self.Mn2, self.Mch1, self.Mch2, verbose=0, x_eps = self.x_eps, T_eps = self.T_eps, deriv_order = self.deriv_order)
        self.mgr = A4_reduced_vev1(self.Mn1, self.Mn2, self.Mch1, self.Mch2, dM0 = self.mphys.dM0, dL0 = self.mphys.dL0, dL1 = self.mphys.dL1, dL2 = self.mphys.dL2, dL3 = self.mphys.dL3, dL4 = self.mphys.dL4, verbose=self.verbose, x_eps = self.x_eps, T_eps = self.T_eps, deriv_order = self.deriv_order)
        self.cache = {'phases': self.mgr.phases, 'modelinfo': self.modelinfo, 'spectrainfo': self.spectrainfo}
        self.load(cachefile)
        
        
    def getPhases(self):
        if self.mgr.phases is None:
            self.mgr.getPhases()
            self.__geninfo()
            self.cache.update({'phases': self.mgr.phases})

    def findAllTransitions(self):
        try:
            if self.forcetrans:
                it=0
                switchflag = 1.
                self.mgr.findAllTransitions()
                while len(self.mgr.TnTrans)==0 or it<self.maxit:
                    print("No transitions found with x_eps = {0} and T_eps = {1} .".format(self.mgr.x_eps, self.mgr.T_eps))
                    if switchflag>0:
                        self.mgr.T_eps /= 2.
                        print('Reducing T_eps to {0} .'.format(self.mgr.T_eps))
                    else:
                        self.mgr.x_eps /= 2.
                        print('Reducing x_eps to {0} .'.format(self.mgr.x_eps))
                    if self.itTonly == False:
                        switchflag *= -1.
                    it += 1
                    self.mgr.getPhases()        
                    self.mgr.findAllTransitions()        
            else:
                self.mgr.findAllTransitions()
            
            if len(self.mgr.TnTrans)==0:
                print('Could not find transitions.')
            else:
                self.T_eps = self.mgr.T_eps
                self.x_eps = self.mgr.x_eps
                self.mphys.T_eps = self.T_eps
                self.mphys.x_eps = self.x_eps
            
            self.__geninfo()
            self.cache.update({'phases': self.mgr.phases})
        except:
            print('Could not find transitions.')
            pass

    def genspec(self):
        spectra = []
        if True: #self.mphys.tree_lvl_conditions():# and self.mphys.unitary():
            if self.mgr.TnTrans is None:
                self.findAllTransitions()
                if self.mgr.TnTrans is None:
                    print('No transitions. Stopping GW spectra generation.')
                    return
                if len(self.mgr.TnTrans)==0:
                    print('No transitions. Stopping GW spectra generation.')
                    return

            if(len(self.mgr.TnTrans)>0):
                for ind in range(0,len(self.mgr.TnTrans)):
                    if(self.mgr.TnTrans[0]['trantype']==1):
                        print('Finding gw spectrum for: Mn1={0:7.3f}, Mn2={1:7.3f}, Mch1={2:7.3f}, Mch2={3:7.3f}'.format(self.mgr.Mn1, self.mgr.Mn2, self.mgr.Mch1, self.mgr.Mch2)) 
                        try:
                            gw = gw_spectrum(self.mgr, ind, turb_on=True)
                            if(self.betamin < gw.beta < self.betamax):
                                spectra.append(gw)
                            else:
                                gw.info = {'beta': np.inf}
                                spectra.append(gw)
                                print('Beta out of bounds: {0}'.format(gw.beta))
                        except:
                            print('Could not generate gw spectrum for transition {0}.'.format(ind))
                            pass
        else:
            print('Couplings do not verify constraints:\nBFB - {0}\nUnitary - {1}\nStopping GW spectra generation.'.format(self.mphys.tree_lvl_conditions(),self.mphys.unitary()))
        self.__geninfo(spectra)
        return spectra

    def __geninfo(self, spectra = None):
        self.modelinfo = {
            'Mn1': self.mgr.Mn1,
            'Mn2': self.mgr.Mn2,
            'Mch1': self.mgr.Mch1,
            'Mch2': self.mgr.Mch2,
            'M0': self.mgr.M0,
            'L0': self.mgr.L0,
            'L1': self.mgr.L1,
            'L2': self.mgr.L2,
            'L3': self.mgr.L3,
            'L4': self.mgr.L4,
            'dM0': self.mgr.dM0,
            'dL0': self.mgr.dL0,
            'dL1': self.mgr.dL1,
            'dL2': self.mgr.dL2,
            'dL3': self.mgr.dL3,
            'dL4': self.mgr.dL4
        }

        if self.mgr.phases is not None:
            self.modelinfo.update({'NPhases': len(self.mgr.phases)})

        if self.mgr.TnTrans is not None:
            self.modelinfo.update({'NTrans': len(self.mgr.TnTrans)})

        if spectra is not None:
            for spectrum in spectra:
                self.spectrainfo.append(spectrum.info)

        self.cache.update({'modelinfo': self.modelinfo, 'spectrainfo': self.spectrainfo})
        print('here')


    def printinfo(self):
        for info in self.spectrainfo:
            print(info)

    def getinfo(self):
        output = []
        for spec in self.spectrainfo:
            output.append({**self.modelinfo, **spec})
        
        return output

    #------------Save/Load Methods----------------

    def save(self, cachefile = None):
        if cachefile is None:
            cachefile = self.path + self.nameid + '.pickle'
        with open(cachefile, 'wb') as f:
                pickle.dump(self.cache,f,-1)
                print('Saved Successfully!')
                return True
        with open(self.path + self.nameid + '.pickle', 'wb') as f:
                pickle.dump(self.__dict__,f,-1)
                print('Saved Successfully!')
                return True
        try:
            with open(self.path + self.nameid, 'wb') as f:
                pickle.dump(self.__dict__,f,-1)
                print('Saved Successfully!')
                return True
        except:
            print('Save Failed!')
            return False

    def load(self, cachefile = None):
        try:
            if cachefile is None:
                cachefile = self.path + self.nameid + '.pickle'
            with open(cachefile, 'rb') as f:
                tmp_cache = pickle.load(f)
                self.cache.update(tmp_cache)
                self.modelinfo = self.cache['modelinfo']
                self.spectrainfo = self.cache['spectrainfo']
                self.Mn1 = self.modelinfo['Mn1']
                self.Mn2 = self.modelinfo['Mn2']
                self.Mch1 = self.modelinfo['Mch1']
                self.Mch2 = self.modelinfo['Mch2']
                self.mphys = A4_vev1(self.Mn1, self.Mn2, self.Mch1, self.Mch2, verbose=0, x_eps = self.x_eps, T_eps = self.T_eps, deriv_order = self.deriv_order)
                self.mgr = A4_reduced_vev1(self.Mn1, self.Mn2, self.Mch1, self.Mch2, dM0 = self.mphys.dM0, dL0 = self.mphys.dL0, dL1 = self.mphys.dL1, dL2 = self.mphys.dL2, dL3 = self.mphys.dL3, dL4 = self.mphys.dL4, verbose=self.verbose, x_eps = self.x_eps, T_eps = self.T_eps, deriv_order = self.deriv_order)
                self.mgr.phases = self.cache['phases']
            print('Loaded Successfully!')
            return True
        except:
            print('Load Failed!')
            return False

    #------------Plotting Methods-----------------

    def plot2d(self, box, T=[0], treelevel=False, offset=0, xaxis=[0], yaxis=[1], n=50, clevs=200, cfrac=.8, filled=False, **contourParams):
        """
        Makes a countour plot of the potential.
        Parameters
        ----------
        box : tuple
            The bounding box for the plot, (xlow, xhigh, ylow, yhigh).
        T : list of floats, optional
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
        XY = np.zeros((n,n,self.mgr.Ndim))
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
                Z = self.mgr.V0(XY) if treelevel else self.mgr.Vtot(XY,T[k])
                minZ, maxZ = min(Z.ravel()), max(Z.ravel())
                N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
                axs[i,j].contourf(X,Y,Z, N, **contourParams) if filled else axs[i,j].contour(X,Y,Z, N, **contourParams)
                axs[i,j].set_title('T = {:.1f} [GeV]'.format(T[k]))
                axs[i,j].set_aspect(1)
                k+=1
          else:
            for j in range(ncols-rest):
                Z = self.mgr.V0(XY) if treelevel else self.mgr.Vtot(XY,T[k])
                minZ, maxZ = min(Z.ravel()), max(Z.ravel())
                N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
                axs[i,j].contourf(X,Y,Z, N, **contourParams) if filled else axs[i,j].contour(X,Y,Z, N, **contourParams)
                axs[i,j].set_title('T = {:.1f} [GeV]'.format(T[k]))
                axs[i,j].set_aspect(1)
                k+=1

    def plot1d(self, x1, x2, T=[0], treelevel=False, subtract=True, n=500, **plotParams):
        plt.figure()
        if self.mgr.Ndim == 1:
            x = np.linspace(x1,x2,n)
            X = x[:,np.newaxis]
        else:
            dX = np.array(x2)-np.array(x1)
            X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
            x = np.linspace(0,1,n)*np.sum(dX**2)**.5

        for t in T:
            if treelevel:
                y = self.mgr.V0(X) - self.mgr.V0(X*0) if subtract else self.mgr.V0(X)
            else:
                y = self.mgr.DVtot(X,t) if subtract else self.mgr.Vtot(X, t)
            plt.plot(x,y, **plotParams, label = 'T = {:.1f} [GeV]'.format(t))
        plt.xlabel(R"$|\phi|$")
        plt.ylabel(R"$V(\phi)$")
        plt.legend()
        plt.axhline(y=0, color="grey", linestyle="--")

    def plot1dtht(self, tmin, tmax, vabs, caxs=[0], saxs=[1], T=[0], treelevel=False, subtract=True, n=500, **plotParams):
        plt.figure()
        X = np.zeros((n,self.mgr.Ndim))
        tht = np.linspace(tmin,tmax,n)

        for i in range(n):
            X[i,0] = vabs
            for ax in caxs:
                X[i,np.abs(ax)] = np.sign(ax)*vabs*np.cos(tht[i])
            for ax in saxs:
                X[i,np.abs(ax)] = np.sign(ax)*vabs*np.sin(tht[i])



        for t in T:
            if treelevel:
                y = self.mgr.V0(X) - self.mgr.V0(X*0) if subtract else self.mgr.V0(X)
            else:
                y = self.mgr.DVtot(X,t) if subtract else self.mgr.Vtot(X, t)
            plt.plot(tht,y, **plotParams, label = 'T = {:.1f} [GeV]'.format(t))
        plt.xlabel(R"$\theta$")
        plt.ylabel(R"$V(\phi)$")
        plt.legend()
        plt.axhline(y=0, color="grey", linestyle="--")


    def plotActionT(self, transind, Tmin=0.001, Tmax=500., n=50):
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

        x1 = self.mgr.TnTrans[transind]['low_vev']
        x0 = self.mgr.TnTrans[transind]['high_vev']


        T_vec = np.linspace(Tmin, Tmax, n)
        S_vec = np.zeros_like(T_vec)

        for i in range(0, len(S_vec)):
            try:
                print("Calculating Action for Temperature:", T_vec[i])
                S_vec[i] = compute_action_my(self.mgr,x1,x0,T_vec[i])
            except:
                print("Error calculating")

        ind_to_rem = np.where(np.abs(S_vec)<1e-8)
        T_vec = np.delete(T_vec, ind_to_rem)
        S_vec = np.delete(S_vec, ind_to_rem)

        return T_vec, S_vec
