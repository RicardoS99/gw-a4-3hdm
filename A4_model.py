from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder
import importlib
importlib.reload(transitionFinder)
from cosmoTransitions import pathDeformation as pd
import numpy as np


class A4_vev1(generic_potential.generic_potential):
    def init(self,Mn1=265.95,Mn2=174.10,Mch1=197.64,Mch2=146.84,Mh=125.10,mZ = 91.1876,mW = 80.379,mtop = 172.76,mbot = 4.18,mtau = 1.77686, mcharm = 1.27, mstrange = 0.093, mmuon = 0.1056583745, mup = 0.00216, mdown = 0.00467, melectron = 0.0005109989461, vh = 246.22, counterterms = True, verbose = 1, x_eps = 1e-3, T_eps = 1e-3, deriv_order = 4):
        # SU(2)_L and U(1)_Y couplings
        self.mW = mW
        self.mZ = mZ
        self.vh = vh

        self.gl = 2*self.mW/self.vh
        self.gy = 2*np.sqrt(self.mZ**2 - self.mW**2)/self.vh
        
        self.Ndim = 6 # Number of dynamic classical fields. 1 real field + 2 complex fields -> 5 real fields 
        self.renormScaleSq = float(self.vh**2) # Renormalisation scale
        self.Tmax = 250.
        self.x_eps = x_eps
        self.T_eps = T_eps
        self.deriv_order = deriv_order
        
        if ((6.*(float(Mn1)**2 - float(Mn2)**2)/(self.vh**2))**2 - 12*(2.*np.sqrt(3)*(float(Mch1)**2 - float(Mch2)**2)/(self.vh**2))**2 >0. ): #Check if masses are allowed
            self.Mn1 = float(Mn1)
            self.Mn2 = float(Mn2)
            self.Mch1 = float(Mch1)
            self.Mch2 = float(Mch2) 
        else:
            raise ValueError ('Invalid Masses')
        
        #Calculate constants from masses
        self.M0 = np.sqrt(3)/2.*Mh**2
        self.L1 = -(self.Mch1**2 + self.Mch2**2)/(self.vh**2)
        self.L4 = 2.*np.sqrt(3)*(self.Mch1**2 - self.Mch2**2)/(self.vh**2)
        self.L0 = np.sqrt(3)*self.M0/(self.vh**2) - self.L1
        xc1 = 6.*(self.Mn1**2 + self.Mn2**2)/(self.vh**2) + 5*self.L1
        xc2 = np.sqrt((6.*(self.Mn1**2 - self.Mn2**2)/(self.vh**2))**2 - 12*self.L4**2)
        self.L2 = 1/6.*(xc1 + xc2 + self.L1)
        self.L3 = 1/4.*(xc1 - xc2 - self.L1)

        sf = np.sqrt(12)*self.L4
        cf = self.L1-3*self.L2+2*self.L3
        f = np.sqrt(sf**2 +cf**2)
        self.sina = sf / f
        self.cosa = cf / f

        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)

        self.dM0 = 0
        self.dL0 = 0
        self.dL1 = 0
        self.dL2 = 0
        self.dL3 = 0
        self.dL4 = 0
        
        # Yukawa Couplings
        self.yt   = np.sqrt(2)*mtop/self.vh
        self.yb   = np.sqrt(2)*mbot/self.vh
        self.ytau = np.sqrt(2)*mtau/self.vh
        self.yc   = np.sqrt(2)*mcharm/self.vh
        self.ys   = np.sqrt(2)*mstrange/self.vh
        self.ymuon = np.sqrt(2)*mmuon/self.vh 
        self.yu   = np.sqrt(2)*mup/self.vh
        self.yd   = np.sqrt(2)*mdown/self.vh
        self.ye = np.sqrt(2)*melectron/self.vh     

        # Daisy thermal loop corrections
        self.cT = (14./3.*self.L0 + self.L1 + self.L2 + 2./3.*self.L3 + 3./2.*self.gl**2 + 3./4.*(self.gl**2 + self.gy**2) + 3.*(self.yt**2 + self.yb**2 + self.yc**2 + self.ys**2 + self.yu**2 + self.yd**2) + (self.ytau**2 + self.ymuon**2 + self.ye**2))/24.

        # Counterterms: Apply the renormalization conditions both to fix the minimum of the potential and the masses (first and second derivatives)
        tl_vevs = np.array([0.,self.vh ,0., 0., 0., 0.]) # Set VEVs at T=0
        gradT0 = self.gradV(tl_vevs,0.) # Take the first derivatives of the potential at the minimum
        Hess = self.d2V(tl_vevs,0.) # Use built-in d2V function to extrac the second derivative of the T=0 potential (tree + CW)        
        Hess_tree = self.massSqMatrix(tl_vevs) # Use analytical form of mass matrix in the gauge eigenbasis in the minimum
        d2V_CW = Hess - Hess_tree # Extrac the second derivative of the CW potential by removing the tree-level part from the full tree+CW T=0 bit 

        d1 = -gradT0[1]
        d2 = -d2V_CW[1,1]
        d3 = -(d2V_CW[2,2] + d2V_CW[3,3])/2.
        d4 = -(d2V_CW[4,4] + d2V_CW[5,5])/2.
        d5 = -(d2V_CW[2,4] + d2V_CW[4,2] - d2V_CW[5,3] - d2V_CW[3,5])/4.

        if counterterms is True:
            if verbose > 1 :
                print('d2V_CW[0,0]: ', d2V_CW[0,0])
                print('d2V_CW[1,1]: ', d2V_CW[1,1])
                print('d2V_CW[2,2]: ', d2V_CW[2,2])
                print('d2V_CW[3,3]: ', d2V_CW[3,3])
                print('d2V_CW[4,4]: ', d2V_CW[4,4])
                print('d2V_CW[5,5]: ', d2V_CW[5,5])

                print('d2V_CW[2,4]: ', d2V_CW[2,4])
                print('d2V_CW[4,2]: ', d2V_CW[4,2])
                print('d2V_CW[5,3]: ', d2V_CW[5,3])
                print('d2V_CW[3,5]: ', d2V_CW[3,5])

                print('d1: ', d1)
                print('d2: ', d2)
                print('d3: ', d3)
                print('d4: ', d4)
                print('d5: ', d5)

                print('1st derivatives', self.gradV(tl_vevs,0.))
            self.dM0 = -(3*np.sqrt(3)*d1 - np.sqrt(3)*d2*self.vh)/(2.*self.vh)
            self.dL1 = (3*(-d1 + d2*self.vh))/(2.*self.vh**3)
            self.dL2 = -(7*d1 - 3*d2*self.vh - 2*d3*self.vh - 2*d4*self.vh - 2*d3*self.cosa*self.vh + 2*d4*self.cosa*self.vh - 4*d5*self.vh*sqcm*sqcp)/(2.*self.vh**3)
            self.dL3 = (3*(-3*d1 + d2*self.vh + d3*self.vh + d4*self.vh - d3*self.cosa*self.vh + d4*self.cosa*self.vh - 2*d5*self.vh*sqcm*sqcp))/(2.*self.vh**3)
            self.dL4 = -((np.sqrt(3)*d3*sqcm - np.sqrt(3)*d4*sqcm + np.sqrt(3)*d3*self.cosa*sqcm - np.sqrt(3)*d4*self.cosa*sqcm - 2*np.sqrt(3)*d5*self.cosa*sqcp)/ (np.sqrt(1 + self.cosa)*self.vh**2))
        
            if verbose > 1 :
                print('1st derivatives', self.gradV(tl_vevs,0.))
                d2V_ct = self.d2V(tl_vevs,0.) - Hess
                d2V_CW[np.abs(d2V_CW) < 1] = 0
                d2V_ct[np.abs(d2V_ct) < 1] = 0
                dif = d2V_ct + d2V_CW
                dif[np.abs(dif) < 1] = 0
                print('2nd derivatives: ',dif)
                print('Hessian Eigenvalues: ', np.linalg.eigvals(self.d2V(tl_vevs,0.)))
        
        if verbose > 0 :
            # Printing Constants, Counterterms, VEVs at T=0 and Days thermal loop corrections
            print('Couplings:     M0 = {0:8.2f},  L0 = {1:6.3f},  L1 = {2:6.3f},  L2 = {3:6.3f},  L3 = {4:6.3f},  L4 = {5:6.3f}'.format(self.M0,self.L0,self.L1,self.L2,self.L3,self.L4))
            print('Counterterms: dM0 = {0:8.2f}, dL0 = {1:6.3f}, dL1 = {2:6.3f}, dL2 = {3:6.3f}, dL3 = {4:6.3f}, dL4 = {5:6.3f}'.format(self.dM0,self.dL0,self.dL1,self.dL2,self.dL3,self.dL4))
            print('VEVS:          n0 = {0:8.2f},  n1 = {1:6.2f},  n2 = {2:6.2f},  n3 = {3:6.2f},  n4 = {4:6.2f},  n5 = {5:6.2f}'.format(tl_vevs[0],tl_vevs[1],tl_vevs[2],tl_vevs[3],tl_vevs[4],tl_vevs[5]))
            print('Termal Correction: {0:6.2f}'.format(self.cT))

    def tree_lvl_conditions(self):
        # Here cond=true means that one is in the physical region at tree level
        M0 = self.M0 + self.dM0
        L0 = self.L0 + self.dL0
        L1 = self.L1 + self.dL1
        L2 = self.L2 + self.dL2
        L3 = self.L3 + self.dL3
        L4 = self.L4 + self.dL4

        #x = np.array(np.meshgrid([0,1], [0,3/4.], [0,1], [-np.sqrt(3)/4.,0,np.sqrt(3)/4])).T.reshape(-1, 4)
        x =[
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 3/4., 1/4., 0.],
            [1/4., 3/4., 0., np.sqrt(3)/4.],
            [1/4., 3/4., 0., -np.sqrt(3)/4.],
            [1/2., (9-6*np.sqrt(2))/2., -4.+3*np.sqrt(2), (np.sqrt(6)-np.sqrt(3))/2.],
            [1/2., (9-6*np.sqrt(2))/2., -4.+3*np.sqrt(2), -(np.sqrt(6)-np.sqrt(3))/2.],
            [1/2., 1/2., 0., np.sqrt(3)/6.],
            [1/2., 1/2., 0., -np.sqrt(3)/6.],
        ]
        Li = np.array([[L1],[L2],[L3],[L4]])
        req1 = M0 > 0
        req2 = (L0 + np.dot(x,Li).ravel() > 0).all()
        req3 = L4**2 < 12*L1**2
        req4 = L4**2 < 2*(L3 - L1)*(L2 - L1)
        
        cond = req1 and req2 and req3 and req4

        return cond

    def unitary(self):
        # Here cond=true means that one is in the physical region at tree level

        L0 = self.L0 + self.dL0
        L1 = self.L1 + self.dL1
        L2 = self.L2 + self.dL2
        L3 = self.L3 + self.dL3
        L4 = self.L4 + self.dL4

        req1 = np.abs(2/3.*L0-L1/2.+L2/2.) < 1/(8*np.pi) 
        req2 = np.abs(2/3.*L0+L1-L2) < 1/(8*np.pi) 
        req3 = np.abs(L1/2.+L2/2.+L3) < 1/(8*np.pi) 
        req4 = np.abs(L1/2.+L2/2.-L3) < 1/(8*np.pi) 
        req5 = np.abs(2/3.*L0+L1+L2) < 1/(8*np.pi) 
        req6 = np.abs(2/3.*L0-L1/2.-L2/2.) < 1/(8*np.pi) 
        req7 = np.abs(L1/2.-L2/2.+L3) < 1/(8*np.pi) 
        req8 = np.abs(-L1/2.+L2/2.+L3) < 1/(8*np.pi) 
        req9 = np.abs(2*L0+2*L3-L1/2.-L2/2.) < 1/(8*np.pi) 
        req10 = np.abs(2*L0-4*L3+L1+L2) < 1/(8*np.pi) 
        req11 = np.abs(-L3+5/2.*L1-L2/2.) < 1/(8*np.pi) 
        req12 = np.abs(-L3-L1/2.+5/2.*L2) < 1/(8*np.pi)

        return True
        #return req1 and req2 and req3 and req4 and req5 and req6 and req7 and req8 and req9 and req10 and req11 and req12

    def forbidPhaseCrit(self, X):
        X = np.asanyarray(X)
        n0,n1,n2,n3,n4,n5 = X[...,0], X[...,1], X[...,2], X[...,3], X[...,4], X[...,5]

        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)

        H10 = n1/np.sqrt(3) + ( n2 - n3/np.sqrt(3))/2. * sqcm + (-n4 - n5/np.sqrt(3))/2. * sqcp
        H11 = n0/np.sqrt(3) + ( n5 - n4/np.sqrt(3))/2. * sqcm + (-n3 - n2/np.sqrt(3))/2. * sqcp

        req1 = (H10 < -5).any()
        req2 = (H11 < -5).any()
        req3 = (H11 >  5).any()

        cond = req1 or req2 or req3

        return cond

    def V0(self, X):

        # A4 flavour symmetric potential with three SU(2) doublets
        # Each doublet has a complex VEV on the neutral component treated as a complex field
        # Each complex field will be treated as two real fields
        # Hi0 and Hi1 correspond to the real and complex parts resp.
        
        X = np.asanyarray(X)
        n0,n1,n2,n3,n4,n5 = X[...,0], X[...,1], X[...,2], X[...,3], X[...,4], X[...,5]

        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)

        H10 = n1/np.sqrt(3) + ( n2 - n3/np.sqrt(3))/2. * sqcm + (-n4 - n5/np.sqrt(3))/2. * sqcp
        H11 = n0/np.sqrt(3) + ( n5 - n4/np.sqrt(3))/2. * sqcm + (-n3 - n2/np.sqrt(3))/2. * sqcp
        H20 = n1/np.sqrt(3) + (-n2 - n3/np.sqrt(3))/2. * sqcm + ( n4 - n5/np.sqrt(3))/2. * sqcp
        H21 = n0/np.sqrt(3) + (-n5 - n4/np.sqrt(3))/2. * sqcm + ( n3 - n2/np.sqrt(3))/2. * sqcp
        H30 = n1/np.sqrt(3) + n3/np.sqrt(3) * sqcm + n5/np.sqrt(3) * sqcp
        H31 = n0/np.sqrt(3) + n4/np.sqrt(3) * sqcm + n2/np.sqrt(3) * sqcp

        b11 = 1/2.*(H10**2 + H11**2)
        b22 = 1/2.*(H20**2 + H21**2)
        b33 = 1/2.*(H30**2 + H31**2)
        b12r = 1/2.*(H10*H20 + H11*H21)
        b12i = 1/2.*(-H20*H11 + H10*H21)
        b23r = 1/2.*(H20*H30 + H21*H31)
        b23i = 1/2.*(-H30*H21 + H20*H31)
        b31r = 1/2.*(H30*H10 + H31*H11)
        b31i = 1/2.*(-H10*H31 + H30*H11)
        
        v0 = -(self.M0)/np.sqrt(3.)*(b11 + b22 + b33) \
        + 1/3.*(self.L0)*(b11 + b22 + b33)**2  \
        + 1/3.*(self.L3)*(b11**2 + b22**2 + b33**2 - b11*b22 - b22*b33 - b33*b11)  \
        + (self.L1)*(b12r**2 + b23r**2 + b31r**2)  \
        + (self.L2)*(b12i**2 + b23i**2 + b31i**2)  \
        + (self.L4)*(b12r*b12i + b23r*b23i + b31r*b31i)

        vct = -(self.dM0)/np.sqrt(3.)*(b11 + b22 + b33) \
        + 1/3.*(self.dL0)*(b11 + b22 + b33)**2  \
        + 1/3.*(self.dL3)*(b11**2 + b22**2 + b33**2 - b11*b22 - b22*b33 - b33*b11)  \
        + (self.dL1)*(b12r**2 + b23r**2 + b31r**2)  \
        + (self.dL2)*(b12i**2 + b23i**2 + b31i**2)  \
        + (self.dL4)*(b12r*b12i + b23r*b23i + b31r*b31i)

        return v0 + vct

### Field-dependent masses are generic
# Calculations must be prepared to handle ndarrays as input
# X is a (..., Ndim) matrix containing multiple values for all dynamic fields
# The first axis correspond to the multiple values the fields may take
# The second axis correspond to the fields themselves
# M must be a (..., Nf) matrix where Nf is the number of field-dependent masses

    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        n0,n1,n2,n3,n4,n5 = X[...,0], X[...,1], X[...,2], X[...,3], X[...,4], X[...,5]

        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)

        H10 = n1/np.sqrt(3) + ( n2 - n3/np.sqrt(3))/2. * sqcm + (-n4 - n5/np.sqrt(3))/2. * sqcp
        H11 = n0/np.sqrt(3) + ( n5 - n4/np.sqrt(3))/2. * sqcm + (-n3 - n2/np.sqrt(3))/2. * sqcp
        H20 = n1/np.sqrt(3) + (-n2 - n3/np.sqrt(3))/2. * sqcm + ( n4 - n5/np.sqrt(3))/2. * sqcp
        H21 = n0/np.sqrt(3) + (-n5 - n4/np.sqrt(3))/2. * sqcm + ( n3 - n2/np.sqrt(3))/2. * sqcp
        H30 = n1/np.sqrt(3) + n3/np.sqrt(3) * sqcm + n5/np.sqrt(3) * sqcp
        H31 = n0/np.sqrt(3) + n4/np.sqrt(3) * sqcm + n2/np.sqrt(3) * sqcp
        
         # Scalar fields in gauge basis: H1, Eta1, Chi1, Chip1, H2, Eta2, Chi2, Chip2, H3, Eta3, Chi3, Chip3
        
        # Thermal correction do add on diagonal of mass matrices
        thcorr = np.full_like(n0, 2*self.cT*T**2)

        M0 = self.M0
        L0 = self.L0
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4
        
        # C: charged component fields (Chi1, Chip1, Chi2, Chip2, Chi3, Chip3) mass matrix
        # Careful with 0 (zero) entries. Must be same type as other entries which maybe ndarrays of floats or complexs.
        # Use np.zeros_like(X[...,0]) to assure same type. 
        cm = np.array([[(-4*np.sqrt(3)*M0 + 2*sqcm**2*n2**2*L0 + 2*sqcp**2*n2**2*L0 + 2*sqcm**2*n3**2*L0 + 2*sqcp**2*n3**2*L0 + 2*sqcm**2*n4**2*L0 + 2*sqcp**2*n4**2*L0 + 2*sqcm**2*n5**2*L0 + 2*sqcp**2*n5**2*L0 - (sqcm**2 + sqcp**2)*(n2**2 + n3**2 + n4**2 + n5**2)*L1 + 4*n0**2*(L0 + L1) + 4*n1**2*(L0 + L1) - np.sqrt(3)*sqcm*sqcp*n2**2*L4 - np.sqrt(3)*sqcm*sqcp*n3**2*L4 - np.sqrt(3)*sqcm**2*n2*n4*L4 + np.sqrt(3)*sqcp**2*n2*n4*L4 + np.sqrt(3)*sqcm*sqcp*n4**2*L4 + np.sqrt(3)*sqcm**2*n3*n5*L4 - np.sqrt(3)*sqcp**2*n3*n5*L4 + np.sqrt(3)*sqcm*sqcp*n5**2*L4)/12. + thcorr, \
                np.zeros_like(n0), \
                (-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + sqcm*(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + 2*n0*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 - np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + 2*n1*(2*n3*L1 - 6*n5*L2 + 4*n3*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4) + n0*(-6*n4*L2 + np.sqrt(3)*n4*L4 + n2*(2*L1 + 4*L3 - np.sqrt(3)*L4))))/48., \
                (-(sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + sqcp*(sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n4*L1 - 6*n2*L2 + 4*n4*L3 + np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + n0*(2*n3*L1 + 6*n5*L2 + 4*n3*L3 - np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4)) + sqcm*(2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n0*(-2*n5*L1 + 6*n3*L2 - 4*n5*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4) + n1*(-6*n4*L2 + np.sqrt(3)*n4*L4 + n2*(-2*L1 - 4*L3 + np.sqrt(3)*L4))))/24., \
                (-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*(2*n2*L1 + 6*n4*L2 + 4*n2*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4) + n1*(2*n5*L1 - 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + sqcm*(-2*n0*n2*(6*L2 + np.sqrt(3)*L4) + sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + 2*n0*n4*(2*L1 + 4*L3 + np.sqrt(3)*L4) + 2*n1*(2*n3*L1 + 6*n5*L2 + 4*n3*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/48., \
                (sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) - sqcp*(-(n0*n5*(6*L2 + np.sqrt(3)*L4)) + sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*n3*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n1*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4)) + sqcm*(-(n1*n4*(6*L2 + np.sqrt(3)*L4)) - 2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n1*n2*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n0*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/24.], \
                [np.zeros_like(n0),\
                (-4*np.sqrt(3)*M0 + 2*sqcm**2*n2**2*L0 + 2*sqcp**2*n2**2*L0 + 2*sqcm**2*n3**2*L0 + 2*sqcp**2*n3**2*L0 + 2*sqcm**2*n4**2*L0 + 2*sqcp**2*n4**2*L0 + 2*sqcm**2*n5**2*L0 + 2*sqcp**2*n5**2*L0 - (sqcm**2 + sqcp**2)*(n2**2 + n3**2 + n4**2 + n5**2)*L1 + 4*n0**2*(L0 + L1) + 4*n1**2*(L0 + L1) - np.sqrt(3)*sqcm*sqcp*n2**2*L4 - np.sqrt(3)*sqcm*sqcp*n3**2*L4 - np.sqrt(3)*sqcm**2*n2*n4*L4 + np.sqrt(3)*sqcp**2*n2*n4*L4 + np.sqrt(3)*sqcm*sqcp*n4**2*L4 + np.sqrt(3)*sqcm**2*n3*n5*L4 - np.sqrt(3)*sqcp**2*n3*n5*L4 + np.sqrt(3)*sqcm*sqcp*n5**2*L4)/12. + thcorr, \
                (sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + sqcm*(-2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n2*L1 + 6*n4*L2 + 4*n2*L3 - np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + n0*(2*n5*L1 - 6*n3*L2 + 4*n5*L3 + np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4)) + sqcp*(-(sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + n1*(-2*n4*L1 + 6*n2*L2 - 4*n4*L3 - np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4) + n0*(-6*n5*L2 + np.sqrt(3)*n5*L4 + n3*(-2*L1 - 4*L3 + np.sqrt(3)*L4))))/24., \
                (-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + sqcm*(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + 2*n0*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 - np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + 2*n1*(2*n3*L1 - 6*n5*L2 + 4*n3*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4) + n0*(-6*n4*L2 + np.sqrt(3)*n4*L4 + n2*(2*L1 + 4*L3 - np.sqrt(3)*L4))))/48., \
                (-(sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4)) + sqcp*(-(n0*n5*(6*L2 + np.sqrt(3)*L4)) + sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*n3*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n1*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4)) - sqcm*(-(n1*n4*(6*L2 + np.sqrt(3)*L4)) - 2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n1*n2*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n0*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/24., \
                (-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*(2*n2*L1 + 6*n4*L2 + 4*n2*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4) + n1*(2*n5*L1 - 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + sqcm*(-2*n0*n2*(6*L2 + np.sqrt(3)*L4) + sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + 2*n0*n4*(2*L1 + 4*L3 + np.sqrt(3)*L4) + 2*n1*(2*n3*L1 + 6*n5*L2 + 4*n3*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/48.], \
                [(-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + sqcm*(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + 2*n0*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 - np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + 2*n1*(2*n3*L1 - 6*n5*L2 + 4*n3*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4) + n0*(-6*n4*L2 + np.sqrt(3)*n4*L4 + n2*(2*L1 + 4*L3 - np.sqrt(3)*L4))))/48., \
                (sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + sqcm*(-2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n2*L1 + 6*n4*L2 + 4*n2*L3 - np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + n0*(2*n5*L1 - 6*n3*L2 + 4*n5*L3 + np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4)) + sqcp*(-(sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + n1*(-2*n4*L1 + 6*n2*L2 - 4*n4*L3 - np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4) + n0*(-6*n5*L2 + np.sqrt(3)*n5*L4 + n3*(-2*L1 - 4*L3 + np.sqrt(3)*L4))))/24., \
                (-16*np.sqrt(3)*M0 + 16*n1**2*L0 + 8*sqcm**2*n2**2*L0 + 8*sqcp**2*n2**2*L0 + 8*sqcm**2*n3**2*L0 + 8*sqcp**2*n3**2*L0 + 8*sqcm**2*n4**2*L0 + 8*sqcp**2*n4**2*L0 + 8*sqcm**2*n5**2*L0 + 8*sqcp**2*n5**2*L0 - 8*n1**2*L1 + 2*sqcm**2*n2**2*L1 + 2*sqcp**2*n2**2*L1 + 2*sqcm**2*n3**2*L1 + 2*sqcp**2*n3**2*L1 + 2*sqcm**2*n4**2*L1 + 2*sqcp**2*n4**2*L1 + 2*sqcm**2*n5**2*L1 + 2*sqcp**2*n5**2*L1 + 12*(sqcm**2*(n2*n4 - n3*n5) + sqcp**2*(-(n2*n4) + n3*n5) + sqcm*sqcp*(n2**2 + n3**2 - n4**2 - n5**2))*L2 - 4*np.sqrt(3)*n1**2*L4 + np.sqrt(3)*sqcm**2*n2**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 + np.sqrt(3)*sqcp**2*n2**2*L4 + np.sqrt(3)*sqcm**2*n3**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 + np.sqrt(3)*sqcp**2*n3**2*L4 + 2*np.sqrt(3)*sqcm**2*n2*n4*L4 - 2*np.sqrt(3)*sqcp**2*n2*n4*L4 + np.sqrt(3)*sqcm**2*n4**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 + np.sqrt(3)*sqcp**2*n4**2*L4 - 2*np.sqrt(3)*sqcm**2*n3*n5*L4 + 2*np.sqrt(3)*sqcp**2*n3*n5*L4 + np.sqrt(3)*sqcm**2*n5**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4 + np.sqrt(3)*sqcp**2*n5**2*L4 + 4*n0**2*(4*L0 - 2*L1 - np.sqrt(3)*L4))/48. + thcorr, \
                np.zeros_like(n0), \
                (sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3) + 2*sqcp*(2*sqcm*(n2*n4 + n3*n5)*(2*L1 + L3) + n1*(-2*n5*L1 + 2*n5*L3 + np.sqrt(3)*n3*L4) - n0*(2*n2*(L1 - L3) + np.sqrt(3)*n4*L4)) + sqcm*(-(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3)) + 2*n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - 2*n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4)))/24., \
                (sqcm**2*(n2*n3 + n4*n5)*(2*L1 + L3) + sqcm*(2*n1*n2*(L1 - L3) + 2*n0*n5*(L1 - L3) - 2*sqcp*(n3*n4 - n2*n5)*(2*L1 + L3) + np.sqrt(3)*n0*n3*L4 - np.sqrt(3)*n1*n4*L4) - sqcp*(2*n0*n3*(L1 - L3) + 2*n1*n4*(L1 - L3) + sqcp*(n2*n3 + n4*n5)*(2*L1 + L3) + np.sqrt(3)*n1*n2*L4 - np.sqrt(3)*n0*n5*L4))/12.], \
                [(-(sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + sqcp*(sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n4*L1 - 6*n2*L2 + 4*n4*L3 + np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + n0*(2*n3*L1 + 6*n5*L2 + 4*n3*L3 - np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4)) + sqcm*(2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n0*(-2*n5*L1 + 6*n3*L2 - 4*n5*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4) + n1*(-6*n4*L2 + np.sqrt(3)*n4*L4 + n2*(-2*L1 - 4*L3 + np.sqrt(3)*L4))))/24., \
                (-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4)) + sqcm*(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + 2*n0*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 - np.sqrt(3)*n2*L4 - np.sqrt(3)*n4*L4) + 2*n1*(2*n3*L1 - 6*n5*L2 + 4*n3*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 - np.sqrt(3)*L4) + n1*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 - np.sqrt(3)*n5*L4) + n0*(-6*n4*L2 + np.sqrt(3)*n4*L4 + n2*(2*L1 + 4*L3 - np.sqrt(3)*L4))))/48., \
                np.zeros_like(n0), \
                (-16*np.sqrt(3)*M0 + 16*n1**2*L0 + 8*sqcm**2*n2**2*L0 + 8*sqcp**2*n2**2*L0 + 8*sqcm**2*n3**2*L0 + 8*sqcp**2*n3**2*L0 + 8*sqcm**2*n4**2*L0 + 8*sqcp**2*n4**2*L0 + 8*sqcm**2*n5**2*L0 + 8*sqcp**2*n5**2*L0 - 8*n1**2*L1 + 2*sqcm**2*n2**2*L1 + 2*sqcp**2*n2**2*L1 + 2*sqcm**2*n3**2*L1 + 2*sqcp**2*n3**2*L1 + 2*sqcm**2*n4**2*L1 + 2*sqcp**2*n4**2*L1 + 2*sqcm**2*n5**2*L1 + 2*sqcp**2*n5**2*L1 + 12*(sqcm**2*(n2*n4 - n3*n5) + sqcp**2*(-(n2*n4) + n3*n5) + sqcm*sqcp*(n2**2 + n3**2 - n4**2 - n5**2))*L2 - 4*np.sqrt(3)*n1**2*L4 + np.sqrt(3)*sqcm**2*n2**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 + np.sqrt(3)*sqcp**2*n2**2*L4 + np.sqrt(3)*sqcm**2*n3**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 + np.sqrt(3)*sqcp**2*n3**2*L4 + 2*np.sqrt(3)*sqcm**2*n2*n4*L4 - 2*np.sqrt(3)*sqcp**2*n2*n4*L4 + np.sqrt(3)*sqcm**2*n4**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 + np.sqrt(3)*sqcp**2*n4**2*L4 - 2*np.sqrt(3)*sqcm**2*n3*n5*L4 + 2*np.sqrt(3)*sqcp**2*n3*n5*L4 + np.sqrt(3)*sqcm**2*n5**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4 + np.sqrt(3)*sqcp**2*n5**2*L4 + 4*n0**2*(4*L0 - 2*L1 - np.sqrt(3)*L4))/48. + thcorr, \
                (-(sqcm**2*(n2*n3 + n4*n5)*(2*L1 + L3)) + sqcp*(2*n0*n3*(L1 - L3) + 2*n1*n4*(L1 - L3) + sqcp*(n2*n3 + n4*n5)*(2*L1 + L3) + np.sqrt(3)*n1*n2*L4 - np.sqrt(3)*n0*n5*L4) + sqcm*(2*n1*n2*(-L1 + L3) + 2*sqcp*(n3*n4 - n2*n5)*(2*L1 + L3) + np.sqrt(3)*n1*n4*L4 - n0*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4)))/12., \
                (sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3) + 2*sqcp*(2*sqcm*(n2*n4 + n3*n5)*(2*L1 + L3) + n1*(-2*n5*L1 + 2*n5*L3 + np.sqrt(3)*n3*L4) - n0*(2*n2*(L1 - L3) + np.sqrt(3)*n4*L4)) + sqcm*(-(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3)) + 2*n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - 2*n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4)))/24.], \
                [(-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*(2*n2*L1 + 6*n4*L2 + 4*n2*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4) + n1*(2*n5*L1 - 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + sqcm*(-2*n0*n2*(6*L2 + np.sqrt(3)*L4) + sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + 2*n0*n4*(2*L1 + 4*L3 + np.sqrt(3)*L4) + 2*n1*(2*n3*L1 + 6*n5*L2 + 4*n3*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/48., \
                (-(sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4)) + sqcp*(-(n0*n5*(6*L2 + np.sqrt(3)*L4)) + sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*n3*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n1*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4)) - sqcm*(-(n1*n4*(6*L2 + np.sqrt(3)*L4)) - 2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n1*n2*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n0*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/24., \
                (sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3) + 2*sqcp*(2*sqcm*(n2*n4 + n3*n5)*(2*L1 + L3) + n1*(-2*n5*L1 + 2*n5*L3 + np.sqrt(3)*n3*L4) - n0*(2*n2*(L1 - L3) + np.sqrt(3)*n4*L4)) + sqcm*(-(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3)) + 2*n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - 2*n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4)))/24., \
                (-(sqcm**2*(n2*n3 + n4*n5)*(2*L1 + L3)) + sqcp*(2*n0*n3*(L1 - L3) + 2*n1*n4*(L1 - L3) + sqcp*(n2*n3 + n4*n5)*(2*L1 + L3) + np.sqrt(3)*n1*n2*L4 - np.sqrt(3)*n0*n5*L4) + sqcm*(2*n1*n2*(-L1 + L3) + 2*sqcp*(n3*n4 - n2*n5)*(2*L1 + L3) + np.sqrt(3)*n1*n4*L4 - n0*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4)))/12., \
                (-16*np.sqrt(3)*M0 + 16*n1**2*L0 + 8*sqcm**2*n2**2*L0 + 8*sqcp**2*n2**2*L0 + 8*sqcm**2*n3**2*L0 + 8*sqcp**2*n3**2*L0 + 8*sqcm**2*n4**2*L0 + 8*sqcp**2*n4**2*L0 + 8*sqcm**2*n5**2*L0 + 8*sqcp**2*n5**2*L0 - 8*n1**2*L1 + 2*sqcm**2*n2**2*L1 + 2*sqcp**2*n2**2*L1 + 2*sqcm**2*n3**2*L1 + 2*sqcp**2*n3**2*L1 + 2*sqcm**2*n4**2*L1 + 2*sqcp**2*n4**2*L1 + 2*sqcm**2*n5**2*L1 + 2*sqcp**2*n5**2*L1 - 12*(sqcm**2*(n2*n4 - n3*n5) + sqcp**2*(-(n2*n4) + n3*n5) + sqcm*sqcp*(n2**2 + n3**2 - n4**2 - n5**2))*L2 + 4*np.sqrt(3)*n1**2*L4 - np.sqrt(3)*sqcm**2*n2**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 - np.sqrt(3)*sqcp**2*n2**2*L4 - np.sqrt(3)*sqcm**2*n3**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 - np.sqrt(3)*sqcp**2*n3**2*L4 + 2*np.sqrt(3)*sqcm**2*n2*n4*L4 - 2*np.sqrt(3)*sqcp**2*n2*n4*L4 - np.sqrt(3)*sqcm**2*n4**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 - np.sqrt(3)*sqcp**2*n4**2*L4 - 2*np.sqrt(3)*sqcm**2*n3*n5*L4 + 2*np.sqrt(3)*sqcp**2*n3*n5*L4 - np.sqrt(3)*sqcm**2*n5**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4 - np.sqrt(3)*sqcp**2*n5**2*L4 + 4*n0**2*(4*L0 - 2*L1 + np.sqrt(3)*L4))/48. + thcorr, \
                np.zeros_like(n0)], \
                [(sqcm**2*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) - sqcp*(-(n0*n5*(6*L2 + np.sqrt(3)*L4)) + sqcp*(n2*n3 + n4*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*n3*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n1*(2*n4*L1 + 6*n2*L2 + 4*n4*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4)) + sqcm*(-(n1*n4*(6*L2 + np.sqrt(3)*L4)) - 2*sqcp*(n3*n4 - n2*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n1*n2*(2*L1 + 4*L3 + np.sqrt(3)*L4) + n0*(2*n5*L1 + 6*n3*L2 + 4*n5*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/24., \
                (-(sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4)) + 2*sqcp*(-2*sqcm*(n2*n4 + n3*n5)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + n0*(2*n2*L1 + 6*n4*L2 + 4*n2*L3 + np.sqrt(3)*n2*L4 + np.sqrt(3)*n4*L4) + n1*(2*n5*L1 - 6*n3*L2 + 4*n5*L3 - np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)) + sqcm*(-2*n0*n2*(6*L2 + np.sqrt(3)*L4) + sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 - 2*L3 + np.sqrt(3)*L4) + 2*n0*n4*(2*L1 + 4*L3 + np.sqrt(3)*L4) + 2*n1*(2*n3*L1 + 6*n5*L2 + 4*n3*L3 + np.sqrt(3)*n3*L4 + np.sqrt(3)*n5*L4)))/48., \
                (sqcm**2*(n2*n3 + n4*n5)*(2*L1 + L3) + sqcm*(2*n1*n2*(L1 - L3) + 2*n0*n5*(L1 - L3) - 2*sqcp*(n3*n4 - n2*n5)*(2*L1 + L3) + np.sqrt(3)*n0*n3*L4 - np.sqrt(3)*n1*n4*L4) - sqcp*(2*n0*n3*(L1 - L3) + 2*n1*n4*(L1 - L3) + sqcp*(n2*n3 + n4*n5)*(2*L1 + L3) + np.sqrt(3)*n1*n2*L4 - np.sqrt(3)*n0*n5*L4))/12., \
                (sqcp**2*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3) + 2*sqcp*(2*sqcm*(n2*n4 + n3*n5)*(2*L1 + L3) + n1*(-2*n5*L1 + 2*n5*L3 + np.sqrt(3)*n3*L4) - n0*(2*n2*(L1 - L3) + np.sqrt(3)*n4*L4)) + sqcm*(-(sqcm*(n2**2 - n3**2 - n4**2 + n5**2)*(2*L1 + L3)) + 2*n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - 2*n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4)))/24., \
                np.zeros_like(n0), \
                (-16*np.sqrt(3)*M0 + 16*n1**2*L0 + 8*sqcm**2*n2**2*L0 + 8*sqcp**2*n2**2*L0 + 8*sqcm**2*n3**2*L0 + 8*sqcp**2*n3**2*L0 + 8*sqcm**2*n4**2*L0 + 8*sqcp**2*n4**2*L0 + 8*sqcm**2*n5**2*L0 + 8*sqcp**2*n5**2*L0 - 8*n1**2*L1 + 2*sqcm**2*n2**2*L1 + 2*sqcp**2*n2**2*L1 + 2*sqcm**2*n3**2*L1 + 2*sqcp**2*n3**2*L1 + 2*sqcm**2*n4**2*L1 + 2*sqcp**2*n4**2*L1 + 2*sqcm**2*n5**2*L1 + 2*sqcp**2*n5**2*L1 - 12*(sqcm**2*(n2*n4 - n3*n5) + sqcp**2*(-(n2*n4) + n3*n5) + sqcm*sqcp*(n2**2 + n3**2 - n4**2 - n5**2))*L2 + 4*np.sqrt(3)*n1**2*L4 - np.sqrt(3)*sqcm**2*n2**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 - np.sqrt(3)*sqcp**2*n2**2*L4 - np.sqrt(3)*sqcm**2*n3**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 - np.sqrt(3)*sqcp**2*n3**2*L4 + 2*np.sqrt(3)*sqcm**2*n2*n4*L4 - 2*np.sqrt(3)*sqcp**2*n2*n4*L4 - np.sqrt(3)*sqcm**2*n4**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 - np.sqrt(3)*sqcp**2*n4**2*L4 - 2*np.sqrt(3)*sqcm**2*n3*n5*L4 + 2*np.sqrt(3)*sqcp**2*n3*n5*L4 - np.sqrt(3)*sqcm**2*n5**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4 - np.sqrt(3)*sqcp**2*n5**2*L4 + 4*n0**2*(4*L0 - 2*L1 + np.sqrt(3)*L4))/48. + thcorr]])
        # N: neutral component fields (H1, Eta1, H2, Eta2, H3, Eta3) mass matrix
        nm = np.array([[(-4*np.sqrt(3)*M0 + 4*n1**2*L0 + 2*sqcm**2*n2**2*L0 + 2*sqcp**2*n2**2*L0 + 2*sqcm**2*n3**2*L0 + 2*sqcp**2*n3**2*L0 + 2*sqcm**2*n4**2*L0 + 2*sqcp**2*n4**2*L0 + 2*sqcm**2*n5**2*L0 + 2*sqcp**2*n5**2*L0 + 4*n1**2*L1 - sqcm**2*n2**2*L1 - sqcm**2*n3**2*L1 + 2*sqcm*sqcp*n2*n4*L1 - sqcp**2*n4**2*L1 - 2*sqcm*sqcp*n3*n5*L1 - sqcp**2*n5**2*L1 + 12*n0**2*(L0 + L1) + 3*sqcm**2*n2**2*L2 + 3*sqcm**2*n3**2*L2 - 6*sqcm*sqcp*n2*n4*L2 + 3*sqcp**2*n4**2*L2 + 6*sqcm*sqcp*n3*n5*L2 + 3*sqcp**2*n5**2*L2 + 2*(sqcp**2*(n2**2 + n3**2) + 2*sqcm*sqcp*(n2*n4 - n3*n5) + sqcm**2*(n4**2 + n5**2))*L3 - 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 - 2*np.sqrt(3)*sqcm**2*n2*n4*L4 + 2*np.sqrt(3)*sqcp**2*n2*n4*L4 + 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 + 2*np.sqrt(3)*sqcm**2*n3*n5*L4 - 2*np.sqrt(3)*sqcp**2*n3*n5*L4 + 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4)/12. + thcorr, \
                (8*n0*n1*(L0 + L1) + (sqcm**2 + sqcp**2)*(n3*n4 + n2*n5)*(L1 - 3*L2 + 2*L3))/12., \
                (sqcm**3*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) - 3*np.sqrt(3)*n2**2*L4 + 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(n4 - n5)*(n4 + n5)*L4) + 2*sqcp**2*(2*n1*n5*(L1 - 3*L2 + 2*L3) + 4*n0*(2*n2*(L0 + L3) + np.sqrt(3)*n4*L4) - sqcp*(3*n2**2*(L1 - L3) - (n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n3**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4)) + 2*sqcm**2*(2*n1*n5*(L1 - 3*L2 + 2*L3) + 4*n0*(n2*(2*L0 - L1 + 3*L2) - np.sqrt(3)*n4*L4) + sqcp*(3*n2**2*(L1 - L3) - 5*(n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n3**2*(-L1 + L3) + 5*np.sqrt(3)*n2*n4*L4 + 5*np.sqrt(3)*n3*n5*L4)) + sqcm*sqcp*(8*n0*(n4*(L1 - 3*L2 + 2*L3) - 2*np.sqrt(3)*n2*L4) + sqcp*(20*n2*n4*(-L1 + L3) + 20*n3*n5*(-L1 + L3) + 3*np.sqrt(3)*n2**2*L4 - 3*np.sqrt(3)*n3**2*L4 + 5*np.sqrt(3)*(-n4**2 + n5**2)*L4)))/48., \
                (sqcm**3*(2*n2*n5*(L1 - L3) + 2*n3*n4*(-L1 + L3) + 3*np.sqrt(3)*n2*n3*L4 + np.sqrt(3)*n4*n5*L4) + sqcm**2*(2*n1*n4*(L1 - 3*L2 + 2*L3) - 5*sqcp*n4*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4) + 4*n0*(n3*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n5*L4) + sqcp*n2*(6*n3*(-L1 + L3) + 5*np.sqrt(3)*n5*L4)) + sqcp**2*(8*n0*n3*(L0 + L3) + 2*n1*n4*(L1 - 3*L2 + 2*L3) - 4*np.sqrt(3)*n0*n5*L4 + sqcp*(6*n2*n3*(L1 - L3) + 2*n4*n5*(L1 - L3) + np.sqrt(3)*n3*n4*L4 - np.sqrt(3)*n2*n5*L4)) + sqcm*sqcp*(-4*n0*(n5*(L1 - 3*L2 + 2*L3) + 2*np.sqrt(3)*n3*L4) + sqcp*(10*n3*n4*(L1 - L3) - 3*np.sqrt(3)*n2*n3*L4 - 5*n5*(2*n2*(L1 - L3) + np.sqrt(3)*n4*L4))))/24., \
                (2*sqcm**3*(n2**2*(L1 - L3) - 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + n3**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4) - 2*sqcm*sqcp*(-4*n0*(n2*(L1 - 3*L2 + 2*L3) + 2*np.sqrt(3)*n4*L4) + sqcp*(5*n2**2*(L1 - L3) - 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + 5*n3**2*(-L1 + L3) + 5*np.sqrt(3)*n2*n4*L4 + 5*np.sqrt(3)*n3*n5*L4)) + sqcp**2*(4*n1*n3*(L1 - 3*L2 + 2*L3) + 8*n0*(n4*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n2*L4) + sqcp*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) - np.sqrt(3)*n2**2*L4 + np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(n4 - n5)*(n4 + n5)*L4)) + sqcm**2*(4*n1*n3*(L1 - 3*L2 + 2*L3) + 8*n0*(2*n4*(L0 + L3) - np.sqrt(3)*n2*L4) + sqcp*(20*n2*n4*(-L1 + L3) + 20*n3*n5*(-L1 + L3) + 5*np.sqrt(3)*n2**2*L4 - 5*np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(-n4**2 + n5**2)*L4)))/48., \
                (sqcm**3*(2*n2*n3*(L1 - L3) + 6*n4*n5*(L1 - L3) + np.sqrt(3)*n3*n4*L4 - np.sqrt(3)*n2*n5*L4) + sqcm**2*(10*sqcp*n2*n5*(L1 - L3) + 8*n0*n5*(L0 + L3) + 2*n1*n2*(L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n0*n3*L4 + 3*np.sqrt(3)*sqcp*n4*n5*L4 + 5*sqcp*n3*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4)) + sqcm*sqcp*(-4*n0*n3*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n0*n5*L4 + sqcp*n4*(6*n5*(-L1 + L3) - 5*np.sqrt(3)*n3*L4) + 5*sqcp*n2*(2*n3*(-L1 + L3) + np.sqrt(3)*n5*L4)) - sqcp**2*(-2*n1*n2*(L1 - 3*L2 + 2*L3) + 4*n0*(n5*(-2*L0 + L1 - 3*L2) + np.sqrt(3)*n3*L4) + sqcp*(2*n2*n5*(L1 - L3) + 2*n3*n4*(-L1 + L3) + np.sqrt(3)*n2*n3*L4 + 3*np.sqrt(3)*n4*n5*L4)))/24.], \
                [(8*n0*n1*(L0 + L1) + (sqcm**2 + sqcp**2)*(n3*n4 + n2*n5)*(L1 - 3*L2 + 2*L3))/12., \
                (-4*np.sqrt(3)*M0 + 12*n1**2*L0 + 2*sqcm**2*n2**2*L0 + 2*sqcp**2*n2**2*L0 + 2*sqcm**2*n3**2*L0 + 2*sqcp**2*n3**2*L0 + 2*sqcm**2*n4**2*L0 + 2*sqcp**2*n4**2*L0 + 2*sqcm**2*n5**2*L0 + 2*sqcp**2*n5**2*L0 + 12*n1**2*L1 - sqcp**2*n2**2*L1 - sqcp**2*n3**2*L1 - 2*sqcm*sqcp*n2*n4*L1 - sqcm**2*n4**2*L1 + 2*sqcm*sqcp*n3*n5*L1 - sqcm**2*n5**2*L1 + 4*n0**2*(L0 + L1) + 3*sqcp**2*n2**2*L2 + 3*sqcp**2*n3**2*L2 + 6*sqcm*sqcp*n2*n4*L2 + 3*sqcm**2*n4**2*L2 - 6*sqcm*sqcp*n3*n5*L2 + 3*sqcm**2*n5**2*L2 + 2*(sqcm**2*(n2**2 + n3**2) + 2*sqcm*sqcp*(-(n2*n4) + n3*n5) + sqcp**2*(n4**2 + n5**2))*L3 - 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 - 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 - 2*np.sqrt(3)*sqcm**2*n2*n4*L4 + 2*np.sqrt(3)*sqcp**2*n2*n4*L4 + 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 + 2*np.sqrt(3)*sqcm**2*n3*n5*L4 - 2*np.sqrt(3)*sqcp**2*n3*n5*L4 + 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4)/12. + thcorr, \
                (sqcm**3*(6*n2*n3*(L1 - L3) + 2*n4*n5*(L1 - L3) - np.sqrt(3)*n3*n4*L4 + np.sqrt(3)*n2*n5*L4) - sqcm*sqcp*(4*n1*n4*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n1*n2*L4 + 5*sqcp*n4*(2*n5*(L1 - L3) - np.sqrt(3)*n3*L4) + sqcp*n2*(6*n3*(L1 - L3) + 5*np.sqrt(3)*n5*L4)) + sqcp**2*(2*n0*n5*(L1 - 3*L2 + 2*L3) + 4*n1*(n2*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n4*L4) + sqcp*(2*n3*n4*(L1 - L3) + 2*n2*n5*(-L1 + L3) + 3*np.sqrt(3)*n2*n3*L4 + np.sqrt(3)*n4*n5*L4)) + sqcm**2*(2*n0*n5*(L1 - 3*L2 + 2*L3) + n1*(8*n2*(L0 + L3) - 4*np.sqrt(3)*n4*L4) - sqcp*(10*n3*n4*(L1 - L3) + 3*np.sqrt(3)*n2*n3*L4 + 5*n5*(2*n2*(-L1 + L3) + np.sqrt(3)*n4*L4))))/24., \
                (-2*sqcm**3*(3*n3**2*(L1 - L3) + (n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n2**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4) + sqcm**2*(-20*sqcp*n2*n4*L1 - 20*sqcp*n3*n5*L1 + 4*n0*n4*(L1 - 3*L2) + 8*n0*n4*L3 + 20*sqcp*(n2*n4 + n3*n5)*L3 + 16*n1*n3*(L0 + L3) + 8*np.sqrt(3)*n1*n5*L4 + np.sqrt(3)*sqcp*(-3*n2**2 + 3*n3**2 + 5*(n4 - n5)*(n4 + n5))*L4) - 2*sqcm*sqcp*(-4*n1*n5*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n1*n3*L4 + sqcp*(3*n2**2*(L1 - L3) - 5*(n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n3**2*(-L1 + L3) - 5*np.sqrt(3)*n2*n4*L4 - 5*np.sqrt(3)*n3*n5*L4)) + sqcp**2*(4*n0*n4*(L1 - 3*L2 + 2*L3) + 8*n1*(n3*(2*L0 - L1 + 3*L2) - np.sqrt(3)*n5*L4) + sqcp*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) + 3*np.sqrt(3)*n2**2*L4 - 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(-n4**2 + n5**2)*L4)))/48., \
                (-(sqcm**3*(2*n3*n4*(L1 - L3) + 2*n2*n5*(-L1 + L3) + np.sqrt(3)*n2*n3*L4 + 3*np.sqrt(3)*n4*n5*L4)) + sqcm**2*(2*n0*n3*(L1 - 3*L2 + 2*L3) + 4*n1*(n4*(2*L0 - L1 + 3*L2) - np.sqrt(3)*n2*L4) + sqcp*n4*(6*n5*(-L1 + L3) + 5*np.sqrt(3)*n3*L4) - 5*sqcp*n2*(2*n3*(L1 - L3) + np.sqrt(3)*n5*L4)) + sqcp**2*(8*n1*n4*(L0 + L3) + 2*n0*n3*(L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n1*n2*L4 + sqcp*(2*n2*n3*(L1 - L3) + 6*n4*n5*(L1 - L3) - np.sqrt(3)*n3*n4*L4 + np.sqrt(3)*n2*n5*L4)) + sqcm*sqcp*(-4*n1*n2*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n1*n4*L4 + sqcp*(10*n2*n5*(-L1 + L3) + 3*np.sqrt(3)*n4*n5*L4 + 5*n3*(2*n4*(L1 - L3) + np.sqrt(3)*n2*L4))))/24., \
                (sqcm**3*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) + np.sqrt(3)*n2**2*L4 - np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(-n4**2 + n5**2)*L4) + 2*sqcm**2*(2*n0*n2*(L1 - 3*L2 + 2*L3) + 4*n1*(n5*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n3*L4) + sqcp*(5*n2**2*(L1 - L3) - 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + 5*n3**2*(-L1 + L3) - 5*np.sqrt(3)*n2*n4*L4 - 5*np.sqrt(3)*n3*n5*L4)) + 2*sqcp**2*(2*n0*n2*(L1 - 3*L2 + 2*L3) + n1*(8*n5*(L0 + L3) - 4*np.sqrt(3)*n3*L4) + sqcp*(n3**2*(L1 - L3) + 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + n2**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4)) + sqcm*sqcp*(8*n1*(n3*(L1 - 3*L2 + 2*L3) + 2*np.sqrt(3)*n5*L4) + sqcp*(20*n2*n4*(-L1 + L3) + 20*n3*n5*(-L1 + L3) - 5*np.sqrt(3)*n2**2*L4 + 5*np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(n4 - n5)*(n4 + n5)*L4)))/48.], \
                [(sqcm**3*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) - 3*np.sqrt(3)*n2**2*L4 + 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(n4 - n5)*(n4 + n5)*L4) + 2*sqcp**2*(2*n1*n5*(L1 - 3*L2 + 2*L3) + 4*n0*(2*n2*(L0 + L3) + np.sqrt(3)*n4*L4) - sqcp*(3*n2**2*(L1 - L3) - (n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n3**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4)) + 2*sqcm**2*(2*n1*n5*(L1 - 3*L2 + 2*L3) + 4*n0*(n2*(2*L0 - L1 + 3*L2) - np.sqrt(3)*n4*L4) + sqcp*(3*n2**2*(L1 - L3) - 5*(n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n3**2*(-L1 + L3) + 5*np.sqrt(3)*n2*n4*L4 + 5*np.sqrt(3)*n3*n5*L4)) + sqcm*sqcp*(8*n0*(n4*(L1 - 3*L2 + 2*L3) - 2*np.sqrt(3)*n2*L4) + sqcp*(20*n2*n4*(-L1 + L3) + 20*n3*n5*(-L1 + L3) + 3*np.sqrt(3)*n2**2*L4 - 3*np.sqrt(3)*n3**2*L4 + 5*np.sqrt(3)*(-n4**2 + n5**2)*L4)))/48., \
                (sqcm**3*(6*n2*n3*(L1 - L3) + 2*n4*n5*(L1 - L3) - np.sqrt(3)*n3*n4*L4 + np.sqrt(3)*n2*n5*L4) - sqcm*sqcp*(4*n1*n4*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n1*n2*L4 + 5*sqcp*n4*(2*n5*(L1 - L3) - np.sqrt(3)*n3*L4) + sqcp*n2*(6*n3*(L1 - L3) + 5*np.sqrt(3)*n5*L4)) + sqcp**2*(2*n0*n5*(L1 - 3*L2 + 2*L3) + 4*n1*(n2*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n4*L4) + sqcp*(2*n3*n4*(L1 - L3) + 2*n2*n5*(-L1 + L3) + 3*np.sqrt(3)*n2*n3*L4 + np.sqrt(3)*n4*n5*L4)) + sqcm**2*(2*n0*n5*(L1 - 3*L2 + 2*L3) + n1*(8*n2*(L0 + L3) - 4*np.sqrt(3)*n4*L4) - sqcp*(10*n3*n4*(L1 - L3) + 3*np.sqrt(3)*n2*n3*L4 + 5*n5*(2*n2*(-L1 + L3) + np.sqrt(3)*n4*L4))))/24., \
                (-16*np.sqrt(3)*M0 + 16*sqcm**2*n2**2*L0 + 16*sqcp**2*n2**2*L0 + 16*sqcm**2*sqcp**2*n2**2*L0 + 8*sqcm**2*n3**2*L0 + 8*sqcp**2*n3**2*L0 + 16*sqcm**3*sqcp*n2*n4*L0 - 16*sqcm*sqcp**3*n2*n4*L0 + 16*sqcm**2*n4**2*L0 + 16*sqcp**2*n4**2*L0 - 16*sqcm**2*sqcp**2*n4**2*L0 + 8*sqcm**2*n5**2*L0 + 8*sqcp**2*n5**2*L0 + 8*sqcm**2*n2**2*L1 + 8*sqcp**2*n2**2*L1 - 4*sqcm**2*sqcp**2*n2**2*L1 + 6*sqcm**2*n3**2*L1 + 6*sqcp**2*n3**2*L1 - 8*sqcm**2*sqcp**2*n3**2*L1 - 4*sqcm**3*sqcp*n2*n4*L1 + 4*sqcm*sqcp**3*n2*n4*L1 + 8*sqcm**2*n4**2*L1 + 8*sqcp**2*n4**2*L1 + 4*sqcm**2*sqcp**2*n4**2*L1 + 8*sqcm**3*sqcp*n3*n5*L1 - 8*sqcm*sqcp**3*n3*n5*L1 + 6*sqcm**2*n5**2*L1 + 6*sqcp**2*n5**2*L1 + 8*sqcm**2*sqcp**2*n5**2*L1 + 6*sqcm**2*n2**2*L2 + 6*sqcp**2*n2**2*L2 + 24*sqcm**2*sqcp**2*n2**2*L2 + 12*sqcm**2*sqcp**2*n3**2*L2 + 24*sqcm**3*sqcp*n2*n4*L2 - 24*sqcm*sqcp**3*n2*n4*L2 + 6*sqcm**2*n4**2*L2 + 6*sqcp**2*n4**2*L2 - 24*sqcm**2*sqcp**2*n4**2*L2 - 12*sqcm**3*sqcp*n3*n5*L2 + 12*sqcm*sqcp**3*n3*n5*L2 - 12*sqcm**2*sqcp**2*n5**2*L2 + 2*(2*sqcm*sqcp**3*(n2*n4 - n3*n5) + 2*sqcm**3*sqcp*(-(n2*n4) + n3*n5) + sqcp**2*(n2**2 + n3**2 + n4**2 + n5**2) + sqcm**2*((1 - 2*sqcp**2)*n2**2 + n3**2 + n4**2 + n5**2 + 2*sqcp**2*(-n3**2 + n4**2 + n5**2)))*L3 + 6*np.sqrt(3)*sqcm*sqcp*n2**2*L4 + 3*np.sqrt(3)*sqcm**3*sqcp*n2**2*L4 + 3*np.sqrt(3)*sqcm*sqcp**3*n2**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 + np.sqrt(3)*sqcm**3*sqcp*n3**2*L4 + np.sqrt(3)*sqcm*sqcp**3*n3**2*L4 + 6*np.sqrt(3)*sqcm**2*n2*n4*L4 - 6*np.sqrt(3)*sqcp**2*n2*n4*L4 - 6*np.sqrt(3)*sqcm*sqcp*n4**2*L4 + 3*np.sqrt(3)*sqcm**3*sqcp*n4**2*L4 + 3*np.sqrt(3)*sqcm*sqcp**3*n4**2*L4 - 2*np.sqrt(3)*sqcm**2*n3*n5*L4 + 2*np.sqrt(3)*sqcp**2*n3*n5*L4 - 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4 + np.sqrt(3)*sqcm**3*sqcp*n5**2*L4 + np.sqrt(3)*sqcm*sqcp**3*n5**2*L4 + 4*n0**2*(4*L0 - L1 + 3*L2 + 2*L3 - 2*np.sqrt(3)*sqcm*sqcp*L4) + 4*n1**2*(4*L0 - L1 + 3*L2 + 2*L3 - 2*np.sqrt(3)*sqcm*sqcp*L4) + 4*n0*(-(sqcm*(2*n4*(L1 - L3) + np.sqrt(3)*n2*L4)) + sqcm*sqcp**2*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) + sqcp*(2*(-1 + sqcm**2)*n2*(L1 - L3) + np.sqrt(3)*(1 + sqcm**2)*n4*L4)) - 4*n1*(-(sqcp*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4)) + sqcm**2*sqcp*(2*n5*(-L1 + L3) + np.sqrt(3)*n3*L4) + sqcm*(2*(-1 + sqcp**2)*n3*(L1 - L3) + np.sqrt(3)*(1 + sqcp**2)*n5*L4)) - 2*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*sqcp*(2*n0*n2*L1 + 2*n1*n5*L1 - 2*(n0*n2 + n1*n5)*L3 - np.sqrt(3)*n1*n3*L4 + np.sqrt(3)*n0*n4*L4) - 4*sqcm*(-2*n0*n4*L1 + 2*n0*n4*L3 + 2*n1*n3*(-L1 + L3) + sqcp*n2*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) + np.sqrt(3)*n0*n2*L4 - np.sqrt(3)*n1*n5*L4))*self.cosa)/48. + thcorr, \
                (sqcm**2*((1 + 2*sqcp**2)*n2*n3*(4*L0 + L1 + 3*L2) + (-1 + 2*sqcp**2)*n4*n5*(4*L0 + L1 + 3*L2) + 4*sqcp*n0*n3*(-L1 + L3) + 4*sqcp*n1*n4*(-L1 + L3) + np.sqrt(3)*n3*n4*L4 + 2*np.sqrt(3)*sqcp*n0*n5*L4 - np.sqrt(3)*n2*(2*sqcp*n1 + n5)*L4) + sqcm**3*sqcp*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) + sqcp*(2*n1*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) - sqcp*n4*(n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n3*L4) + sqcp*n2*(n3*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n5*L4) + 2*n0*(2*n3*(L1 - L3) + np.sqrt(3)*n5*L4)) + sqcm*(2*np.sqrt(3)*sqcp*(n2*n3 + n4*n5)*L4 + 2*(2*n1*n2*(L1 - L3) + 2*n0*n5*(-L1 + L3) + np.sqrt(3)*n0*n3*L4 + np.sqrt(3)*n1*n4*L4) + sqcp**2*(4*n1*n2*(-L1 + L3) + 4*n0*n5*(-L1 + L3) - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4) + sqcp**3*(-(n3*n4*(4*L0 + L1 + 3*L2)) + n2*n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*n3*L4 - np.sqrt(3)*n4*n5*L4)) + (-(sqcm**2*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3)) + 2*sqcm*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 + sqcp*(n3*n4 - n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4) + sqcp*(8*n0*n3*L1 + 8*n1*n4*L1 - 8*n0*n3*L3 - 8*n1*n4*L3 + sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n1*n2*L4 - 4*np.sqrt(3)*n0*n5*L4))*self.cosa)/24., \
                (-2*sqcm*sqcp*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*sqcp*(2*n0*n2*L1 + 2*n1*n5*L1 - 2*(n0*n2 + n1*n5)*L3 - np.sqrt(3)*n1*n3*L4 + np.sqrt(3)*n0*n4*L4) - 4*sqcm*(-2*n0*n4*L1 + 2*n0*n4*L3 + 2*n1*n3*(-L1 + L3) + sqcp*n2*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) + np.sqrt(3)*n0*n2*L4 - np.sqrt(3)*n1*n5*L4)) - (-8*np.sqrt(3)*(n0**2 + n1**2)*L4 + 4*sqcm*(2*n0*n2*L1 + 2*n1*n5*L1 - 2*(n0*n2 + n1*n5)*L3 + sqcp*(n2**2*(4*L0 - L1 + 6*L2 - L3) + n4**2*(-4*L0 + L1 - 6*L2 + L3) - (n3 - n5)*(n3 + n5)*(2*L1 - 3*L2 + L3)) - np.sqrt(3)*n1*n3*L4 + np.sqrt(3)*n0*n4*L4) + sqcp*(8*n1*n3*(-L1 + L3) + 8*n0*n4*(-L1 + L3) + 4*np.sqrt(3)*n0*n2*L4 - 4*np.sqrt(3)*n1*n5*L4) + sqcp**2*(4*n2*n4*(-4*L0 + L1 - 6*L2 + L3) - 4*n3*n5*(2*L1 - 3*L2 + L3) + 3*np.sqrt(3)*n2**2*L4 + np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(3*n4**2 + n5**2)*L4) + sqcm**2*(4*n2*n4*(4*L0 - L1 + 6*L2 - L3) + 4*n3*n5*(2*L1 - 3*L2 + L3) + 3*np.sqrt(3)*n2**2*L4 + np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(3*n4**2 + n5**2)*L4))*self.cosa)/48., \
                (4*n0*n1*(L1 - 3*L2 + 2*L3) + sqcp**2*(n3*n4 + n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcm**3*sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcm**2*(8*sqcp*n1*n2*(L1 - L3) + 8*sqcp*n0*n5*(L1 - L3) - (-1 + 2*sqcp**2)*n3*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) + (1 + 2*sqcp**2)*n2*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*sqcp*n0*n3*L4 - 4*np.sqrt(3)*sqcp*n1*n4*L4) - sqcm*sqcp**2*(sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*n1*(2*n4*L1 - 2*n4*L3 + np.sqrt(3)*n2*L4) + n0*(8*n3*L1 - 8*n3*L3 - 4*np.sqrt(3)*n5*L4)) + (2*sqcm*(-2*n0*n3*L1 - 2*n1*n4*L1 + sqcp*(n2*n3 + n4*n5)*(4*L0 + L1 + 3*L2) + 2*n0*n3*L3 + 2*n1*n4*L3 - np.sqrt(3)*n1*n2*L4 + np.sqrt(3)*n0*n5*L4) + sqcm**2*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) + sqcp*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4 + sqcp*(-(n3*n4*(4*L0 + L1 + 3*L2)) + n2*n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*n3*L4 - np.sqrt(3)*n4*n5*L4)))*self.cosa)/24.], \
                [(sqcm**3*(2*n2*n5*(L1 - L3) + 2*n3*n4*(-L1 + L3) + 3*np.sqrt(3)*n2*n3*L4 + np.sqrt(3)*n4*n5*L4) + sqcm**2*(2*n1*n4*(L1 - 3*L2 + 2*L3) - 5*sqcp*n4*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4) + 4*n0*(n3*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n5*L4) + sqcp*n2*(6*n3*(-L1 + L3) + 5*np.sqrt(3)*n5*L4)) + sqcp**2*(8*n0*n3*(L0 + L3) + 2*n1*n4*(L1 - 3*L2 + 2*L3) - 4*np.sqrt(3)*n0*n5*L4 + sqcp*(6*n2*n3*(L1 - L3) + 2*n4*n5*(L1 - L3) + np.sqrt(3)*n3*n4*L4 - np.sqrt(3)*n2*n5*L4)) + sqcm*sqcp*(-4*n0*(n5*(L1 - 3*L2 + 2*L3) + 2*np.sqrt(3)*n3*L4) + sqcp*(10*n3*n4*(L1 - L3) - 3*np.sqrt(3)*n2*n3*L4 - 5*n5*(2*n2*(L1 - L3) + np.sqrt(3)*n4*L4))))/24., \
                (-2*sqcm**3*(3*n3**2*(L1 - L3) + (n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n2**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4) + sqcm**2*(-20*sqcp*n2*n4*L1 - 20*sqcp*n3*n5*L1 + 4*n0*n4*(L1 - 3*L2) + 8*n0*n4*L3 + 20*sqcp*(n2*n4 + n3*n5)*L3 + 16*n1*n3*(L0 + L3) + 8*np.sqrt(3)*n1*n5*L4 + np.sqrt(3)*sqcp*(-3*n2**2 + 3*n3**2 + 5*(n4 - n5)*(n4 + n5))*L4) - 2*sqcm*sqcp*(-4*n1*n5*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n1*n3*L4 + sqcp*(3*n2**2*(L1 - L3) - 5*(n4 - n5)*(n4 + n5)*(L1 - L3) + 3*n3**2*(-L1 + L3) - 5*np.sqrt(3)*n2*n4*L4 - 5*np.sqrt(3)*n3*n5*L4)) + sqcp**2*(4*n0*n4*(L1 - 3*L2 + 2*L3) + 8*n1*(n3*(2*L0 - L1 + 3*L2) - np.sqrt(3)*n5*L4) + sqcp*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) + 3*np.sqrt(3)*n2**2*L4 - 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(-n4**2 + n5**2)*L4)))/48., \
                (sqcm**2*((1 + 2*sqcp**2)*n2*n3*(4*L0 + L1 + 3*L2) + (-1 + 2*sqcp**2)*n4*n5*(4*L0 + L1 + 3*L2) + 4*sqcp*n0*n3*(-L1 + L3) + 4*sqcp*n1*n4*(-L1 + L3) + np.sqrt(3)*n3*n4*L4 + 2*np.sqrt(3)*sqcp*n0*n5*L4 - np.sqrt(3)*n2*(2*sqcp*n1 + n5)*L4) + sqcm**3*sqcp*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) + sqcp*(2*n1*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) - sqcp*n4*(n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n3*L4) + sqcp*n2*(n3*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n5*L4) + 2*n0*(2*n3*(L1 - L3) + np.sqrt(3)*n5*L4)) + sqcm*(2*np.sqrt(3)*sqcp*(n2*n3 + n4*n5)*L4 + 2*(2*n1*n2*(L1 - L3) + 2*n0*n5*(-L1 + L3) + np.sqrt(3)*n0*n3*L4 + np.sqrt(3)*n1*n4*L4) + sqcp**2*(4*n1*n2*(-L1 + L3) + 4*n0*n5*(-L1 + L3) - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4) + sqcp**3*(-(n3*n4*(4*L0 + L1 + 3*L2)) + n2*n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*n3*L4 - np.sqrt(3)*n4*n5*L4)) + (-(sqcm**2*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3)) + 2*sqcm*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 + sqcp*(n3*n4 - n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4) + sqcp*(8*n0*n3*L1 + 8*n1*n4*L1 - 8*n0*n3*L3 - 8*n1*n4*L3 + sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n1*n2*L4 - 4*np.sqrt(3)*n0*n5*L4))*self.cosa)/24., \
                (-16*np.sqrt(3)*M0 + 8*sqcm**2*n2**2*L0 + 8*sqcp**2*n2**2*L0 + 16*sqcm**2*n3**2*L0 + 16*sqcp**2*n3**2*L0 + 16*sqcm**2*sqcp**2*n3**2*L0 + 8*sqcm**2*n4**2*L0 + 8*sqcp**2*n4**2*L0 - 16*sqcm**3*sqcp*n3*n5*L0 + 16*sqcm*sqcp**3*n3*n5*L0 + 16*sqcm**2*n5**2*L0 + 16*sqcp**2*n5**2*L0 - 16*sqcm**2*sqcp**2*n5**2*L0 + 6*sqcm**2*n2**2*L1 + 6*sqcp**2*n2**2*L1 - 8*sqcm**2*sqcp**2*n2**2*L1 + 8*sqcm**2*n3**2*L1 + 8*sqcp**2*n3**2*L1 - 4*sqcm**2*sqcp**2*n3**2*L1 - 8*sqcm**3*sqcp*n2*n4*L1 + 8*sqcm*sqcp**3*n2*n4*L1 + 6*sqcm**2*n4**2*L1 + 6*sqcp**2*n4**2*L1 + 8*sqcm**2*sqcp**2*n4**2*L1 + 4*sqcm**3*sqcp*n3*n5*L1 - 4*sqcm*sqcp**3*n3*n5*L1 + 8*sqcm**2*n5**2*L1 + 8*sqcp**2*n5**2*L1 + 4*sqcm**2*sqcp**2*n5**2*L1 + 12*sqcm**2*sqcp**2*n2**2*L2 + 6*sqcm**2*n3**2*L2 + 6*sqcp**2*n3**2*L2 + 24*sqcm**2*sqcp**2*n3**2*L2 + 12*sqcm**3*sqcp*n2*n4*L2 - 12*sqcm*sqcp**3*n2*n4*L2 - 12*sqcm**2*sqcp**2*n4**2*L2 - 24*sqcm**3*sqcp*n3*n5*L2 + 24*sqcm*sqcp**3*n3*n5*L2 + 6*sqcm**2*n5**2*L2 + 6*sqcp**2*n5**2*L2 - 24*sqcm**2*sqcp**2*n5**2*L2 + 2*(2*sqcm*sqcp**3*(n2*n4 - n3*n5) + 2*sqcm**3*sqcp*(-(n2*n4) + n3*n5) + sqcp**2*(n2**2 + n3**2 + n4**2 + n5**2) + sqcm**2*((1 - 2*sqcp**2)*n2**2 + n3**2 + n4**2 + n5**2 + 2*sqcp**2*(-n3**2 + n4**2 + n5**2)))*L3 + 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 + np.sqrt(3)*sqcm**3*sqcp*n2**2*L4 + np.sqrt(3)*sqcm*sqcp**3*n2**2*L4 + 6*np.sqrt(3)*sqcm*sqcp*n3**2*L4 + 3*np.sqrt(3)*sqcm**3*sqcp*n3**2*L4 + 3*np.sqrt(3)*sqcm*sqcp**3*n3**2*L4 + 2*np.sqrt(3)*sqcm**2*n2*n4*L4 - 2*np.sqrt(3)*sqcp**2*n2*n4*L4 - 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 + np.sqrt(3)*sqcm**3*sqcp*n4**2*L4 + np.sqrt(3)*sqcm*sqcp**3*n4**2*L4 - 6*np.sqrt(3)*sqcm**2*n3*n5*L4 + 6*np.sqrt(3)*sqcp**2*n3*n5*L4 - 6*np.sqrt(3)*sqcm*sqcp*n5**2*L4 + 3*np.sqrt(3)*sqcm**3*sqcp*n5**2*L4 + 3*np.sqrt(3)*sqcm*sqcp**3*n5**2*L4 + 4*n0**2*(4*L0 - L1 + 3*L2 + 2*L3 - 2*np.sqrt(3)*sqcm*sqcp*L4) + 4*n1**2*(4*L0 - L1 + 3*L2 + 2*L3 - 2*np.sqrt(3)*sqcm*sqcp*L4) - 4*n0*(-(sqcm*(2*n4*(L1 - L3) + np.sqrt(3)*n2*L4)) + sqcm*sqcp**2*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) + sqcp*(2*(-1 + sqcm**2)*n2*(L1 - L3) + np.sqrt(3)*(1 + sqcm**2)*n4*L4)) + 4*n1*(-(sqcp*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4)) + sqcm**2*sqcp*(2*n5*(-L1 + L3) + np.sqrt(3)*n3*L4) + sqcm*(2*(-1 + sqcp**2)*n3*(L1 - L3) + np.sqrt(3)*(1 + sqcp**2)*n5*L4)) - 2*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcp*(8*n0*n2*(-L1 + L3) + 8*n1*n5*(-L1 + L3) + 4*np.sqrt(3)*n1*n3*L4 - 4*np.sqrt(3)*n0*n4*L4) + 4*sqcm*(sqcp*n3*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) + n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4)))*self.cosa)/48. + thcorr, \
                (4*n0*n1*(L1 - 3*L2 + 2*L3) + sqcp**2*(n3*n4 + n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcm**3*sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcm**2*(8*sqcp*n1*n2*(-L1 + L3) + 8*sqcp*n0*n5*(-L1 + L3) + (1 + 2*sqcp**2)*n3*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) - (-1 + 2*sqcp**2)*n2*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) - 4*np.sqrt(3)*sqcp*n0*n3*L4 + 4*np.sqrt(3)*sqcp*n1*n4*L4) + sqcm*sqcp**2*(sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*n1*(2*n4*L1 - 2*n4*L3 + np.sqrt(3)*n2*L4) + n0*(8*n3*L1 - 8*n3*L3 - 4*np.sqrt(3)*n5*L4)) - (2*sqcm*(-2*n0*n3*L1 - 2*n1*n4*L1 + sqcp*(n2*n3 + n4*n5)*(4*L0 + L1 + 3*L2) + 2*n0*n3*L3 + 2*n1*n4*L3 - np.sqrt(3)*n1*n2*L4 + np.sqrt(3)*n0*n5*L4) + sqcm**2*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) + sqcp*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4 + sqcp*(-(n3*n4*(4*L0 + L1 + 3*L2)) + n2*n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*n3*L4 - np.sqrt(3)*n4*n5*L4)))*self.cosa)/24., \
                (2*sqcm*sqcp*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcp*(8*n0*n2*(-L1 + L3) + 8*n1*n5*(-L1 + L3) + 4*np.sqrt(3)*n1*n3*L4 - 4*np.sqrt(3)*n0*n4*L4) + 4*sqcm*(sqcp*n3*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) + n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4))) + (-8*np.sqrt(3)*(n0**2 + n1**2)*L4 + 4*sqcm*(-2*n0*n2*L1 - 2*n1*n5*L1 + 2*n0*n2*L3 + 2*n1*n5*L3 + sqcp*(n3**2*(4*L0 - L1 + 6*L2 - L3) + n5**2*(-4*L0 + L1 - 6*L2 + L3) - (n2 - n4)*(n2 + n4)*(2*L1 - 3*L2 + L3)) + np.sqrt(3)*n1*n3*L4 - np.sqrt(3)*n0*n4*L4) + 4*sqcp*(2*n1*n3*L1 + 2*n0*n4*L1 - 2*(n1*n3 + n0*n4)*L3 - np.sqrt(3)*n0*n2*L4 + np.sqrt(3)*n1*n5*L4) + sqcm**2*(4*n3*n5*(-4*L0 + L1 - 6*L2 + L3) + 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(n4**2 + 3*n5**2)*L4 + n2*(-4*n4*(2*L1 - 3*L2 + L3) + np.sqrt(3)*n2*L4)) + sqcp**2*(4*n3*n5*(4*L0 - L1 + 6*L2 - L3) + 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(n4**2 + 3*n5**2)*L4 + n2*(4*n4*(2*L1 - 3*L2 + L3) + np.sqrt(3)*n2*L4)))*self.cosa)/48.], \
                [(2*sqcm**3*(n2**2*(L1 - L3) - 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + n3**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4) - 2*sqcm*sqcp*(-4*n0*(n2*(L1 - 3*L2 + 2*L3) + 2*np.sqrt(3)*n4*L4) + sqcp*(5*n2**2*(L1 - L3) - 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + 5*n3**2*(-L1 + L3) + 5*np.sqrt(3)*n2*n4*L4 + 5*np.sqrt(3)*n3*n5*L4)) + sqcp**2*(4*n1*n3*(L1 - 3*L2 + 2*L3) + 8*n0*(n4*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n2*L4) + sqcp*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) - np.sqrt(3)*n2**2*L4 + np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(n4 - n5)*(n4 + n5)*L4)) + sqcm**2*(4*n1*n3*(L1 - 3*L2 + 2*L3) + 8*n0*(2*n4*(L0 + L3) - np.sqrt(3)*n2*L4) + sqcp*(20*n2*n4*(-L1 + L3) + 20*n3*n5*(-L1 + L3) + 5*np.sqrt(3)*n2**2*L4 - 5*np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(-n4**2 + n5**2)*L4)))/48., \
                (-(sqcm**3*(2*n3*n4*(L1 - L3) + 2*n2*n5*(-L1 + L3) + np.sqrt(3)*n2*n3*L4 + 3*np.sqrt(3)*n4*n5*L4)) + sqcm**2*(2*n0*n3*(L1 - 3*L2 + 2*L3) + 4*n1*(n4*(2*L0 - L1 + 3*L2) - np.sqrt(3)*n2*L4) + sqcp*n4*(6*n5*(-L1 + L3) + 5*np.sqrt(3)*n3*L4) - 5*sqcp*n2*(2*n3*(L1 - L3) + np.sqrt(3)*n5*L4)) + sqcp**2*(8*n1*n4*(L0 + L3) + 2*n0*n3*(L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n1*n2*L4 + sqcp*(2*n2*n3*(L1 - L3) + 6*n4*n5*(L1 - L3) - np.sqrt(3)*n3*n4*L4 + np.sqrt(3)*n2*n5*L4)) + sqcm*sqcp*(-4*n1*n2*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n1*n4*L4 + sqcp*(10*n2*n5*(-L1 + L3) + 3*np.sqrt(3)*n4*n5*L4 + 5*n3*(2*n4*(L1 - L3) + np.sqrt(3)*n2*L4))))/24., \
                (-2*sqcm*sqcp*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*sqcp*(2*n0*n2*L1 + 2*n1*n5*L1 - 2*(n0*n2 + n1*n5)*L3 - np.sqrt(3)*n1*n3*L4 + np.sqrt(3)*n0*n4*L4) - 4*sqcm*(-2*n0*n4*L1 + 2*n0*n4*L3 + 2*n1*n3*(-L1 + L3) + sqcp*n2*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) + np.sqrt(3)*n0*n2*L4 - np.sqrt(3)*n1*n5*L4)) - (-8*np.sqrt(3)*(n0**2 + n1**2)*L4 + 4*sqcm*(2*n0*n2*L1 + 2*n1*n5*L1 - 2*(n0*n2 + n1*n5)*L3 + sqcp*(n2**2*(4*L0 - L1 + 6*L2 - L3) + n4**2*(-4*L0 + L1 - 6*L2 + L3) - (n3 - n5)*(n3 + n5)*(2*L1 - 3*L2 + L3)) - np.sqrt(3)*n1*n3*L4 + np.sqrt(3)*n0*n4*L4) + sqcp*(8*n1*n3*(-L1 + L3) + 8*n0*n4*(-L1 + L3) + 4*np.sqrt(3)*n0*n2*L4 - 4*np.sqrt(3)*n1*n5*L4) + sqcp**2*(4*n2*n4*(-4*L0 + L1 - 6*L2 + L3) - 4*n3*n5*(2*L1 - 3*L2 + L3) + 3*np.sqrt(3)*n2**2*L4 + np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(3*n4**2 + n5**2)*L4) + sqcm**2*(4*n2*n4*(4*L0 - L1 + 6*L2 - L3) + 4*n3*n5*(2*L1 - 3*L2 + L3) + 3*np.sqrt(3)*n2**2*L4 + np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(3*n4**2 + n5**2)*L4))*self.cosa)/48., \
                (4*n0*n1*(L1 - 3*L2 + 2*L3) + sqcp**2*(n3*n4 + n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcm**3*sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcm**2*(8*sqcp*n1*n2*(-L1 + L3) + 8*sqcp*n0*n5*(-L1 + L3) + (1 + 2*sqcp**2)*n3*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) - (-1 + 2*sqcp**2)*n2*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) - 4*np.sqrt(3)*sqcp*n0*n3*L4 + 4*np.sqrt(3)*sqcp*n1*n4*L4) + sqcm*sqcp**2*(sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*n1*(2*n4*L1 - 2*n4*L3 + np.sqrt(3)*n2*L4) + n0*(8*n3*L1 - 8*n3*L3 - 4*np.sqrt(3)*n5*L4)) - (2*sqcm*(-2*n0*n3*L1 - 2*n1*n4*L1 + sqcp*(n2*n3 + n4*n5)*(4*L0 + L1 + 3*L2) + 2*n0*n3*L3 + 2*n1*n4*L3 - np.sqrt(3)*n1*n2*L4 + np.sqrt(3)*n0*n5*L4) + sqcm**2*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) + sqcp*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4 + sqcp*(-(n3*n4*(4*L0 + L1 + 3*L2)) + n2*n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*n3*L4 - np.sqrt(3)*n4*n5*L4)))*self.cosa)/24., \
                (-16*np.sqrt(3)*M0 + 16*sqcm**2*n2**2*L0 + 16*sqcp**2*n2**2*L0 - 16*sqcm**2*sqcp**2*n2**2*L0 + 8*sqcm**2*n3**2*L0 + 8*sqcp**2*n3**2*L0 - 16*sqcm**3*sqcp*n2*n4*L0 + 16*sqcm*sqcp**3*n2*n4*L0 + 16*sqcm**2*n4**2*L0 + 16*sqcp**2*n4**2*L0 + 16*sqcm**2*sqcp**2*n4**2*L0 + 8*sqcm**2*n5**2*L0 + 8*sqcp**2*n5**2*L0 + 8*sqcm**2*n2**2*L1 + 8*sqcp**2*n2**2*L1 + 4*sqcm**2*sqcp**2*n2**2*L1 + 6*sqcm**2*n3**2*L1 + 6*sqcp**2*n3**2*L1 + 8*sqcm**2*sqcp**2*n3**2*L1 + 4*sqcm**3*sqcp*n2*n4*L1 - 4*sqcm*sqcp**3*n2*n4*L1 + 8*sqcm**2*n4**2*L1 + 8*sqcp**2*n4**2*L1 - 4*sqcm**2*sqcp**2*n4**2*L1 - 8*sqcm**3*sqcp*n3*n5*L1 + 8*sqcm*sqcp**3*n3*n5*L1 + 6*sqcm**2*n5**2*L1 + 6*sqcp**2*n5**2*L1 - 8*sqcm**2*sqcp**2*n5**2*L1 + 6*sqcm**2*n2**2*L2 + 6*sqcp**2*n2**2*L2 - 24*sqcm**2*sqcp**2*n2**2*L2 - 12*sqcm**2*sqcp**2*n3**2*L2 - 24*sqcm**3*sqcp*n2*n4*L2 + 24*sqcm*sqcp**3*n2*n4*L2 + 6*sqcm**2*n4**2*L2 + 6*sqcp**2*n4**2*L2 + 24*sqcm**2*sqcp**2*n4**2*L2 + 12*sqcm**3*sqcp*n3*n5*L2 - 12*sqcm*sqcp**3*n3*n5*L2 + 12*sqcm**2*sqcp**2*n5**2*L2 + 2*(2*sqcm**3*sqcp*(n2*n4 - n3*n5) + 2*sqcm*sqcp**3*(-(n2*n4) + n3*n5) + sqcp**2*(n2**2 + n3**2 + n4**2 + n5**2) + sqcm**2*((1 + 2*sqcp**2)*n2**2 + (1 + 2*sqcp**2)*n3**2 - (-1 + 2*sqcp**2)*(n4**2 + n5**2)))*L3 + 6*np.sqrt(3)*sqcm*sqcp*n2**2*L4 - 3*np.sqrt(3)*sqcm**3*sqcp*n2**2*L4 - 3*np.sqrt(3)*sqcm*sqcp**3*n2**2*L4 + 2*np.sqrt(3)*sqcm*sqcp*n3**2*L4 - np.sqrt(3)*sqcm**3*sqcp*n3**2*L4 - np.sqrt(3)*sqcm*sqcp**3*n3**2*L4 + 6*np.sqrt(3)*sqcm**2*n2*n4*L4 - 6*np.sqrt(3)*sqcp**2*n2*n4*L4 - 6*np.sqrt(3)*sqcm*sqcp*n4**2*L4 - 3*np.sqrt(3)*sqcm**3*sqcp*n4**2*L4 - 3*np.sqrt(3)*sqcm*sqcp**3*n4**2*L4 - 2*np.sqrt(3)*sqcm**2*n3*n5*L4 + 2*np.sqrt(3)*sqcp**2*n3*n5*L4 - 2*np.sqrt(3)*sqcm*sqcp*n5**2*L4 - np.sqrt(3)*sqcm**3*sqcp*n5**2*L4 - np.sqrt(3)*sqcm*sqcp**3*n5**2*L4 + 4*n0**2*(4*L0 - L1 + 3*L2 + 2*L3 + 2*np.sqrt(3)*sqcm*sqcp*L4) + 4*n1**2*(4*L0 - L1 + 3*L2 + 2*L3 + 2*np.sqrt(3)*sqcm*sqcp*L4) - 4*n0*(sqcm*(2*n4*(L1 - L3) + np.sqrt(3)*n2*L4) + sqcm*sqcp**2*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) + sqcp*(2*(1 + sqcm**2)*n2*(L1 - L3) + np.sqrt(3)*(-1 + sqcm**2)*n4*L4)) + 4*n1*(sqcp*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4) + sqcm**2*sqcp*(2*n5*(-L1 + L3) + np.sqrt(3)*n3*L4) + sqcm*(2*(1 + sqcp**2)*n3*(L1 - L3) + np.sqrt(3)*(-1 + sqcp**2)*n5*L4)) + 2*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n2 - n4)*(n2 + n4)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*sqcp*(2*n0*n2*L1 + 2*n1*n5*L1 - 2*(n0*n2 + n1*n5)*L3 - np.sqrt(3)*n1*n3*L4 + np.sqrt(3)*n0*n4*L4) - 4*sqcm*(-2*n0*n4*L1 + 2*n0*n4*L3 + 2*n1*n3*(-L1 + L3) + sqcp*n2*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) + np.sqrt(3)*n0*n2*L4 - np.sqrt(3)*n1*n5*L4))*self.cosa)/48. + thcorr, \
                (sqcm**3*sqcp*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) - sqcp*(2*n1*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) - sqcp*n4*(n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n3*L4) + sqcp*n2*(n3*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n5*L4) + 2*n0*(2*n3*(L1 - L3) + np.sqrt(3)*n5*L4)) + sqcm**2*((-1 + 2*sqcp**2)*n2*n3*(4*L0 + L1 + 3*L2) + (1 + 2*sqcp**2)*n4*n5*(4*L0 + L1 + 3*L2) + 4*sqcp*n1*n4*(-L1 + L3) - np.sqrt(3)*n3*n4*L4 + np.sqrt(3)*n2*(-2*sqcp*n1 + n5)*L4 + 2*sqcp*n0*(2*n3*(-L1 + L3) + np.sqrt(3)*n5*L4)) - sqcm*(2*np.sqrt(3)*sqcp*(n2*n3 + n4*n5)*L4 + 2*sqcp**2*(2*n1*n2*(L1 - L3) + 2*n0*n5*(L1 - L3) + np.sqrt(3)*n0*n3*L4 - np.sqrt(3)*n1*n4*L4) + 2*(2*n1*n2*(L1 - L3) + 2*n0*n5*(-L1 + L3) + np.sqrt(3)*n0*n3*L4 + np.sqrt(3)*n1*n4*L4) + sqcp**3*(n3*n4*(4*L0 + L1 + 3*L2) - n2*n5*(4*L0 + L1 + 3*L2) - np.sqrt(3)*n2*n3*L4 + np.sqrt(3)*n4*n5*L4)) + (-(sqcm**2*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3)) + 2*sqcm*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 + sqcp*(n3*n4 - n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4) + sqcp*(8*n0*n3*L1 + 8*n1*n4*L1 - 8*n0*n3*L3 - 8*n1*n4*L3 + sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n1*n2*L4 - 4*np.sqrt(3)*n0*n5*L4))*self.cosa)/24.], \
                [(sqcm**3*(2*n2*n3*(L1 - L3) + 6*n4*n5*(L1 - L3) + np.sqrt(3)*n3*n4*L4 - np.sqrt(3)*n2*n5*L4) + sqcm**2*(10*sqcp*n2*n5*(L1 - L3) + 8*n0*n5*(L0 + L3) + 2*n1*n2*(L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n0*n3*L4 + 3*np.sqrt(3)*sqcp*n4*n5*L4 + 5*sqcp*n3*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4)) + sqcm*sqcp*(-4*n0*n3*(L1 - 3*L2 + 2*L3) + 8*np.sqrt(3)*n0*n5*L4 + sqcp*n4*(6*n5*(-L1 + L3) - 5*np.sqrt(3)*n3*L4) + 5*sqcp*n2*(2*n3*(-L1 + L3) + np.sqrt(3)*n5*L4)) - sqcp**2*(-2*n1*n2*(L1 - 3*L2 + 2*L3) + 4*n0*(n5*(-2*L0 + L1 - 3*L2) + np.sqrt(3)*n3*L4) + sqcp*(2*n2*n5*(L1 - L3) + 2*n3*n4*(-L1 + L3) + np.sqrt(3)*n2*n3*L4 + 3*np.sqrt(3)*n4*n5*L4)))/24., \
                (sqcm**3*(4*n2*n4*(L1 - L3) + 4*n3*n5*(L1 - L3) + np.sqrt(3)*n2**2*L4 - np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(-n4**2 + n5**2)*L4) + 2*sqcm**2*(2*n0*n2*(L1 - 3*L2 + 2*L3) + 4*n1*(n5*(2*L0 - L1 + 3*L2) + np.sqrt(3)*n3*L4) + sqcp*(5*n2**2*(L1 - L3) - 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + 5*n3**2*(-L1 + L3) - 5*np.sqrt(3)*n2*n4*L4 - 5*np.sqrt(3)*n3*n5*L4)) + 2*sqcp**2*(2*n0*n2*(L1 - 3*L2 + 2*L3) + n1*(8*n5*(L0 + L3) - 4*np.sqrt(3)*n3*L4) + sqcp*(n3**2*(L1 - L3) + 3*(n4 - n5)*(n4 + n5)*(L1 - L3) + n2**2*(-L1 + L3) + np.sqrt(3)*n2*n4*L4 + np.sqrt(3)*n3*n5*L4)) + sqcm*sqcp*(8*n1*(n3*(L1 - 3*L2 + 2*L3) + 2*np.sqrt(3)*n5*L4) + sqcp*(20*n2*n4*(-L1 + L3) + 20*n3*n5*(-L1 + L3) - 5*np.sqrt(3)*n2**2*L4 + 5*np.sqrt(3)*n3**2*L4 + 3*np.sqrt(3)*(n4 - n5)*(n4 + n5)*L4)))/48., \
                (4*n0*n1*(L1 - 3*L2 + 2*L3) + sqcp**2*(n3*n4 + n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcm**3*sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcm**2*(8*sqcp*n1*n2*(L1 - L3) + 8*sqcp*n0*n5*(L1 - L3) - (-1 + 2*sqcp**2)*n3*n4*(4*L0 + 5*L1 - 3*L2 + 2*L3) + (1 + 2*sqcp**2)*n2*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*sqcp*n0*n3*L4 - 4*np.sqrt(3)*sqcp*n1*n4*L4) - sqcm*sqcp**2*(sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*n1*(2*n4*L1 - 2*n4*L3 + np.sqrt(3)*n2*L4) + n0*(8*n3*L1 - 8*n3*L3 - 4*np.sqrt(3)*n5*L4)) + (2*sqcm*(-2*n0*n3*L1 - 2*n1*n4*L1 + sqcp*(n2*n3 + n4*n5)*(4*L0 + L1 + 3*L2) + 2*n0*n3*L3 + 2*n1*n4*L3 - np.sqrt(3)*n1*n2*L4 + np.sqrt(3)*n0*n5*L4) + sqcm**2*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) + sqcp*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4 + sqcp*(-(n3*n4*(4*L0 + L1 + 3*L2)) + n2*n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*n3*L4 - np.sqrt(3)*n4*n5*L4)))*self.cosa)/24., \
                (2*sqcm*sqcp*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcp*(8*n0*n2*(-L1 + L3) + 8*n1*n5*(-L1 + L3) + 4*np.sqrt(3)*n1*n3*L4 - 4*np.sqrt(3)*n0*n4*L4) + 4*sqcm*(sqcp*n3*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) + n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4))) + (-8*np.sqrt(3)*(n0**2 + n1**2)*L4 + 4*sqcm*(-2*n0*n2*L1 - 2*n1*n5*L1 + 2*n0*n2*L3 + 2*n1*n5*L3 + sqcp*(n3**2*(4*L0 - L1 + 6*L2 - L3) + n5**2*(-4*L0 + L1 - 6*L2 + L3) - (n2 - n4)*(n2 + n4)*(2*L1 - 3*L2 + L3)) + np.sqrt(3)*n1*n3*L4 - np.sqrt(3)*n0*n4*L4) + 4*sqcp*(2*n1*n3*L1 + 2*n0*n4*L1 - 2*(n1*n3 + n0*n4)*L3 - np.sqrt(3)*n0*n2*L4 + np.sqrt(3)*n1*n5*L4) + sqcm**2*(4*n3*n5*(-4*L0 + L1 - 6*L2 + L3) + 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(n4**2 + 3*n5**2)*L4 + n2*(-4*n4*(2*L1 - 3*L2 + L3) + np.sqrt(3)*n2*L4)) + sqcp**2*(4*n3*n5*(4*L0 - L1 + 6*L2 - L3) + 3*np.sqrt(3)*n3**2*L4 + np.sqrt(3)*(n4**2 + 3*n5**2)*L4 + n2*(4*n4*(2*L1 - 3*L2 + L3) + np.sqrt(3)*n2*L4)))*self.cosa)/48., \
                (sqcm**3*sqcp*(n3*(n4*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n2*L4) - n5*(n2*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n4*L4)) - sqcp*(2*n1*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) - sqcp*n4*(n5*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n3*L4) + sqcp*n2*(n3*(4*L0 + L1 + 3*L2) + np.sqrt(3)*n5*L4) + 2*n0*(2*n3*(L1 - L3) + np.sqrt(3)*n5*L4)) + sqcm**2*((-1 + 2*sqcp**2)*n2*n3*(4*L0 + L1 + 3*L2) + (1 + 2*sqcp**2)*n4*n5*(4*L0 + L1 + 3*L2) + 4*sqcp*n1*n4*(-L1 + L3) - np.sqrt(3)*n3*n4*L4 + np.sqrt(3)*n2*(-2*sqcp*n1 + n5)*L4 + 2*sqcp*n0*(2*n3*(-L1 + L3) + np.sqrt(3)*n5*L4)) - sqcm*(2*np.sqrt(3)*sqcp*(n2*n3 + n4*n5)*L4 + 2*sqcp**2*(2*n1*n2*(L1 - L3) + 2*n0*n5*(L1 - L3) + np.sqrt(3)*n0*n3*L4 - np.sqrt(3)*n1*n4*L4) + 2*(2*n1*n2*(L1 - L3) + 2*n0*n5*(-L1 + L3) + np.sqrt(3)*n0*n3*L4 + np.sqrt(3)*n1*n4*L4) + sqcp**3*(n3*n4*(4*L0 + L1 + 3*L2) - n2*n5*(4*L0 + L1 + 3*L2) - np.sqrt(3)*n2*n3*L4 + np.sqrt(3)*n4*n5*L4)) + (-(sqcm**2*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3)) + 2*sqcm*(-4*n1*n2*L1 - 4*n0*n5*L1 + 4*n1*n2*L3 + 4*n0*n5*L3 + sqcp*(n3*n4 - n2*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - 2*np.sqrt(3)*n0*n3*L4 + 2*np.sqrt(3)*n1*n4*L4) + sqcp*(8*n0*n3*L1 + 8*n1*n4*L1 - 8*n0*n3*L3 - 8*n1*n4*L3 + sqcp*(n2*n3 + n4*n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + 4*np.sqrt(3)*n1*n2*L4 - 4*np.sqrt(3)*n0*n5*L4))*self.cosa)/24., \
                (-16*np.sqrt(3)*M0 + 8*sqcm**2*n2**2*L0 + 8*sqcp**2*n2**2*L0 + 16*sqcm**2*n3**2*L0 + 16*sqcp**2*n3**2*L0 - 16*sqcm**2*sqcp**2*n3**2*L0 + 8*sqcm**2*n4**2*L0 + 8*sqcp**2*n4**2*L0 + 16*sqcm**3*sqcp*n3*n5*L0 - 16*sqcm*sqcp**3*n3*n5*L0 + 16*sqcm**2*n5**2*L0 + 16*sqcp**2*n5**2*L0 + 16*sqcm**2*sqcp**2*n5**2*L0 + 6*sqcm**2*n2**2*L1 + 6*sqcp**2*n2**2*L1 + 8*sqcm**2*sqcp**2*n2**2*L1 + 8*sqcm**2*n3**2*L1 + 8*sqcp**2*n3**2*L1 + 4*sqcm**2*sqcp**2*n3**2*L1 + 8*sqcm**3*sqcp*n2*n4*L1 - 8*sqcm*sqcp**3*n2*n4*L1 + 6*sqcm**2*n4**2*L1 + 6*sqcp**2*n4**2*L1 - 8*sqcm**2*sqcp**2*n4**2*L1 - 4*sqcm**3*sqcp*n3*n5*L1 + 4*sqcm*sqcp**3*n3*n5*L1 + 8*sqcm**2*n5**2*L1 + 8*sqcp**2*n5**2*L1 - 4*sqcm**2*sqcp**2*n5**2*L1 - 12*sqcm**2*sqcp**2*n2**2*L2 + 6*sqcm**2*n3**2*L2 + 6*sqcp**2*n3**2*L2 - 24*sqcm**2*sqcp**2*n3**2*L2 - 12*sqcm**3*sqcp*n2*n4*L2 + 12*sqcm*sqcp**3*n2*n4*L2 + 12*sqcm**2*sqcp**2*n4**2*L2 + 24*sqcm**3*sqcp*n3*n5*L2 - 24*sqcm*sqcp**3*n3*n5*L2 + 6*sqcm**2*n5**2*L2 + 6*sqcp**2*n5**2*L2 + 24*sqcm**2*sqcp**2*n5**2*L2 + 2*(2*sqcm**3*sqcp*(n2*n4 - n3*n5) + 2*sqcm*sqcp**3*(-(n2*n4) + n3*n5) + sqcp**2*(n2**2 + n3**2 + n4**2 + n5**2) + sqcm**2*((1 + 2*sqcp**2)*n2**2 + (1 + 2*sqcp**2)*n3**2 - (-1 + 2*sqcp**2)*(n4**2 + n5**2)))*L3 + 2*np.sqrt(3)*sqcm*sqcp*n2**2*L4 - np.sqrt(3)*sqcm**3*sqcp*n2**2*L4 - np.sqrt(3)*sqcm*sqcp**3*n2**2*L4 + 6*np.sqrt(3)*sqcm*sqcp*n3**2*L4 - 3*np.sqrt(3)*sqcm**3*sqcp*n3**2*L4 - 3*np.sqrt(3)*sqcm*sqcp**3*n3**2*L4 + 2*np.sqrt(3)*sqcm**2*n2*n4*L4 - 2*np.sqrt(3)*sqcp**2*n2*n4*L4 - 2*np.sqrt(3)*sqcm*sqcp*n4**2*L4 - np.sqrt(3)*sqcm**3*sqcp*n4**2*L4 - np.sqrt(3)*sqcm*sqcp**3*n4**2*L4 - 6*np.sqrt(3)*sqcm**2*n3*n5*L4 + 6*np.sqrt(3)*sqcp**2*n3*n5*L4 - 6*np.sqrt(3)*sqcm*sqcp*n5**2*L4 - 3*np.sqrt(3)*sqcm**3*sqcp*n5**2*L4 - 3*np.sqrt(3)*sqcm*sqcp**3*n5**2*L4 + 4*n0**2*(4*L0 - L1 + 3*L2 + 2*L3 + 2*np.sqrt(3)*sqcm*sqcp*L4) + 4*n1**2*(4*L0 - L1 + 3*L2 + 2*L3 + 2*np.sqrt(3)*sqcm*sqcp*L4) + 4*n0*(sqcm*(2*n4*(L1 - L3) + np.sqrt(3)*n2*L4) + sqcm*sqcp**2*(2*n4*(-L1 + L3) + np.sqrt(3)*n2*L4) + sqcp*(2*(1 + sqcm**2)*n2*(L1 - L3) + np.sqrt(3)*(-1 + sqcm**2)*n4*L4)) - 4*n1*(sqcp*(2*n5*(L1 - L3) + np.sqrt(3)*n3*L4) + sqcm**2*sqcp*(2*n5*(-L1 + L3) + np.sqrt(3)*n3*L4) + sqcm*(2*(1 + sqcp**2)*n3*(L1 - L3) + np.sqrt(3)*(-1 + sqcp**2)*n5*L4)) + 2*(-2*(n0 - n1)*(n0 + n1)*(L1 - 3*L2 + 2*L3) + sqcm**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) - sqcp**2*(n3 - n5)*(n3 + n5)*(4*L0 + 5*L1 - 3*L2 + 2*L3) + sqcp*(8*n0*n2*(-L1 + L3) + 8*n1*n5*(-L1 + L3) + 4*np.sqrt(3)*n1*n3*L4 - 4*np.sqrt(3)*n0*n4*L4) + 4*sqcm*(sqcp*n3*n5*(4*L0 + 5*L1 - 3*L2 + 2*L3) + n0*(-2*n4*L1 + 2*n4*L3 + np.sqrt(3)*n2*L4) - n1*(2*n3*L1 - 2*n3*L3 + np.sqrt(3)*n5*L4)))*self.cosa)/48. + thcorr]])
        
        # Swapping axes so the last two axes correspond to the matrices. Necessary to compute eigenvalues.
        nm = np.moveaxis(nm, [0, 1], [-2, -1])
        cm = np.moveaxis(cm, [0, 1], [-2, -1])

        # Physical fields correspond to the eigenvalues of the mass matrix + thermal corrections
        neigvals = np.linalg.eigvalsh(nm)
        ceigvals = np.linalg.eigvalsh(cm)

        # Swapping axes so the first axis correspond to fields
        neigvals = np.moveaxis(neigvals, -1, 0)
        ceigvals = np.moveaxis(ceigvals, -1, 0)
        
        # V: Vector bosons W+, W-, Z, gam (L = 0, T = +/-1) -- only L-polarisations have to be T-corrected!

        # thermally-corrected longitudinal (L) vector bosons
        
        mWL   = (((H10**2 + H11**2 + H20**2 + H21**2 + H30**2 + H31**2)/4.+11./6.*T**2)*self.gl**2)

        vm11  = mWL
        vm22  = ((H10**2 + H11**2 + H20**2 + H21**2 + H30**2 + H31**2)/4.+11./6.*T**2)*self.gy**2
        vm12  = -self.gl*self.gy*(H10**2 + H11**2 + H20**2 + H21**2 + H30**2 + H31**2)/4.

        mZL   = 0.5*(vm11+vm22+np.sqrt(4*vm12**2+vm11**2-2*vm11*vm22+vm22**2))
        mgamL = 0.5*(vm11+vm22-np.sqrt(4*vm12**2+vm11**2-2*vm11*vm22+vm22**2))

        # thermally-uncorrected transverse (T) vector bosons
        mWT   = (H10**2 + H11**2 + H20**2 + H21**2 + H30**2 + H31**2)/4.*self.gl**2
        mZT   = (H10**2 + H11**2 + H20**2 + H21**2 + H30**2 + H31**2)/4.*(self.gl**2+self.gy**2)
        
        # M: total boson masses
        M = np.array([neigvals[0].astype(float) ,neigvals[1].astype(float) ,neigvals[2].astype(float) ,neigvals[3].astype(float) ,neigvals[4].astype(float) ,neigvals[5].astype(float) ,ceigvals[0].astype(float) ,ceigvals[1].astype(float) ,ceigvals[2].astype(float) ,ceigvals[3].astype(float) ,ceigvals[4].astype(float) ,ceigvals[5].astype(float) ,mWL,mZL,mgamL,mWT,mZT])
        #M = np.array([mWL,mZL,mgamL,mWT,mZT])

        # The number of degrees of freedom for the masses.
        dof = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 2])
        #dof = np.array([2, 1, 1, 4, 2])
       
        # CW constant
        c = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0.5,0.5])
        #c = np.array([1.5,1.5,1.5,0.5,0.5])
        
        # Swapping axes so the last axis correspond to fields      
        M = np.rollaxis(M, 0, len(M.shape))

        return M, dof, c

    def fermion_massSq(self, X):
        X = np.asanyarray(X)
        n0,n1,n2,n3,n4,n5 = X[...,0], X[...,1], X[...,2], X[...,3], X[...,4], X[...,5]

        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)

        H10 = n1/np.sqrt(3) + ( n2 - n3/np.sqrt(3))/2. * sqcm + (-n4 - n5/np.sqrt(3))/2. * sqcp
        H11 = n0/np.sqrt(3) + ( n5 - n4/np.sqrt(3))/2. * sqcm + (-n3 - n2/np.sqrt(3))/2. * sqcp
        H20 = n1/np.sqrt(3) + (-n2 - n3/np.sqrt(3))/2. * sqcm + ( n4 - n5/np.sqrt(3))/2. * sqcp
        H21 = n0/np.sqrt(3) + (-n5 - n4/np.sqrt(3))/2. * sqcm + ( n3 - n2/np.sqrt(3))/2. * sqcp
        H30 = n1/np.sqrt(3) + n3/np.sqrt(3) * sqcm + n5/np.sqrt(3) * sqcp
        H31 = n0/np.sqrt(3) + n4/np.sqrt(3) * sqcm + n2/np.sqrt(3) * sqcp

        #Lepton Mass Matrix
        lep_mm = np.array([[1/2.*(self.ye**2 + self.ymuon**2 + self.ytau**2)*(H10**2 + H11**2), \
                            1/4.*(2*self.ye**2 + (-1-1j*np.sqrt(3))*self.ymuon**2 + (-1+1j*np.sqrt(3))*self.ytau**2)*(H11 - H10*1j)*(H21 + H20*1j), \
                            1/4.*(2*self.ye**2 + (-1+1j*np.sqrt(3))*self.ymuon**2 + (-1-1j*np.sqrt(3))*self.ytau**2)*(H11 - H10*1j)*(H31 + H30*1j)], \
                           [1/4.*(2*self.ye**2 + (-1+1j*np.sqrt(3))*self.ymuon**2 + (-1-1j*np.sqrt(3))*self.ytau**2)*(H21 - H20*1j)*(H11 + H10*1j), \
                            1/2.*(self.ye**2 + self.ymuon**2 + self.ytau**2)*(H20**2 + H21**2), \
                            1/4.*(2*self.ye**2 + (-1-1j*np.sqrt(3))*self.ymuon**2 + (-1+1j*np.sqrt(3))*self.ytau**2)*(H21 - H20*1j)*(H31 + H30*1j)], \
                           [1/4.*(2*self.ye**2 + (-1-1j*np.sqrt(3))*self.ymuon**2 + (-1+1j*np.sqrt(3))*self.ytau**2)*(H31 - H30*1j)*(H11 + H10*1j), \
                            1/4.*(2*self.ye**2 + (-1+1j*np.sqrt(3))*self.ymuon**2 + (-1-1j*np.sqrt(3))*self.ytau**2)*(H31 - H30*1j)*(H21 + H20*1j), \
                            1/2.*(self.ye**2 + self.ymuon**2 + self.ytau**2)*(H30**2 + H31**2)]])
        
        #Up-quarks Mass Matrix
        up_mm = np.array([[1/2.*(self.yu**2 + self.yc**2 + self.yt**2)*(H10**2 + H11**2), \
                            1/4.*(2*self.yu**2 + (-1-1j*np.sqrt(3))*self.yc**2 + (-1+1j*np.sqrt(3))*self.yt**2)*(H11 + H10*1j)*(H21 - H20*1j), \
                            1/4.*(2*self.yu**2 + (-1+1j*np.sqrt(3))*self.yc**2 + (-1-1j*np.sqrt(3))*self.yt**2)*(H11 + H10*1j)*(H31 - H30*1j)], \
                           [1/4.*(2*self.yu**2 + (-1+1j*np.sqrt(3))*self.yc**2 + (-1-1j*np.sqrt(3))*self.yt**2)*(H21 + H20*1j)*(H11 - H10*1j), \
                            1/2.*(self.yu**2 + self.yc**2 + self.yt**2)*(H20**2 + H21**2), \
                            1/4.*(2*self.yu**2 + (-1-1j*np.sqrt(3))*self.yc**2 + (-1+1j*np.sqrt(3))*self.yt**2)*(H21 + H20*1j)*(H31 - H30*1j)], \
                           [1/4.*(2*self.yu**2 + (-1-1j*np.sqrt(3))*self.yc**2 + (-1+1j*np.sqrt(3))*self.yt**2)*(H31 + H30*1j)*(H11 - H10*1j), \
                            1/4.*(2*self.yu**2 + (-1+1j*np.sqrt(3))*self.yc**2 + (-1-1j*np.sqrt(3))*self.yt**2)*(H31 + H30*1j)*(H21 - H20*1j), \
                            1/2.*(self.yu**2 + self.yc**2 + self.yt**2)*(H30**2 + H31**2)]])
        
        #Down-quarks Mass Matrix
        down_mm = np.array([[1/2.*(self.yd**2 + self.ys**2 + self.yb**2)*(H10**2 + H11**2), \
                            1/4.*(2*self.yd**2 + (-1-1j*np.sqrt(3))*self.ys**2 + (-1+1j*np.sqrt(3))*self.yb**2)*(H11 - H10*1j)*(H21 + H20*1j), \
                            1/4.*(2*self.yd**2 + (-1+1j*np.sqrt(3))*self.ys**2 + (-1-1j*np.sqrt(3))*self.yb**2)*(H11 - H10*1j)*(H31 + H30*1j)], \
                           [1/4.*(2*self.yd**2 + (-1+1j*np.sqrt(3))*self.ys**2 + (-1-1j*np.sqrt(3))*self.yb**2)*(H21 - H20*1j)*(H11 + H10*1j), \
                            1/2.*(self.yd**2 + self.ys**2 + self.yb**2)*(H20**2 + H21**2), \
                            1/4.*(2*self.yd**2 + (-1-1j*np.sqrt(3))*self.ys**2 + (-1+1j*np.sqrt(3))*self.yb**2)*(H21 - H20*1j)*(H31 + H30*1j)], \
                           [1/4.*(2*self.yd**2 + (-1-1j*np.sqrt(3))*self.ys**2 + (-1+1j*np.sqrt(3))*self.yb**2)*(H31 - H30*1j)*(H11 + H10*1j), \
                            1/4.*(2*self.yd**2 + (-1+1j*np.sqrt(3))*self.ys**2 + (-1-1j*np.sqrt(3))*self.yb**2)*(H31 - H30*1j)*(H21 + H20*1j), \
                            1/2.*(self.ys**2 + self.ys**2 + self.yb**2)*(H30**2 + H31**2)]])
        
        # Swapping axes so the last two axes correspond to the matrices. Necessary to compute eigenvalues.
        lep_mm = np.moveaxis(lep_mm, [0, 1], [-2, -1])
        up_mm = np.moveaxis(up_mm, [0, 1], [-2, -1])
        down_mm = np.moveaxis(down_mm, [0, 1], [-2, -1])
      
        # Physical fields correspond to the eigenvalues of the mass matrix + thermal corrections
        lep_mm_eigvals = np.real(np.linalg.eigvalsh(lep_mm))
        up_mm_eigvals = np.real(np.linalg.eigvalsh(up_mm))
        down_mm_eigvals = np.real(np.linalg.eigvalsh(down_mm))

        # Swapping axes so the first axis correspond to fields
        lep_mm_eigvals = np.moveaxis(lep_mm_eigvals, -1, 0)
        up_mm_eigvals = np.moveaxis(up_mm_eigvals, -1, 0)
        down_mm_eigvals = np.moveaxis(down_mm_eigvals, -1, 0)

        melec2 = lep_mm_eigvals[0].astype(float) 
        mmuon2 = lep_mm_eigvals[1].astype(float) 
        mtau2 = lep_mm_eigvals[2].astype(float) 
        
        mup2 = up_mm_eigvals[0].astype(float) 
        mcharm2 = up_mm_eigvals[1].astype(float) 
        mtop2 = up_mm_eigvals[2].astype(float) 
        
        mdown2 = down_mm_eigvals[0].astype(float) 
        mstrange2 = down_mm_eigvals[1].astype(float) 
        mbot2 = down_mm_eigvals[2].astype(float) 
        
        # M: total boson masses
        M = np.array([mtop2,mbot2,mtau2,mcharm2,mstrange2,mmuon2,mup2,mdown2,melec2])

        # Swapping axes so the last axis correspond to fields  
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses.
        dof = np.array([12,12,4,12,12,4,12,12,4])

        return M,dof

    def approxZeroTMin(self):
        # Approximate minimum at T=0. Giving tree-level minimum
        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)
        min0 = np.array([0.,0. ,0., 0., 0., 0.])
        min1 = np.array([0.,self.vh ,0., 0., 0., 0.])
        min2 = np.array([0.,self.vh/np.sqrt(3), self.vh/2. * sqcm, -self.vh/np.sqrt(3)/2. * sqcm, -self.vh/2. * sqcp, -self.vh/np.sqrt(3)/2. * sqcp])
        min3 =np.array([0.,self.vh *2/3., (self.vh/np.sqrt(3)* sqcm - self.vh*np.sqrt(3)* sqcp)/4., -self.vh/12.* sqcm + self.vh/4.* sqcp,  (-self.vh/np.sqrt(3)* sqcp - self.vh*np.sqrt(3)* sqcm)/4.,-self.vh/4.* sqcm - self.vh/12.* sqcp])
        return [min1]

    def approxFiniteTMin(self):
        # Approximate minimum at T=0. Giving tree-level minimum
        sqcm = np.sqrt(1-self.cosa)
        sqcp = np.sqrt(1+self.cosa)
        min0 = np.array([0.,0. ,0., 0., 0., 0.])
        min1 = np.array([0.,self.vh ,0., 0., 0., 0.])
        min2 = np.array([0.,self.vh/np.sqrt(3), self.vh/2. * sqcm, -self.vh/np.sqrt(3)/2. * sqcm, -self.vh/2. * sqcp, -self.vh/np.sqrt(3)/2. * sqcp])
        min3 =np.array([0.,self.vh *2/3., (self.vh/np.sqrt(3)* sqcm - self.vh*np.sqrt(3)* sqcp)/4., -self.vh/12.* sqcm + self.vh/4.* sqcp,  (-self.vh/np.sqrt(3)* sqcp - self.vh*np.sqrt(3)* sqcm)/4.,-self.vh/4.* sqcm - self.vh/12.* sqcp])
        return [[min0,200.0]]

    def getPhases(self,tracingArgs={}):
        """
        Find different phases as functions of temperature

        Parameters
        ----------
        tracingArgs : dict
            Parameters to pass to :func:`transitionFinder.traceMultiMin`.

        Returns
        -------
        dict
            Each item in the returned dictionary is an instance of
            :class:`transitionFinder.Phase`, and each phase is
            identified by a unique key. This value is also stored in
            `self.phases`.
        """
        tstop = self.Tmax
        points = []
        for x0 in self.approxFiniteTMin():
            points.append(x0)
        for x0 in self.approxZeroTMin():
            points.append([x0,0.0])
        tracingArgs_ = dict(forbidCrit=self.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        phases = transitionFinder.traceMultiMin(
            self.Vtot, self.dgradV_dT, self.d2V, points,
            tLow=0.0, tHigh=tstop, deltaX_target=2*np.pi*100*self.x_eps,
            **tracingArgs_)
        self.phases = phases
        transitionFinder.removeRedundantPhases(
            self.Vtot, phases, self.x_eps*1e-2, 10)
        return self.phases