import numpy as np
import pickle
import os
from A4_spectrum import A4_spectrum

def parsL(box, n, bfb = True, positivity = True, unitary = False, Mh=125.10, vh=246.22):
    M0 = np.sqrt(3)/2.*Mh**2
    def bfbFilter(L1, L2, L3, L4):
        L0 = np.sqrt(3)/vh**2*M0 - L1
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

    def unitFilter(L1, L2, L3, L4):
        L0 = np.sqrt(3)/vh**2*M0 - L1
        req0 = abs(L0) < 2.0
        req1 = abs(L1) < 2.0
        req2 = abs(L2) < 2.0
        req3 = abs(L3) < 2.0
        req4 = abs(L4) < 2.0
        cond = req0 and req1 and req2 and req3 and req4
        return cond

    print('Creating Parameters List...')
    L1 = np.linspace(box[0][0], box[-1][0], n[0], dtype=float)
    L2 = np.linspace(box[0][1], box[-1][1], n[1], dtype=float)
    L3 = np.linspace(box[0][2], box[-1][2], n[2], dtype=float)
    L4 = np.linspace(box[0][3], box[-1][3], n[3], dtype=float)
    L = np.array(np.meshgrid(L1, L2, L3, L4)).T.reshape(-1, 4)

    L = np.unique(L, axis=0)


    pars = []
    ind_to_remove = []
    for i in range(L.shape[0]):
        if bfb:
            if bfbFilter(L[i][0], L[i][1], L[i][2], L[i][3]):
                ind_to_remove.append(i)
        if unitary:
            if unitFilter(L[i][0], L[i][1], L[i][2], L[i][3]):
                ind_to_remove.append(i)
    L = np.delete(L, ind_to_remove, 0)
    print(L)
    for i in range(L.shape[0]):
        Mn1 = vh**2/12 *(-5*L[i][0]+3*L[i][1]+2*L[i][2] + np.sqrt((-L[i][0]+3*L[i][1]-2*L[i][2])**2 + 12*L[i][3]**2))
        Mn2 = vh**2/12 *(-5*L[i][0]+3*L[i][1]+2*L[i][2] - np.sqrt((-L[i][0]+3*L[i][1]-2*L[i][2])**2 + 12*L[i][3]**2))
        Mch1 = vh**2 * (-L[i][0]/2 + L[i][3]/(4*np.sqrt(3)))
        Mch2 = vh**2 * (-L[i][0]/2 - L[i][3]/(4*np.sqrt(3)))
        if Mn1>0. and Mn2>0. and Mch1>0. and Mch2>0.:
            pars.append([np.sqrt(Mn1), np.sqrt(Mn2), np.sqrt(Mch1), np.sqrt(Mch2)])

    return pars 


def createPars(box, n, isMasses=True, prefilter = True): #isMasses=True: Mn1, Mn2, Mch1, Mch2; isMasses=False: Mn1+Mn2, Mn1-Mn2, Mch1+Mch2, Mch1-Mch2
    def preunitfilter(ms, Mh=125.1, vh=246.22):
        M0 = np.sqrt(3)/2.*Mh**2
        L1 = -(ms[2]**2 + ms[3]**2)/(vh**2)
        L4 = 2.*np.sqrt(3)*(ms[2]**2 - ms[3]**2)/(vh**2)
        L0 = np.sqrt(3)*M0/(vh**2) - L1
        xc1 = 6.*(ms[0]**2 + ms[1]**2)/(vh**2) + 5*L1
        xc2 = np.sqrt((6.*(ms[0]**2 - ms[1]**2)/(vh**2))**2 - 12*L4**2)
        L2 = 1/6.*(xc1 + xc2 + L1)
        L3 = 1/4.*(xc1 - xc2 - L1)

        req1 = np.abs(L0) < np.pi/2.
        req2 = np.abs(L1) < np.pi/2.
        req3 = np.abs(L2) < np.pi/2.
        req4 = np.abs(L3) < np.pi/2.
        req5 = np.abs(L4) < np.pi/2.
  
        cond = req1 and req2 and req3 and req4 and req5

        return cond

    print('Creating Parameters List...')
    Mn1 = np.linspace(box[0][0], box[-1][0], n[0], dtype=float)
    Mn2 = np.linspace(box[0][1], box[-1][1], n[1], dtype=float)
    Mch1 = np.linspace(box[0][2], box[-1][2], n[2], dtype=float)
    Mch2 = np.linspace(box[0][3], box[-1][3], n[3], dtype=float)
    pars = np.array(np.meshgrid(Mn1, Mn2, Mch1, Mch2)).T.reshape(-1, 4)
    if(isMasses is False):
        ind_to_remove = np.where(pars[...,3]== 0.)
        pars = np.delete(pars, ind_to_remove, 0)
        ind_to_remove = np.where(pars[...,0]-np.abs(pars[...,1])<0.)
        pars = np.delete(pars, ind_to_remove, 0)
        ind_to_remove = np.where(pars[...,2]-np.abs(pars[...,3])<0.)
        pars = np.delete(pars, ind_to_remove, 0)

        pars_temp = np.zeros_like(pars)
        pars_temp[...,0] = (pars[...,0] + pars[...,1])/2.
        pars_temp[...,1] = (pars[...,0] - pars[...,1])/2.
        pars_temp[...,2] = (pars[...,2] + pars[...,3])/2.
        pars_temp[...,3] = (pars[...,2] - pars[...,3])/2.
        pars = pars_temp


    print('Raw parameters list: ', pars.shape)
    pars = np.unique(pars, axis=0)
    ind_to_remove = np.where(np.abs(pars[...,2] - pars[...,3]) < 1)
    pars = np.delete(pars, ind_to_remove, 0)
        
    print('Unique parameters list shape before removing invalid masses: ', pars.shape)
    ind_to_remove = np.where((6.*(pars[...,0]**2 - pars[...,1]**2))**2 - 12*(2.*np.sqrt(3)*(pars[...,2]**2 - pars[...,3]**2))**2 <= 0. )
    pars = np.delete(pars, ind_to_remove, 0)
    print('Unique parameters list shape after removing invalid masses: ', pars.shape)
    pars_list = []
    for i in range(pars.shape[0]):
        if prefilter:
            if preunitfilter(pars[i]):
                pars_list.append(pars[i])
        else:
            pars_list.append(pars[i])

    print('Final list size: {0}'.format(len(pars_list)))
    return pars_list

def checkFiles(path = './bin/', redofile = 'redopars.csv', logfile = None):
    is_empty = []
    has_phases = []
    no_trans = []
    has_trans = []
    has_gw = []

    for filename in os.listdir(path):
        f = os.path.join(path,filename)
        if os.path.isfile(f) and os.path.getsize(f)>0:
            print(f, os.path.getsize(f))
            #try:
            m = A4_spectrum(cachefile=f, verbose=0)
            if(m.modelinfo):
                if m.mgr.phases is None:
                    is_empty.append(m)
                elif m.modelinfo.has_key('NTrans'):
                    has_phases.append(m)
                elif m.modelinfo['NTrans'] == 0:
                    no_trans.append(m)
                elif m.modelinfo['NTrans'] > 0 and len(m.spectrainfo) == 0:
                    has_trans.append(m)
                elif m.modelinfo['NTrans'] > 0 and len(m.spectrainfo) > 0:
                    has_gw.append(m)
                else:
                    #The code should never enter this block
                    print('Something went wrong.')
            #except:
            #    print('Could not load file {0}.'.format(f))
            #    pass
    
    redo = is_empty + has_phases + no_trans
    redo_pars = []
    for mt in redo:
        redo_pars.append([mt.modelinfo['Mn1'], mt.modelinfo['Mn2'], mt.modelinfo['Mch1'], mt.modelinfo['Mch2']])
    redo_pars = np.asanyarray(redo_pars)
    np.savetxt(redofile, redo_pars, delimiter=',')

    if logfile is not None:
        with open(logfile, 'w') as f:
            f.write('Log: Stored Uncorrupt Files ({0})\n'.format(len(redo + has_trans + has_gw)))
            f.write('# Objects just initialized: {0}\n'.format(len(is_empty)))
            f.write('# Objects with Phases: {0}\n'.format(len(has_phases + no_trans + has_trans + has_gw)))
            f.write('# Objects with No Transitions: {0}\n'.format(len(has_phases + no_trans)))
            f.write('# Objects with Transitions: {0}\n'.format(len(has_trans + has_gw)))
            f.write('# Objects with GW Spectra: {0}\n'.format(len(has_gw)))

    return redo_pars

def main():
    checkFiles(logfile = 'log.txt')
    #print(np.array(np.meshgrid([0,1], [0,3/4.], [0,1], [-np.sqrt(3)/4.,0,np.sqrt(3)/4])).T.reshape(-1, 4))
    #m = A4_spectrum(Mn1=10.,Mn2=20.,Mch1=20.,Mch2=22., verbose = 1, forcetrans=False, T_eps=5e-4, path='./testing/')
    #print(m.mgr.tree_lvl_conditions())
    #print(m.mgr.unitary())
    
if __name__ == '__main__':
  main()