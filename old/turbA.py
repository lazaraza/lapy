import os
import numpy as np
import matplotlib.pyplot as plt
import aotools
from astropy.io import fits
from ABBOT.gradientOperator import gradientOperatorType1 as gOT1
import tqdm

try:
    from aotools import zernike_noll as zern
except:
    from aotools import zernike as zern

import mypy
import turbV


from tqdm import tqdm, trange, tqdm_notebook
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import sys

SH_scaling = 436.1 # nm/pxl
Mag_factor = 1.

sys.path.append('/home/ls/dev/turb')
darcconf   = __import__('turb')
wfsObj     = turbV.WFS(confile='turb')
homedir    = os.environ['HOME']
wavelength = 633

class fitNoll:
    def __init__(self, inp=None, **kwargs):
        def_kw = {
            'n_zerns'   :99,
            'nx_size'   :970,
            'D'         :0.0098,
            'r0'        :0.0009,
            'L0'        :None,
            'wavelength':633
            }
        # initialize all allowed keys to false
        self.__dict__.update((key, value) for key, value in def_kw.items())
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in def_kw)
        print(self.__dict__)

        # Set up parameters for creating phase screens
        #self.n_zerns = kwargs['n_zerns']
        #self.nx_size = kwargs['nx_size']
        #self.D = kwargs['D']
        #self.r0 = kwargs['r0']
        #self.L0 = kwargs['L0']
        self.pxl_scale = self.D/self.nx_size
        self.wind_speed = .0 #m/s - just arbitrarily set
        self.noll = ( fits.getdata("/home/ls/dev/noll.fits").diagonal() * (wavelength/(2*np.pi))**2 )#in [nm]
        #self.r0f = self.r0



        if inp is None:
            self.inp()
        else:
            self.in_spec = inp
            self.invokeMin()

        self.plotNoll()

    def __repr__(self):
        return '{self.__class__.__name__}(D={self.D}, r0={self.r0}, pxl_size={self.pxl_scale})'.format(self=self)

    def getNoll(self, pars=[0.0003,], inp = None):
        D    = self.D#0.05#pars[0]
        r0   = pars[0]
        expo = 5./3
        mynoll = self.noll[:self.n_zerns] * ((D/r0)**(expo))
        if inp is None:
            inp = self.in_spec
        #else:
        #plt.pause(0.01)
        self.mynoll = mynoll.copy()
        #self.plotNoll()

        #minimisation criteria
        #crit = (((mynoll[:] - inp[:])**2).sum())
        crit = (((np.log10(mynoll[:]) - np.log10(inp[:]))**2).sum())

        print( pars, crit )
        self.r0 = r0
        return crit

    def invokeMin(self):
        minimize(self.getNoll, self.r0, method='Nelder-Mead')

    def plotNoll(self):
        plt.figure( 'Noll vs. data' )
        plt.clf()
        plt.plot( range(2, self.n_zerns+2), self.in_spec, '.:', label='data' )
        plt.plot( range(2, self.n_zerns+2), self.mynoll, "k", label="Noll")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Noll index');
        plt.ylabel('Power, [nm$^2$]')
        plt.title( 'r$_0$={:.2e} [m]'.format(self.r0) )
        plt.grid(True)
        plt.tight_layout()


    def inp(self):
        time_step = self.pxl_scale/self.wind_speed # This is timestep, as one pixel added on each iteration
        stencil_length_factor = 16
        n_tests = 25 # 16
        n_scrns = 100
        z_coeffs_inf_f = np.zeros((n_tests, n_scrns, self.n_zerns+1))
        Zs = aotools.zernikeArray(self.n_zerns+1, self.nx_size)
        for n in tqdm(range(n_tests)):
            phase_screen = PhaseScreenKolmogorov(self.nx_size, self.pxl_scale, self.r0, self.L0, stencil_length_factor=stencil_length_factor)
            for i in range(n_scrns):
                # Go in all directions
                phase_screen.add_row()
                if n%4 == 0:
                    scrn = phase_screen.scrn
                elif n%4 == 1:
                    scrn = phase_screen.scrn.T
                elif n%4 == 2:
                    scrn = phase_screen.scrn[::-1]
                else:
                    scrn = phase_screen.scrn[::-1].T

                z_coeffs_inf_f[n, i] = (scrn * Zs).sum((-1, -2))/(Zs[0].sum())

        z_vars_inf_f = z_coeffs_inf_f.var(1)

        ##Example Phase Screen
        #stencil_length_factor = 32
        #phase_screen = PhaseScreenKolmogorov(self.nx_size, self.pxl_scale, self.r0, self.L0, stencil_length_factor=stencil_length_factor)
        #print(phase_screen.scrn.shape)
        #plt.figure()
        #plt.imshow(phase_screen.scrn)
        #cbar = plt.colorbar()
        #cbar.set_label('Wavefront deviation (radians)', labelpad=8)

        #n_tests = 25 # 16
        #n_scrns = 100
        ##Zernike Breakdown of Phase Screen
        #Zs = aotools.zernikeArray(self.n_zerns+1, self.nx_size)
        #time_step = self.pxl_scale/self.wind_speed # This is timestep, as one pixel added on each iteration
        #z_coeffs_inf_f = np.zeros((n_tests, n_scrns, self.n_zerns+1))
        #for n in tqdm( range(n_tests) ):
            #phase_screen = PhaseScreenKolmogorov(self.nx_size, self.pxl_scale, self.r0, self.L0, stencil_length_factor=stencil_length_factor)
            #for i in range(n_scrns):
                ## Go in all directions
                #phase_screen.add_row()
                #if n%4 == 0:
                    #scrn = phase_screen.scrn
                #elif n%4 == 1:
                    #scrn = phase_screen.scrn.T
                #elif n%4 == 2:
                    #scrn = phase_screen.scrn[::-1]
                #else:
                    #scrn = phase_screen.scrn[::-1].T

                #z_coeffs_inf_f[n, i] = (scrn * Zs).sum((-1, -2))/(Zs[0].sum())

        #z_vars_inf_f = z_coeffs_inf_f.var(1)
        self.in_spec = z_vars_inf_f.mean(0)[1:] * wavelength/(2*np.pi)
        self.getNoll( pars=[self.r0] )



def drawSHmap( cent, DRAW=True, ax='xy' ):
    '''
    draws a 2D centroid map, the centroids must be Alified
        ax='xy' draws the ||r||
        ax='f'  draws the flux
    '''
    cm = wfsObj.samask
    subapmap = cm.copy()
    rr=subapmap.shape[0]
    k=0
    for i in range( rr ):
        for j in range( rr ):
            if cm[i][j] == 1:
                if ax == 'xy':
                    subapmap[i][j] = np.sqrt( cent[k]**2 + cent[k+1]**2 )
                    k+=2
                elif ax == 'f':
                    subapmap[i][j] = cent[k]
                    k+=1

    if DRAW:
        plt.figure('SH map')
        plt.clf()
        #plt.imshow( 10**np.log10( subapmap -0.0000001) )
        plt.imshow( ( subapmap/cm ) )#, clim=[0, .1] )
        plt.colorbar()
        #plt.pause(0.01)

    return ( subapmap )

def stdAmp( img ):
    '''
    returns the RMS amplitude of img in nm, elMasq is applied
    '''
    mask  = zern( 1, img.shape[0] )

    return (np.ma.masked_invalid(img[mask == 1]).std())

def dealistr( alistr ):
    dealistrd = alistr.copy()
    if alistr.ndim == 1:
        middle = dealistrd.shape[0]//2
        dealistrd[:middle] = alistr[::2]
        dealistrd[middle:] = alistr[1::2]
    elif alistr.ndim == 2:
        middle = dealistrd.shape[1]//2
        dealistrd[:,:middle] = alistr[:,::2]
        dealistrd[:,middle:] = alistr[:,1::2]

    return dealistrd

def wf2D(wf1D, DRAW=True, im=None, mask=1):
    '''
    Transform 1D wf vector in 2D array.
    input:
        wf1D - 1D vector to be mapped onto illuminatedCorners
        DRAW - boolean if to draw or not
        im   - matplotlib image object in case animation is displayed
    returns:
        2D array of the WF
    '''

    wf2D=np.zeros(gO.n_)
    wf2D.ravel()[gO.illuminatedCornersIdx]=wf1D
    if DRAW:
        if im is None:
            plt.figure('wf')
            plt.clf()
            plt.imshow(wf2D/mask)
            plt.title('RMS={:4.1f}, PtV={:4.1f}'.format( wf1D.std(), wf1D.ptp()))
            plt.colorbar()
            plt.pause(0.01)
        else:
            im.set_array( wf2D/mask )
            plt.title( 'RMS={:4.1f} nm'.format( wf1D.std() ) )
            plt.pause( 0.001 )

    return wf2D*mask


gO=gOT1( wfsObj.samask ) #ABBOT initialisation
gM=gO.returnOp()         #ABBOT initialisation
def StoWF( cent, DRAW=False, gM_i=None, deali=True ):
    '''
    gets centroids returns wavefront
    '''

    slopesV = cent*SH_scaling*Mag_factor
    #print( SH_scaling, ' applied!!!!!!!!!!!!!!' )
    if deali:
        #middle = slopesV.shape[0]//2
        #slopesV[:middle] = cent[::2]*SH_scaling*Mag_factor
        #slopesV[middle:] = cent[1::2]*SH_scaling*Mag_factor
        slopesV = dealistr( cent )*SH_scaling*Mag_factor
    #valid_subap_map=wfsObj.samask

    if gM_i is None:
        if not os.path.exists( homedir+'/dev/gM_i_30x30.fits'):
            print('gradient ma3x NOT FOUND, recalculating...')
            gM_i=np.linalg.pinv( gM )
        else:
            gM_i = fits.getdata( homedir+'/dev/gM_i_30x30.fits' )

    if slopesV.ndim == 1:
        wfV=gM_i.dot(slopesV)
        wf2D=np.zeros(gO.n_)
        wf2D.ravel()[gO.illuminatedCornersIdx]=wfV
        if DRAW:
            plt.figure('wf')
            plt.clf()
            plt.imshow(wf2D)
            plt.title('RMS={:3.1f}, PtV={:3.1f}'.format( stdAmp(wf2D), wf2D.ptp()))
            plt.colorbar()
            plt.pause(0.01)
        return wf2D, wfV

    elif slopesV.ndim == 2:
        wfV=np.matmul( gM_i, slopesV.T)
        #shpe = list(gO.n_)#.copy()
        #tmp=np.zeros(gO.n_)
        #shpe.append( slopesV.shape[0] )
        #wf2D=np.zeros( shpe )
        ##print( shpe, wf2D.shape )
        #for i in range( slopesV.shape[0] ):
            ##print(wf2D[:,:,i].shape, wfV[i].shape )
            #tmp.ravel()[gO.illuminatedCornersIdx] = wfV[:,i]#.copy()
            #wf2D[:,:,i] = tmp.copy()

        return wfV.T# wf2D.T# gM

def WFtoS( wf, nSA=52, DRAW=False ):
    '''
    returns slopes from a wavefront
    '''
    valid_subap_map=mypy.SAflag(nSA).reshape( (nSA,nSA) )
    gO=gOT1( valid_subap_map )
    gM=gO.returnOp()

    #this below matches the darc Sub Aperture settings in the gradient space
    a=26.7                                       # this
    masq = ellipMask( a, a, 51, 51, 25.5, 25.5 ) # and this


    wfv     = wf[masq==1]
    synslop = ( wfv ).dot( gM.T )/SH_scaling

    if DRAW:
        plt.figure('syn')
        plt.clf();
        plt.plot( wfv )
        plt.pause(0.1)

        plt.figure('wf')
        plt.clf()
        plt.subplot( 1,3,1 )
        plt.imshow( wf )
        plt.title('In, RMS{:3.2f}'.format( stdAmp( wf, wf.shape[0] ) ) )
        plt.subplot( 1,3,2 )
        recWF = StoWF( synslop )
        plt.imshow( recWF  )
        plt.title('Out, RMS{:3.2f}'.format( stdAmp( recWF[1:-1,1:-1], recWF.shape[0]-2 ) ) )
        plt.subplot( 1,3,3 )
        plt.imshow( recWF[1:-1,1:-1] - wf )
        plt.title('Diff, RMS{:3.2f}'.format( stdAmp( recWF[1:-1,1:-1] - wf, wf.shape[0] ) ) )
        plt.pause(0.1)

        plt.figure('decomposition')
        plt.plot( ( decompoZergOld( StoWF( synslop ) ) ), '.:' )
        plt.pause(0.1)

    return synslop

def sqSum(zdc):
    if zdc.ndim == 1:
        return ((zdc**2).sum())**0.5
    elif zdc.ndim == 2:
        return ((zdc**2).sum(0))**0.5

def decompoZerg(wf, Zorder=10, Zsize=53, mask=None):
    '''
    Orthonormalisation of pixelised Zernike distributions and decomposition of a wave front against it
    Input parameters:
        wf      - the wavefront to be decomposed
        Zorder  - the highest order of Zernike to be included
        Zsize   - the size of the discretisation matrix
        mask    - which elements of the matrix to be included in the returned one

    returns the Zernike decomposition of a pixelised WF
    '''

    if mask is None:
        mask = zern( 1, Zsize )
    else:
        mask[mask>0]=1

    zn      = zern( 1, Zsize )
    sf_norm = (1/(zn*zn).sum())**0.5

    if Zorder == 10:
        zMatx = zMatx10_31
    elif Zorder == 50:
        zMatx = zMatx50_31
    elif Zorder == 100:
        zMatx = zMatx100_31
    #elif Zorder == 1000:
        #zMatx = zMatx100_31
    else:
        print('zMatrx generation')
        zMatx = zMatxGen( Zorder, Zsize, mask )
    #for i in range( 2, Zorder + 1):
        ##zMatx[i-2] = sf_norm*( rotFlip( zern( i, Zsize ), 180, True) )[ mask==1 ]
        #zMatx[i-2] = zern( i, Zsize )[ mask>0 ]

    #print( mask.shape )
    if wf.ndim == 2:
        #print('ndim2')
        if wf.shape[0] == wf.shape[1]:
            #print('=')
            return np.dot( zMatx, wf[ mask>0 ])
        else:
            #print('!=')
            return np.matmul( zMatx, wf)
    elif wf.ndim == 1:
        #print('ndim1')
        return np.dot( zMatx, wf)
    else:
        #print('ndim: ', wf.ndim)
        return np.matmul( zMatx, wf)

def zMatxGen( Zorder=10, Zsize=31, mask=None):

    if mask is None:
        mask = zern( 1, Zsize )
    else:
        mask[mask>0]=1

    zMatx =  np.zeros( ( Zorder-1, int( mask.sum() ) ) )
    for i in range( 2, Zorder + 1):
        #zMatx[i-2] = sf_norm*( rotFlip( zern( i, Zsize ), 180, True) )[ mask==1 ]
        zMatx[i-2] = zern( i, Zsize )[ mask>0 ]

    return np.matmul( np.linalg.inv( np.matmul( zMatx, zMatx.T) ), zMatx )


zMatx10_31   = zMatxGen( 10,   31, gO.illuminatedCorners )
zMatx50_31   = zMatxGen( 50,   31, gO.illuminatedCorners )
zMatx100_31  = zMatxGen( 100,  31, gO.illuminatedCorners )
#zMatx1000_31 = zMatxGen( 1000, 31, gO.illuminatedCorners )

def monSpider():
    path = homedir+'/dev/turb/data/'
    fName = 'avg_cent_202101{:02d}.fits'

    lastDay = 19
    skipDays=[6,15]

    cents  = np.zeros( ( lastDay - len(skipDays)+1, 4320, 1432) )
    wfMass = np.zeros( ( lastDay - len(skipDays)+1, 4320, 777) )
    Zdec   = np.zeros( ( lastDay - len(skipDays)+1, 4320, 99) )
    sk = 0
    for i in range( 0, lastDay):
        if i in skipDays:
            sk += 1

        if fits.getdata( path+fName.format(i+1) ).shape[0] == 4320:
            cents[i-sk] =( fits.getdata( path+fName.format(i+1) ) )
            wfMass[i-sk] =( StoWF( cents[-1] - cents[-1].mean(0) ) )
            Zdec[i-sk] = ( decompoZerg( wfMass[-1].T, 100, 31 ) ).T
            print( i-sk, path+fName.format(i+1), cents[i-sk].shape, wfMass[i-sk].shape, Zdec[i-sk].shape )
        elif fits.getdata( path+fName.format(i+1) ).shape[0] == 4321:
            cents[i-sk] =( fits.getdata( path+fName.format(i+1) ) )[:4320]
            wfMass[i-sk] =( StoWF( cents[-1] - cents[-1].mean(0) ) )
            Zdec[i-sk] = ( decompoZerg( wfMass[-1].T, 100, 31 ) ).T
            print( i-sk, path+fName.format(i+1), cents[i-sk].shape, wfMass[i-sk].shape, Zdec[i-sk].shape )


    return ( cents ), ( wfMass ), ( Zdec )

def monPlot(datata = '20210131', path='/home/cfai/dev/turb/data/202101/'):

    tempLabels = ['wire',
                  'bench',
                  'WFS focus',
                  'WFS pole',
                  'duct WFS',
                  'duct wire'
                    ]

    avg_name = path+'avg_{:s}_{:s}.fits';
    temp = fits.getdata( avg_name.format( 'Temp', datata ));
    cents = fits.getdata( avg_name.format('cent', datata ))

    xax = np.linspace( 0, cents.shape[0]/6/60., cents.shape[0])
    wf = StoWF( cents - cents[:100].mean(0) )
    zc = decompoZerg( wf.T, 100, 31 )
    plt.figure('monitor')
    plt.clf()

    fig, (axT, axWF) = plt.subplots(2,1, num='monitor', sharex=True)
    axT.set_title(datata)
    for i, Tem in enumerate( temp.T ):
        axT.plot( xax, Tem, label=tempLabels[i] )
    axT.set_xlim(-1.5, xax.max() )
    axT.set_xlabel( 'Time, [h]')
    axT.set_ylabel( 'Temperature, [C]')
    axT.legend()
    #plt.subplot(2,1,2)
    axWF.plot( xax, ( zc[0] ), '.:', label='Tip', markersize=3., linewidth=0.2 )
    axWF.plot( xax, ( zc[1] ), '.:', label='Tilt', markersize=3., alpha=0.85, linewidth=0.2 )
    axWF.plot( xax, sqSum( zc[2:] ), '.:', label='WF RMS Z$_{4+}$', markersize=2. )
    axWF.set_xlim(-1.5, xax.max() )
    axWF.set_xlabel( 'Time, [h]')
    axWF.set_ylabel( 'RMS WF, [nm]' )
    axWF.legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01)

    plt.figure('spectr')
    plt.clf()
    plt.plot( np.arange(4,101,1), ( zc[2:] ).std(1), '.:', label='HO' )
    plt.xlabel('Zernike, [Noll]'); plt.ylabel( 'STD, [nm]');
    plt.grid(True)
    plt.tight_layout()

    plt.figure('spectr2D')
    plt.clf()
    plt.imshow( abs(zc[2:]), aspect='auto', origin='lower', extent=(0,cents.shape[0]/6/60., 3.5,100.5) )
    plt.ylabel('Zernike, [Noll]'); plt.xlabel( 'Time, [frames]');
    plt.tight_layout()


    return zc, cents, temp, wf.mean(0)

def spectKolmoZerniFit(r0=0.005, L0=0.15, n=10000, comp=None):
    ilC_ones = gO.illuminatedCorners
    N=int(n)
    nsa=31
    rho=0.05/nsa#0.0016129032258064516#50mm/31 in [m]
    D=rho*nsa
    kd_dz = np.zeros((N,99) )
    kd = aotools.PhaseScreenKolmogorov( nsa, rho, r0, L0 )
    for i in trange( N ):
        scrn = kd.add_row()
        kd_dz[i] = decompoZerg( scrn-scrn.mean(), 100, nsa, ilC_ones )*wavelength/(2*np.pi)
    print(r0, sqSum( kd_dz[:,2:].std(0) ) )
    plt.figure('spect')
    plt.clf()
    plt.plot( np.arange(4,101,1), (kd_dz[:,2:].std(0)), '.:', label='synthetic'.format(r0))
    if comp is not None:
        plt.plot( np.arange(4,101,1), (comp), '.:', label='measured'.format(r0))
        plt.legend()
    plt.title(r"D={:2.1e}, $\rho$={:2.1e}, $r_0$={:3.1e}, [m]".format(D,rho,r0) );
    plt.xlabel('Zernike, [Noll]'); plt.ylabel( 'STD, [nm]');
    plt.grid(True)

    return kd_dz


def spectKolmoZerni(L0=25, r=0.135, nsa=31,n=10000):
    ilC_ones = gO.illuminatedCorners#.copy()
    N=int(n)
    kd_dz = np.zeros((3,N,99) )
    #r0=(0.06, 0.125, 0.2)
    r0=(0.06, 0.125, 0.2)
    #L0=25
    rho=r
    D=rho*nsa
    for j, r in enumerate(r0):
        kd = aotools.PhaseScreenKolmogorov( nsa, rho, r, L0 )
        for i in trange( N ):
            scrn = kd.add_row()
            kd_dz[j][i] = decompoZerg( scrn-scrn.mean(), 100, nsa, ilC_ones )*wavelength/(2*np.pi)
        print(r, sqSum( kd_dz[j][:,2:].std(0) ) )
    plt.figure('spect')
    plt.clf()
    plt.plot( np.arange(4,101,1), (kd_dz[0][:,2:].std(0)), '.:', label='$r_0$={:4.3}'.format(r0[0]))
    plt.plot( np.arange(4,101,1), (kd_dz[1][:,2:].std(0)), '.:', label='$r_0$={:4.3}'.format(r0[1]))
    plt.plot( np.arange(4,101,1), (kd_dz[2][:,2:].std(0)), '.:', label='$r_0$={:4.3}'.format(r0[2]))
    plt.title(r"D={:2.1f}, $\rho$={:4.2f} [m]".format(D,rho) );
    plt.xlabel('Zernike, [Noll]'); plt.ylabel( 'STD, [nm]');
    plt.grid(True)
    plt.legend()

    plt.figure('spect2D', figsize=(9, 4))

    #plt.suptitle(r"D={:2.1f}, $\rho$={:4.2f}, $L_0$={:3.0f}, [m]".format(D,rho,L0) );
    plt.suptitle(r"D={:2.1f}, $\rho$={:4.2f} [m]".format(D,rho) );
    plt.subplot(1,3,1)
    plt.imshow( abs(kd_dz[0][:,2:]).T, aspect='auto', origin='lower' )
    plt.ylabel('Zernike, [Noll]'); plt.xlabel( 'Time, [frames]');
    plt.title('$r_0$={:4.3}'.format(r0[0]))
    plt.subplot(1,3,2)
    plt.imshow( abs(kd_dz[1][:,2:]).T, aspect='auto', origin='lower' )
    plt.xlabel( 'Time, [frames]');
    plt.title('$r_0$={:4.3}'.format(r0[1]))
    plt.subplot(1,3,3)
    plt.imshow( abs(kd_dz[2][:,2:]).T, aspect='auto', origin='lower' )
    plt.xlabel( 'Time, [frames]');
    plt.title('$r_0$={:4.3}'.format(r0[2]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.864)

    return kd_dz

def freqSpect(d=10., var=None, ax='f', clf=True):
    '''
    returns frequency spectrum plot of input
    '''
    if var is None:
        print('input is None, please provide some...')
        return -1

    freq = np.fft.fftfreq( var.shape[0], d)

    plt.figure('freqSpec')
    if clf:
        plt.clf()

    print( d, var.shape )
    daspect = np.fft.fft( var, norm='ortho' )
    if ax == 'f':
        plt.plot( abs( freq )[4:-4], abs( daspect )[4:-4], ',:', alpha=0.5 )
        plt.yscale('log')
        plt.xlabel('Frequency, [Hz]')
    elif ax == 'p':
        plt.plot( 1/abs( freq )[4:-4], abs( daspect )[4:-4], '.:', alpha=0.5 )
        plt.xlabel('Period, [s]')

    plt.ylabel('Amplitude, [nm$\cdot Hz^{-1/2}$]')
    plt.tight_layout()
    #plt.pause(0.1)

    return abs( daspect )

def normDifferential(var):
    return np.diff( var.std(0)/var.std(0)[0] )

def freqSpecLong(var=None, Dt=int(86400/2), dt=0.01):

    if var is None:
        print('var should not be None')
        return

    if var.shape[0] <= Dt:
        print(var.shape[0], 'too short dataset')
        return

    plt.figure('freqSpec')
    plt.clf()
    freqSpec = np.zeros( (int(var.shape[0]/Dt), Dt) )
    for i in range( int(var.shape[0]/Dt) ):
        freqSpec[i] = freqSpect( dt, var[int(i*Dt):int((i+1)*Dt)], ax='f', clf=False )
        plt.xscale('log')

    plt.figure('long')
    plt.clf()
    plt.imshow( (freqSpec[:,2:int(Dt/2)]), aspect='auto', extent=[1/(2*dt), Dt/(4*dt),60*freqSpec.shape[0]/dt,0], cmap='terrain', interpolation='bicubic' )
    #plt.xlim(5.5e-5,.4)
    plt.xscale('log');
    plt.xlabel('Frequency, [Hz]');
    plt.ylabel('Time, [min]')

    #plt.ylabel('February')
    #dateL = np.arange(4, 23, 1 )
    #x_positions = np.linspace(0.0, freqSpec.shape[0]-1, (dateL.shape[0]) )
    #plt.yticks(x_positions+0.45, dateL)
    plt.tight_layout()
    plt.pause(0.1)

def animateWF(WF=None, step=100, t=(1000,2000), mask=133):

    if WF is None:
        print('you need to provide WF to animate')
        return

    masq = mypy.circMask( 30, 30, mask )

    draw = []

    for i in trange( t[0], t[1] ):#( 24236, 24339 ):
        draw.append( masq* wf2D( WF[i], DRAW=False ).T )

    clim = ( np.asarray( draw ).min()*0.9, 0.9*np.asarray( draw ).max())
    plt.figure('anim')
    plt.clf()
    im = plt.imshow( draw[0], clim=clim, origin='lower' )#, cmap=plt.cm.jet )
    plt.colorbar()
    for i in trange(len( draw )):
        im.set_array( draw[i] )
        plt.title('{:6.2f}s, $\sigma$={:4.1f}nm'.format( (t[0]+i+1)/100, draw[i].std() ) )
        plt.pause(0.001)

    return np.asarray( draw )

def xcor(imgs=None, mask=133, gain=1., filt=130., thr=.1):
    if imgs is None:
        print('input cannot be None')
        return
    else:
        imgs = (imgs-imgs.mean())*mypy.circMask( 30, 30, mask )

    nri = imgs.shape[0]

    comfilt = mypy.circMask( 30, 30, filt )

    plt.figure('x-cor')
    plt.clf()
    fig, axs = plt.subplots(2, nri, num='x-cor' )

    auto = aotools.centroiders.cross_correlate( imgs[0], imgs[0] )
    avg = imgs.mean(0)
    for i, im in enumerate( imgs ):
        axs[0, i].imshow( im, origin='lower' )
        #axs[0, i].imshow( im, clim=(-400,500), origin='lower' )
        axs[0, i].set_title( 'frame# {:d}'.format(i+1))
        #xcor = aotools.centroiders.cross_correlate( im, imgs[0])*comfilt

        if i == 0:
            xcor = aotools.centroiders.cross_correlate( im, imgs[0])*comfilt
        else:
            xcor = aotools.centroiders.cross_correlate( imgs[i-1], imgs[i])*comfilt

        xcorcom = aotools.centroiders.centre_of_gravity( xcor, threshold=thr )-15
        axs[1, i].imshow( xcor, origin='lower' )
        axs[1, i].set_title( '{:4.2f}/{:+4.2f}'.format( (xcorcom[0])/gain, (xcorcom[1])/gain ) )
        print(  xcorcom )
    plt.tight_layout()

def xcor_sequence():
    kdxcor = np.zeros( (20, 2) )
    thr=0.81
    for i in range( kdxcor.shape[0] ):
        if i == 0:# kdxcor.shape[0]:
            kdxcor[i] = (1.,1.)
            pass#break
        else:
            xcor = aotools.centroiders.cross_correlate( kd2dV[i-1], kd2dV[i])
            kdxcor[i] = aotools.centroiders.centre_of_gravity( xcor, threshold=thr )-15
            kdxcor[i][1]/=(kdxcor[i-1][1])
        print(kdxcor[i-1])

def powlaw(x, a, b) :
    return a * np.power(x, b)
def linlaw(x, a, b) :
    return a + x * b

def curve_fit_log(xdata, ydata) :
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)
