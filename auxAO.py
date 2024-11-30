from imp import reload as re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import time
import os
import datetime
import time
import yaml #as yaml
import pprint
import aotools
import multiprocessing as mp

from numpy.random import default_rng
from matplotlib.patches import Rectangle
from aotools import zernike_noll as zern
from aotools import circle
from astropy.io import fits
from collections import OrderedDict as od
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from ABBOT.gradientOperator import gradientOperatorType1 as gOT1
from ABBOT.gradientOperator import laplacianOperatorType1 as LOT1
from scipy.signal import fftconvolve as fftc
import numpy.ma as ma

from lapy import lapy


ordsmag = {'n': 1e-9,
           'u': 1e-6,
           'm': 1e-3,
           'k': 1e3,
           'M': 1e6,
           'G': 1e9
    }

class Zern:
    def plotZern(self, z=None, figN='plot'):
        '''
        plots the Zernike amplitudes vs. their Noll notation
        '''
        if z is None:
            print('input missing, dunno what to do, aborting...')
            return -1j

        xtitl = 'Zernike, [Noll]'
        ytitl = 'Amplitude, [nm]'
        figN  = 'Zernikes'

        plt.figure( num=figN, clear=1, figsize=(6,3))
        # if len(z) >= 2:
        #     print('auto generating zaxis...')
        #     zax = np.arange( 1, z.shape[0]+1., 1)
        #     for p in z:
        #         plt.plot( zax, p, '.:')
        if z.ndim == 1:
            print('auto generating zaxis...')
            zax = np.arange( 1, z.shape[0]+1., 1)
            plt.plot( zax, z, '.:')
        else:
            print('input format unknown, aborting')
            return -1j

        plt.grid(True)
        plt.xlabel(xtitl)
        plt.ylabel(ytitl)

    def Zaber_maker(self, vZamp=None, DRAW=False):
        # TODO ATTENTION
        # this function need to be abstracted to lapy.py
        # TODO

        if vZamp is not None:
            self.vZamp=vZamp.copy()

        if self.vZamp is None:
            self.Zrandlist( self.nZern )
            #print("Naught to do, generate the zList first,\n\taborting...")
            #return -1
        # vZamp = self.vZamp
        if self.vZamp.ndim == 1:
            # print('nDim==1')
            zNoll = np.linspace(1, self.vZamp.shape[0]+1,  self.vZamp.shape[0])
            vZamp = np.asarray([zNoll, self.vZamp ]).T
        elif self.vZamp.ndim == 2 and self.vZamp.shape[1] == 2:
            vZamp = self.vZamp
        else:
            print('invalid vZamp format, aborting...')
            return -1j

        aberV=np.matmul( vZamp.T[1], self.Zlst_2D[:vZamp.T[1].shape[0],self.Zmask>0] )
        aber = self._1D_2D( aberV, self.rpup)
        aber[self.rpup>0] -= aber[self.rpup>0].mean()
        if DRAW:
            #print(vZamp)
            self.draw1( aber, mask=self.rpup, figN='aber' )

        self.aber=(aber)
        # return aber

    def Zdecompo(self, wf, Zorder=42, aberSz=None, zMatx=None, mask=None):
        '''
        receives a 2D WF as input
        returns a 1D array with Zernike decomposition of the WF
        '''

        try:
            if self.Zmask is None:
                self.Zmask = circle( self.wfs_size/2, self.wfs_size )
            elif mask is not None:
                self.Zmask = mask
        except AttributeError:
            if mask is not None:
                self.Zmask = mask
            else:
                self.Zmask = circle( self.wfs_size/2, self.wfs_size )

        if zMatx is None:
            self.zList( Zorder )
        else:
            self.zMatx=zMatx

        #plt.figure('sancheck', clear=1)
        #plt.imshow( self.Zmask )

        if wf.ndim == 1:
            self.decompdZ = np.dot( self.zMatx, wf)
            wf = wf - (wf[self.Zmask>0]).mean()
            #return self.decompdZ
        elif wf.ndim == 2:
            #print( 'wf.ndim == 2')
            if wf.shape[0]==wf.shape[1]:
                self.decompdZ = np.dot( self.zMatx, wf[ self.Zmask>0 ])
                #return self.decompdZ
            elif wf.shape[0]==self.zMatx.shape[1]:
                self.decompdZ = np.matmul( self.zMatx, wf)
            elif wf.shape[1]==self.zMatx.shape[1]:
                self.decompdZ = np.matmul( self.zMatx, wf.T)


    def zList(self, nZern=None):
        '''
        creates a list of 2D Zernike distributions with the given parameters
        populates a numpy array
        '''
        if nZern is not None:
            self.nZern=nZern

        pist=zern(1,self.spupsz).astype(int)
        self.Zlst_2D = np.zeros( (self.nZern, self.spupsz, self.spupsz) )
        for i in range( 1, self.nZern+1):
            zz = zern( int(i), self.spupsz)
            if i != 1:
                zz[pist>0] -= zz[pist>0].mean()
                zz *= 1/(zz[pist>0].std())

            self.Zlst_2D[i-1] = (zz).copy()
            self.Zmask = pist.copy()
            # lapy.insertImage( zz, endShape=[self.wfs_size]*2, vpos=self.vp, hpos=self.hp )

        self.zMatxGen()

    def zMatxGen(self):
        '''
        tries to open the zMatrix needed for Zernike decomposition, if file not found it generates it and saves it in location defined in the yaml file
        '''
        sz = np.sum(self.Zmask==1)
        try:
            if self.nZern in self.zMCollection:
                if sz in self.zMCollection[self.nZern]:
                    self.zMatx = self.zMCollection[self.nZern][sz]
                    return
        except AttributeError:
            print('zMat NOT in collection...')

        self.zmx_path = 'conf/zMx_{:03d}z_{:03d}s.fits'
        self.zmx_path = self.zmx_path.format(self.nZern, sz)
        try:
            self.zMatx = fits.getdata( self.zmx_path )
            self.zMCollection[self.nZern]={}
            self.zMCollection[self.nZern][sz] = self.zMatx.copy()
            print('zMx opened from: '+self.zmx_path)
        except FileNotFoundError:
            zMask = lapy.insertImage(self.Zlst_2D[:,self.Zmask>0], endShape=[self.wfs_size]*2, vpos=self.vp, hpos=self.hp )
            print(f'File not found, generating and inverting now a matrix of size {self.nZern}x{sz}, have patience...')
            self.zMatx = np.dot( np.linalg.inv( np.dot( zMask, zMask.T) ), zMask)

            fits.writeto( self.zmx_path, self.zMatx )
            self.zMCollection[self.nZern]={}
            self.zMCollection[self.nZern][sz] = self.zMatx.copy()

        self.zMatx[0]*=0

    def Zrandlist(self, N=None, DRAW=False, ask=1):
        if N is None:
            N=self.nZern
        else: self.nZern = N

        if ask:
            try:
                if self.vZamp is None:
                    print('New list with randZerns will be generated...')
                else:
                    ans = input('vZamp exists, replace? [Y/n] ')
                    if ans=='Y' or ans=='y':
                        print('New list with randZerns will be generated...')
                    else:
                        print('Naught done, aborting.')
                        return -1j
            except AttributeError:
                pass

        self.vZamp = np.zeros( (N,2) )
        rng = default_rng()

        #self.vZamp.T[0] = np.sort(rng.choice( np.arange(2,2*N), size=N, replace=False))
        self.vZamp.T[0] = np.arange(1, N+1)
        self.vZamp.T[1] = (rng.choice( np.arange(-1,1,1e-4), size=N, replace=False))

        self.vZamp.T[1][:3] = 0

        if DRAW:
            self.Zaber_maker( DRAW=1 )

class vis(Zern):
    '''
    A class auxiliary to (any) DM class
    '''
    def __init__(self, **kwargs):
        def_kw = {
            'nAct':   820,   # total number of actuators
            'pitch':  1.5,   # actuator separation [mm]
            'diamtr': 45 ,   # diameter of mirror [mm]
            'nAct1D': 32 ,   # number of actuators acros the diameter

            'fCoeff': 1.015, # preserving the 32/820 ratio in circle()
            'infFun': None , # path to file with influnce funciton(s)
            'synthF': {
                'type': 'Gauss',
                'sigm': 0.1    ,
                }
            }
        # initialize all allowed keys to false
        self.__dict__.update((key, value) for key, value in def_kw.items())
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in def_kw)
        #print(self.__dict__)
        self.__dict__.update(kwargs)
        # print(self.__dict__)


    def draw1(self, img, figN='one', mask=None, clip=None, ax=0):
        if mask is None:
            mask = circle( img.shape[0]//2, img.shape[0] )
        #     title = '$\sigma=$None'
        # else:
        #     img = img/mask
        #     title = '$\sigma=${:.3f}'.format( img[mask==1].std() )
        title = '$\sigma=${:.3f}'.format( img[mask==1].std() )
        if clip is not None:
            img[img>=clip]=clip
        plt.figure(figN, clear=1, figsize=[4,3])
        if not ax: plt.axis('off')
        plt.imshow( img )#/mask )
        plt.title( title )
        plt.colorbar()
        plt.pause(0.01)

    def _1D_2D(self, _1D=None, _2D=None):
        if _1D is None or _2D is None:
            print('ERR: Invalid CAM input, aborting...')
            return -1
        elif _1D.shape[0] != _2D.sum():
            print("ERR: shape mismatch...")

        nsub = _2D.shape[0]
        im   = np.zeros(_2D.shape)
        i = 0
        for v in range(nsub):
            for h in range(nsub):
                #If subap is valid, add to list
                if _2D[v,h] > 0:
                    im[v,h] = _1D[i]
                    i+=1
        return im

    def streaIm(self, source=None, fign='streaIm'):
        '''
        streams over Imshow 20Hz
        '''
        if source is None:
            print('source is None, cannot stream None, aborting...')
            return -1j

        self.onoff=True
        plt.ion()
        fig = plt.figure(fign, clear=True)
        fig.canvas.mpl_connect('close_event', self.close)
        fig.canvas.mpl_connect('key_press_event', self.press)

        ax = fig.add_subplot(111)
        a=plt.imshow(source(), cmap='turbo')
        plt.title('press \'q\' to close')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        while self.onoff:
            data = source()
            dmin = data.min()
            dmax = data.max()

            a.set_data(data)
            a.set_clim(dmin,dmax)

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(1/20.)

    def drawNM(self, data=None, N=1, M=None, figN='NM', ttl=None, masq=None, scale=0, fieldR=None):
        '''
        data - [heterogeneous list]
        N    - number of rows
        M    - number of collumns, if M is None: M=N
        masq - boolean array applied to all
        '''
        if data is None:
            print('data is None, cannot process None, aborting...')
            return -1j
        if M is None: M=N
        if fieldR is not None: field=aotools.circle(fieldR, N)

        tot = N*M
        fsV = 3.33*N
        fsH = 3.33*M

        try:
            data = np.asarray(data)
            dmin = data.min()
            dmax = data.max()
        except ValueError:
            print("ValueError: aborting due to inhomogeneous array?..")

        if masq is None:
            masq = circle(data.shape[1]/2,data.shape[1])

        plt.figure(figN, figsize=(fsV,fsH), clear=True)
        i=0
        if N == 1:
            plt.imshow(data/masq)
            if scale: plt.clim(dmin,dmax)
            else: plt.colorbar()
            if ttl is not None:
                plt.title(ttl)
            return

        for H in range(N):
            for V in range(M):
                plt.subplot( M, N, i+1 )
                plt.imshow(data[i]/masq)
                if scale: plt.clim(dmin,dmax)
                else: plt.colorbar()
                plt.axis('off')
                if ttl is not None:
                    plt.title(ttl[i])
                i+=1
            plt.pause(.1)

    def drawN(self, N=None, data=None, scale=1, name='', clip=None):
        if N is None or data is None:
            print('None as input, aborting...')
            return -1
        if scale:
            data = np.asarray(data)
            dmin = data.min()
            dmax = data.max()
        fsV = 3.33*N
        plt.figure(' N drawn '+name, figsize=(fsV,3), clear=True)
        for i, d in enumerate( data ):
            plt.subplot( 1, N, i+1 )
            plt.axis('off')
            if clip is not None:
                # cliping the amplitude to clip value
                d[d>=clip]=clip

            plt.imshow( d )
            if scale: plt.clim(dmin,dmax)
            plt.pause(0.1)
            if i+1 >= N:
                break

    def iActPlot(self, data=None, xax=None, fign='plots'):
        if data is None:
            print('ERR: input is None, naught to do, aborting')
        fig = plt.figure(fign, clear=True)
        fig.canvas.mpl_connect('close_event', self.close)
        ax = fig.add_subplot(111)

        if type(data) is list:
            nplots = len(data)
            shp=data[0].shape[0]
        elif type(data) is np.ndarray:
            if data.ndim>1:
                nplots = data.shape[0]
                shp=data.shape[1]
            else:
                nplots = 1
        else:
            print('ERR: unknown data type')
            return -1j

        if xax is None:
            xax = np.linspace(0,1,shp)

        if nplots > 1:
            for i in range(nplots):
                plt.plot( xax, data[i], '.:' )
        else:
            plt.plot(xax, data, '.:')

        return fig

    def iActive(self, img=None, ovlay=None, pl=None, fign='stream', ttl='title'):
        plt.ion()
        fig = plt.figure(fign, clear=True)
        fig.canvas.mpl_connect('close_event', self.close)
        ax = fig.add_subplot(111)
        a=plt.imshow( img, extent=[-1,1,-1,1] )

        if pl is not None:
            plt.plot(pl[0],pl[1],'ko', markersize=4.5,markerfacecolor="None")

        plt.axis('off')
        plt.colorbar()
        plt.title(ttl)
        if ovlay is not None:  o=plt.imshow( ovlay, extent=[-1,1,-1,1], alpha=0.2, cmap='gray_r' )
        plt.tight_layout()
        plt.show()

        return fig, a


    def press(self, event):
        if event.key == 'r':
            self.refs = self.WA.getFrames( self.shmN, 200 ).mean(0)
            print('resetting plot')
            self.maxSTD = 0.
            self.stdque = self.stdque*0
        elif event.key == 'P':
            self.onoff = False

    def close(self, event):
        '''
        handles keyboard events, in case 'q' key is pressed, the program is terminated
        '''
        self.onoff=False

    def showIm(self, img, fign='im', sagrid=None):
        fig, ax, = plt.subplots( 1, 1, figsize=(4,3.5), num=fign, clear=1)

        im = ax.imshow( img );
        fig.colorbar(im)
        fig.canvas.draw()
        fig.canvas.flush_events()

        if sagrid is not None:
            loc = pltck.MultipleLocator(base=sagrid)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)

            # Add the grid
            ax.grid(which='major', axis='both', linestyle='-')

        #plt.pause(0.1)

    def draw3(self, data, name='', ax=0):

        f, a = plt.subplots(nrows=1, ncols=3, num='three drawn', figsize=[10,3], clear=1)
        for i, cell in enumerate(a):
            cell.imshow( data[i] );
            if not ax:
                cell.axis('off')


        # plt.figure('three drawn'+name, figsize=(10,3), clear=True)
        # plt.subplot( 1,3,1 )
        #tmax=np.abs( one ).max()
        #tsum=(np.abs( one )).sum()
        #title = 'A={:0.0e}, $\sum$={:0.2f}'.format(tmax, tsum)
        # a[0]=plt.imshow( ( one ) );
        # # plt.axis('off')
        # #plt.title(title);
        # plt.pause(0.1)
        #
        # #tstd=ma.masked_invalid( (two/self.pup) ).sum()
        # # plt.subplot( 1,3,2 )
        # a[1]=plt.imshow( ( two ));
        # plt.axis('off')
        # #plt.title(f'$\sum$={tstd:.3f}');
        #
        # #title = '$\sigma^2$={:0.0f}'.format( ma.masked_invalid(three).var() )
        # # plt.subplot( 1,3,3 )
        # a[2]=plt.imshow( three )
        # # plt.axis('off')
        # #plt.title(title);
        f.tight_layout()

        return f, a

    def streamer(self, ani=None, fign='stream'):
        self.onoff=True
        self.HB=100
        if ani is not None:
            self.visObj = ani

        plt.ion()
        fig = plt.figure(fign, clear=True)
        fig.canvas.mpl_connect('close_event', self.close)

        ax = fig.add_subplot(111)
        a=plt.imshow( self.visObj.mean(0), cmap='gray_r')
        plt.axis('off')
        #plt.title('press \'q\' to close')
        #plt.colorbar()
        plt.tight_layout()
        plt.show()
        #a.set_clim(self.visObj.min() ,self.visObj.max())
        a.set_clim( 0, 4)

        #while self.onoff:
        c=0
        for data in self.visObj:
            #print( data.max() )
            #data=self.shm.get_data()
            #dmax=data.max()
            #dmin=data.min()
            a.set_data(data)
            #ax.set_title('#{:03d}, WFRMS={:3.1f}, [nm]'.format(c,data.std()))
            #a.set_clim(dmin ,dmax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1/self.HB)
            c+=1
            if self.onoff==False:
                break


    def StoWF(self, DRAW=False, name='', TT=False ):#, gM_i=None, gO=None):
        '''
        Slopes to Wavefront, gets centroids/slopes as input, returns wavefront, display is optional
        TT - when True T/T is subtracted from centroids
        NOTE works only with 1D TODO 2D
        '''

        if self.p_gO is None:
            self.makeGradOpP(self.fillMap)


        if self.f_gO is None and self.full:
            self.makeGradOpF( self.subapmap )
            gM_if = self.f_gM_i
            gOf   = self.f_gO

        gM_i = self.p_gM_i
        gOp   = self.p_gO


        if self.full: cf = self.wfs_centF.ravel('F')[::-1]#/self.calFFT
        cp = self.wfs_cent[self.sa_coord.T[4]>0].ravel('F')[::-1]#/self.calFFT

        if self.full:
            self.wfVf  = -1*gM_if.dot(cf)
            self.wfVf -= self.wfVf.mean()
            self.wf2Df = np.zeros(gOf.n_)/0
            i=0
            for y in range( gOf.n_[0]):
                for x in range( gOf.n_[1]):
                    if gOf.illuminatedCorners[x][y] > 0:
                        self.wf2Df[self.nSA1D-x][self.nSA1D-y] = self.wfVf[i]
                        i+=1

            if DRAW:
                plt.figure('wf full'+name)
                plt.clf()
                plt.imshow( self.wf2Df)
                plt.title('RMS={:3.1f}, PtV={:3.1f}'.format( self.wfVf.std(), self.wfVf.ptp()))
                plt.colorbar()
                plt.pause(0.01)

        self.wfVp  = gM_i.dot(cp)       # wavefront vector
        self.wfVp -= self.wfVp.mean()   # remove the piston leftover
        self.wf2Dp = np.zeros(gOp.n_) # spread it over the subaps
        i=0
        for y in range( gOp.n_[0]):
            for x in range( gOp.n_[1]):
                if gOp.illuminatedCorners[x][y] > 0:
                    self.wf2Dp[x][y] = self.wfVp[i]
                    i+=1

        if DRAW:
            gOmap = gOp.illuminatedCorners.copy()
            gOmap[gOp.illuminatedCorners>0]=1
            plt.figure('wf part'+name)
            plt.clf()
            plt.imshow( ma.masked_values(self.wf2Dp, 0) )
            plt.title('RMS={:3.1f}, PtV={:3.1f}'.format( self.wfVp.std(), self.wfVp.ptp()))
            plt.colorbar()
            plt.pause(0.01)

class DM_aux(vis):
    '''
    A class auxiliary to (any) DM class
    '''
    def __init__(self, **kwargs):
        def_kw = {
            'nAct':   820,   # total number of actuators
            'pitch':  1.5,   # actuator separation [mm]
            'diamtr': 45 ,   # diameter of mirror [mm]
            'nAct1D': 32 ,   # number of actuators acros the diameter

            'fCoeff': 1.015, # preserving the 32/820 ratio in circle()
            'infFun': None , # path to file with influnce funciton(s)
            'synthF': {
                'type': 'Gauss',
                'sigm': 0.1    ,
                }
            }
        # initialize all allowed keys to false
        self.__dict__.update((key, value) for key, value in def_kw.items())
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in def_kw)
        #print(self.__dict__)
        self.__dict__.update(kwargs)
        # print(self.__dict__)

        self.makeMask()

    def makeMask(self):
        print(self.fCoeff,self.nAct1D)
        self.DMmask  = circle( self.fCoeff*self.nAct1D/2, self.nAct1D).astype(int)

    def showMask(self):
        self.showIm(self.DMmask, 'DM masq, {:}'.format(self.DMmask.sum()))

class WFS_aux(vis):
    '''
    A class auxiliary to (any) WFS class
    '''
    def __init__(self, **kwargs):
        def_kw = {
            'wavelength': 633,
            'nSA1D'     : 12,
            'npxlSA'    : 40,
            # 'pupil'     : None,
            'nZern'     : 101,
            'spupsc'    : 0.833,
            'spupos':  [0.,0.],
            'paths':{
                'fpup':    None, #full pupil path
                'spup':   'PupilTracking/pupils/ngs_onAxis_mask.fits', # sub pupil path
                    }
            }
        # initialize all allowed keys to false
        self.__dict__.update((key, value) for key, value in def_kw.items())
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in def_kw)
        #print(self.__dict__)
        self.__dict__.update(kwargs)

        self.wfs_size = self.nSA1D*self.npxlSA
        self.spupsz = lapy.round_up_to_even(self.wfs_size*self.spupsc)
        # print overview of the initial parameters before populating the big arrays
        pprint.pprint(self.__dict__)
        self.full = False

        self.RtoCarth()

        self.makePupHD()
        # self.makeSPup()
        self.makeSubapMap( self.nSA1D, [self.npxlSA]*2, grid_offs=[self.npxlSA/2]*2, aotff=0.2 )

        #initialising the field over which Zernike polynomials are defined
        # self.Zmask = circle( self.spupsz/2, self.wfs_size, [self.hp,self.vp] )

        #defining a collection of Zernike matricï
        self.zMCollection={}
        self.Zrandlist()
        self.zList()

        self.Zaber_maker() # here self.aber is generated
        self.pupUp()  # to 'fuse' the pupil with the WFS

        # initialising the reconstructor operators
        # gradient based
        self.p_gO  = None
        self.f_gO  = None
        # Zernike based
        self.Z_cent_cal=None
        self.Z_cent_calF=None
        self.Z_FOp_i = None
        self.Z_pOp_i = None

    def makePupHD(self):
        '''
        defines the full pupil (HD)
        '''
        try:
            self.fPup = fits.getdata( self.paths['fpup'] )
        except:
            print('ATTENTION: global pupil not found, generating a circular!')
            self.fPup = circle( self.wfs_size/2, self.wfs_size )

    # def makeSPup(self):
        '''
        defines the sub pupil (HD)
        '''
        try:
            data = fits.getdata( self.paths['spup'] )
            data = lapy.insertImage( data, endShape=[int(data.shape[0]*1.02)]*2 )
            # the magic number above accounts for size being <=400 pxl for any rotatioin of the pupil
            self.sPup = zoom( data, self.spupsz/data.shape[0], grid_mode=1, order=0)
        except FileNotFoundError:
            print('ATTENTION: GCU pupil not found, generating a circular!')
            self.sPup = circle( self.spupsz/2, self.spupsz )

        # combining the pupils, ELT
        self.cpup = lapy.insertImage( self.sPup, endShape=[self.wfs_size]*2, vpos=self.vp, hpos=self.hp)
        # combining the pupils, round WFS
        self.rpup = circle( self.spupsz/2, self.wfs_size, [self.hp, self.vp] )

    def ini_noise(self):
        self.ADU  = 2**self.bpp
        self.eADU = self.ADU/self.welld # electrons per ADU
        self.ronADU = self.ron*self.eADU

    def fitTT(self, wf):
        #wf = wf - wf.mean()

        sz = wf.shape[0]
        TTlist = np.zeros( (3,sz,sz))
        for i in range(1,4):
            TTlist[i-1] = lapy.img_cropper( zern( i, 2*sz ), centre=[sz,sz], cropsize=sz)

        ttmask = TTlist[:, TTlist[0]>0]
        matxTT = np.dot( np.linalg.inv( np.dot( ttmask, ttmask.T) ), ttmask)
        #print(matxTT.shape, ttmask.shape, TTlist.shape)
        return np.matmul( matxTT, wf[TTlist[0]>0] )

    def RtoCarth(self, Rphi=None):
        ''' Converts coordinates from Radial to Carthesian
            Input:
                Rphi - vector [R,phi] where R=[0,1], phi=[rad]
        '''
        if Rphi is None:
            R=int(self.spupos[0]*self.npxlSA)
            phi=(self.spupos[1]*np.pi/180)
        else:
            R=Rphi[0]*self.npxlSA
            phi=Rphi[1]*np.pi/180
        self.hp=np.round(R*np.cos(phi)).astype(int)
        self.vp=np.round(R*np.sin(phi)).astype(int)

        print(Rphi, self.hp, self.vp)

    def pupUp(self):
        '''
        pupils update
        '''
        # self.hp=int(self.spupos[1]*self.npxlSA)
        # self.vp=int(self.spupos[0]*self.npxlSA)

        # self.RtoCarth()
        hp=self.hp
        vp=self.vp
        sz=self.wfs_size

        self.cpup = lapy.insertImage( self.sPup, endShape=[sz]*2, vpos=vp, hpos=hp)
        self.rpup = circle( self.spupsz/2, sz, [hp,vp] )
        self.getFillMap(self.cpup)



    def makeSubapMap(self, nsub=12, n_sub_pxl=[40]*2, sub_sep=[0,0], grid_offs=[0,0], aotff=0.):
        '''
        generates new sub-app grid based on:
            nsub        - the principal number of sub apertures across the WFS
            n_sub_pxl   - number of pixels per subap
            sub_sep     - the spacing between the sub aperture principals
            grid_offs   - the offset of the WFS sub aperture grid [X,Y]
            aotff       - aotools circle fill factor, must be between 0 and 1

        TODO:      make it rectangular, currently principal subaps are in square grid
        '''

        subapmap    = circle( aotff+nsub/(2), nsub )
        self.subapmap = subapmap.copy().astype(int)
        self.n_sub_tot = int(subapmap.sum())

        subx, suby = sub_sep
        nsizx, nsizy = n_sub_pxl
        xoff, yoff = grid_offs
        xsep=[subx-nsizx]
        ysep=[suby-nsizy]

        sa_coord = np.zeros( ( self.n_sub_tot, 5) ).astype(np.float32)

        n=0;
        for v in range(nsub):
            for h in range(nsub):
                #If subap is valid, add to list
                if subapmap[v,h] == 1:
                    #print( (yoff + v*(nsizy+suby)), (xoff + h*(nsizx+subx)) )
                    sa_coord[n] = ( round(yoff + v*(nsizy+suby)), round(xoff + h*(nsizx+subx)), h, v, 0 )
                    n+=1

        self.sa_coord = sa_coord.copy().astype(int)
        # coordinates of the subapertures
        # [0,1] - coordinates on sensor [pxl]
        # [2,3] - index of coordinates [Subaperture ]
        # [4]   - valid subapertures

    def getFillMap(self, inp=None):
        if inp is None:
            return -1j

        self.fillMap = self.subapmap.copy().astype(float)
        sc=self.sa_coord
        szx=szy=self.npxlSA
        sa = []
        for i, c in enumerate(sc):
            c0 = int(c[0]-szx/2) #corners
            c1 = int(c[1]-szy/2) #corners
            c2 = int(c[0]+szx/2) #corners
            c3 = int(c[1]+szy/2) #corners

            #self.fillMap[c[2]][c[3]] = inp[ c1:c3,c0:c2 ].sum()/self.npxlSA**2
            self.fillMap[c[2]][c[3]] = lapy.cropCorn(inp, [c0,c1,c2,c3]).sum()/self.npxlSA**2
            if self.fillMap[c[2]][c[3]] >= self.ff:
                self.sa_coord[i][4] = 1 # valid subaps
            else:
                self.sa_coord[i][4] = 0
            # print( i, self.fillMap[c[2]][c[3]], self.ff, self.sa_coord[i] )

        self.fillMap[self.fillMap<self.ff]=0
        self.fillMap[self.fillMap>=self.ff]=1
        self.n_ssub_tot = self.sa_coord.T[4].sum() # number of sub pupil subaps

    def make_noise(self):
        self.ronim = np.random.normal( self.ronADU, size=[self.wfs_size]*2)
        self.photnoizmap = np.random.poisson(self.wfs_img)

    def makeGradOpP(self, samap):
        ''' make the partial gradient operator '''
        self.p_gO    = gOT1( samap )#, self.wfs_flux )
        self.part_gM = self.p_gO.returnOp()
        self.p_gM_i  = np.linalg.pinv( self.part_gM, rcond=1/105. )

    def makeGradOpF(self, samap):
        ''' make the full gradient operator '''
        self.f_gO = gOT1( samap )#, samap )
        self.full_gM    = self.f_gO.returnOp()
        self.f_gM_i  = np.linalg.pinv( self.full_gM, rcond=1/142. )

    def makeZCents(self, A=.5e-6):
        aborig=self.aber.copy()
        fullC = []
        partC = []
        for z in range(1, self.nZern):
            self.aber = lapy.insertImage( self.Zlst_2D[z]*A, endShape=[self.wfs_size]*2, vpos=self.vp, hpos=self.hp )
            self.makeSubapsImg()
            if self.full: fullC.append(self.wfs_centF.ravel(order='F'))
            partC.append(self.wfs_cent.ravel(order='F'))

        self.Z_cent_cal=np.asarray( partC )/A
        if self.full: self.Z_cent_calF=np.asarray( fullC )/A
        else: self.Z_cent_calF=None

        # reset the operators
        self.Z_FOp_i = None
        self.Z_pOp_i = None

        self.aber=aborig
        # return np.asarray(fullC)/A, np.asarray(partC)/A

    def makeZOp(self, A=.5e-6):#, p=1, f=1):
        '''
            Make Zernike-Centroids operator,
            A - amplitude in pixels
        '''
        f, p = self.Z_cent_calF, self.Z_cent_cal

        if self.full:
            if f is None:
                self.makeZCents(A)

            if self.Z_FOp_i is None:
                # full Operator
                # f, p = self.makeZCents(A)
                self.Z_FOp   = f.copy()
                self.Z_FOp_i = np.linalg.pinv( f.T@f )@f.T
                if self.Z_pOp_i is None:
                    # partial Operator
                    f, p = self.makeZCents(A)
                    self.Z_pOp   = p.copy()
                    self.Z_pOp_i = np.linalg.pinv( p.T@p )@p.T

        if self.Z_pOp_i is None:
            if p is None:
                self.makeZCents(A)
                f, p = self.Z_cent_calF, self.Z_cent_cal
            # partial Operator
            # f, p = self.makeZCents(A)
            self.Z_pOp   = p.copy()
            self.Z_pOp_i = np.linalg.pinv( p.T@p )@p.T

    def makeSubapsImg(self, calF=None, full=False):
        '''
        calF - converting nm RMS to whatever is in FFT space pixels
        full - if True calculate the full pupil centroids
        '''
        self.full=full
        if calF is None: calF = 1/self.calFFT/self.calFPh

        szx = szy = self.npxlSA
        sc        = self.sa_coord
        fpup      = self.fPup
        fphs      = self.aber*calF
        cpup      = self.cpup
        cphs      = self.cpup*self.aber*calF
        subapmask = 1#circle( self.npxlSA/2,self.npxlSA)

        self.wfs_img   = np.zeros( [self.wfs_size]*2 )
        self.wfs_flux  = self.fillMap.copy()
        self.wfs_cent  = np.zeros( (self.n_sub_tot, 2))# will try to filter the illuminted later

        if full:
            self.wfs_imgF = np.zeros( [self.wfs_size]*2 )
            self.wfs_fluxF= self.subapmap.copy()
            self.wfs_centF= np.zeros( (self.n_sub_tot, 2))

        for i, c in enumerate(sc):
            c0 = int(c[0]-szx/2) #corners
            c1 = int(c[1]-szy/2) #corners
            c2 = int(c[0]+szx/2) #corners
            c3 = int(c[1]+szy/2) #corners

            E   = lapy.scale( lapy.cropCorn(cpup, [c0,c1,c2,c3] )*subapmask, [20]*2)#down binning phase
            ph  = lapy.scale( lapy.cropCorn(cphs, [c0,c1,c2,c3] )*subapmask, [20]*2)#down binning phase
            psf = self.makePSF( E, ph, self.pad )

            self.wfs_img[c1:c3,c0:c2] = psf.copy()
            if full:
                E   = lapy.cropCorn(fpup, [c0,c1,c2,c3] )*subapmask
                ph  = lapy.cropCorn(fphs, [c0,c1,c2,c3] )*subapmask
                psf = self.makePSF( E, ph, self.pad  )
                self.wfs_imgF[c1:c3,c0:c2] = psf.copy()
                # self.wfs_flux[c[2]][c[3]] = lapy.cropCorn( self.wfs_img, [c0,c1,c2,c3]).sum()

        self.wfs_img = self.wfs_img/self.wfs_img.max()*(2**self.bpp)
        if full: self.wfs_imgF = self.wfs_imgF/self.wfs_imgF.max()*(2**self.bpp)

        j=0
        for i, c in enumerate(sc):
            c0 = int(c[0]-szx/2) #corners
            c1 = int(c[1]-szy/2) #corners
            c2 = int(c[0]+szx/2) #corners
            c3 = int(c[1]+szy/2) #corners

            psf = lapy.cropCorn(self.wfs_img, [c0,c1,c2,c3])
            if full:
                self.wfs_fluxF[c[2]][c[3]] = psf.sum()
                self.wfs_centF[i] = aotools.centre_of_gravity( psf, self.cog_thrs ) - self.npxlSA/2

            psf = lapy.cropCorn(self.wfs_img, [c0,c1,c2,c3])
            self.wfs_flux[c[2]][c[3]] = psf.sum()
            if self.sa_coord[i][4]:
                self.wfs_cent[i] = aotools.centre_of_gravity( psf, self.cog_thrs ) - self.npxlSA/2
                j+=1

    def decompCents(self, DRAW=1, re=1):
        if self.full: self.decF = (self.wfs_centF.ravel('F'))@(self.Z_FOp_i)

        self.dec = (self.wfs_cent.ravel('F'))@(self.Z_pOp_i)
        if DRAW:
            zax = np.arange( 2, self.dec.shape[0]+2., 1)
            plt.figure( 'decompo', clear=re);
            if self.full: plt.plot( zax, self.decF*1e9, 'r.:', label='full' )
            plt.plot( zax, self.dec*1e9, 'b.:', label='part' )
            plt.xlabel('Zernike, [Noll]'); plt.ylabel('RMS, [nm]')
            plt.legend(); plt.grid(True)


        # self.wfs_centF *= self.calFPh/4
        # self.wfs_cent *= self.calFPh/4

    def subapPlot(self, data=[], size=12, ec='c', fign='', it=0):

        #data[self.Zmask>0] -= data.min()
        #self.onoff=True
        plt.ion()
        szx = szy = size

        fig = plt.figure('subapmap '+fign)
        plt.clf()
        fig.canvas.mpl_connect('close_event', self.close)
        sc=self.sa_coord
        ax = fig.add_subplot(111)
        a=plt.imshow( data[0], cmap='turbo')
        ax.set_xlim(-1, data[0].shape[0]+1)
        ax.set_ylim(-1, data[0].shape[0]+1)
        if len(data)>1:
            b=plt.imshow( data[1], cmap='gray_r', alpha=0.2)
            b.set_clim(0,1)
            #self.getFillMap( self.cpup )

        sa = []
        for i, c in enumerate(sc):
            if self.fillMap[c[2]][c[3]]<=0:
                lc=.5; ec='lime'
            elif self.fillMap[c[2]][c[3]]<=0.5:
                lc=.5; ec = 'c'
            #elif self.fillMap[c[2]][c[3]]<=0.5:
                #lc=.5; ec = 'lime'
            #elif self.fillMap[c[2]][c[3]]<=0.75:
                #lc=.5; ec = 'orange'
            elif self.fillMap[c[2]][c[3]]<=1:
                lc=1; ec = 'k'
            #lc=0.5
            sa.append( ( Rectangle((c[0]-szx/2,c[1]-szy/2),szx,szy,linewidth=lc,edgecolor=ec,facecolor='none'), self.fillMap[c[2]][c[3]] ) )

        sa.sort( key=lambda tup: tup[1] )
        for s in np.asarray(sa).T[0]:
            ax.add_patch( s )

        plt.axis('off')
        #plt.title('press \'q\' to close')
        if it >0:
            a.set_clim( self.aber.min(), self.aber.max() )
        #plt.colorbar()
        plt.tight_layout()
        #plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1/20.)
        if it >0:
            fig.savefig( 'ani/subapmap_{:s}{:03d}.png'.format(fign, (it)))
            #fig.canvas.print_png( 'ani/subapmap'+fign+str(it)+'.png')

        return
    # cheeky by-pass, long fogotten meaning
        while self.onoff:
            sc  = sashm.get_data()
            data= Imshm.get_data()
            dmax=data.max()
            a.set_data( 10**np.log10( data ) )
            a.set_clim(0,dmax)
            for i, c in enumerate(sc):
                #plt.gca().add_patch(Rectangle((c[0]-szx/2,c[1]-szy/2),szx,szy,linewidth=0.33,edgecolor='w',facecolor='none'))#centered
                ax.patches[i].set_x( c[0]-szx/2)
                ax.patches[i].set_y( c[1]-szy/2)


            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1/20.)

    def makePSF(self, A=None, Ph=None, pad=1, ordr='n'):
        '''
        inputs:
            A   - amplitude of Efield
            Ph  - phase
        returns:
            psf - magnitude of resulting field, normalised to 1
        '''
        if pad < 1: pad = 1
        if type(pad) is not int:
            pad = pad.astype(int)

        pa  = lapy.insertImage( A, endShape=[A.shape[0]*pad]*2)
        ph  = lapy.insertImage( Ph, endShape=[Ph.shape[0]*pad]*2)

        # if pad > 1:
        #     pa = np.zeros( ([ A.shape[0]*pad]*2) )
        #     ph = np.zeros( ([Ph.shape[0]*pad]*2) )
        #     pa[::pad, ::pad] = A
        #     ph[::pad, ::pad] = Ph
        # else:
        #     pa = A
        #     ph = Ph


        if Ph is None or A is None:
            print('Input is None, aborting...')
            return -1j

        ph = ph*ordsmag[ordr]/self.wvl

        psf = np.abs( np.fft.fftshift( np.fft.fft2(pa*np.exp(1j*ph) ) ) )**2
        # psf = np.abs( np.fft.fftshift( np.fft.fft2(A*np.exp(1j*Ph), s=[A.shape[0]*pad]*2 ) ) )**2

        # print(psf.shape, A.shape)
        c = int(A.shape[0]*(pad-1)/2)
        corns = [c, c, c+A.shape[0], c+A.shape[0]]
        # return lapy.rebin(psf, [A.shape[0],A.shape[1]])
        # return lapy.cropCorn(psf, corns)
        return psf#/psf.sum()

class MP_SubapsImg(mp.Process):
    def __init__(self, group=None, target=None, name=None, args=()):
        mp.Process.__init__(self, group=group, target=target, name=name)
        self.args   = args

    def run(self):
        szx = szy = self.npxlSA
        sc        = self.sa_coord
        fpup      = self.fPup
        fphs      = self.aber
        cpup      = self.cpup
        cphs      = self.cpup*self.aber
        subapmask = circle( self.npxlSA/2,self.npxlSA)

        self.wfs_img   = np.zeros( [self.wfs_size]*2 )
        self.wfs_flux  = self.fillMap.copy()
        self.wfs_cent  = np.zeros( (self.n_sub_tot, 2))# will try to filter the illuminted later

        self.wfs_imgF = np.zeros( [self.wfs_size]*2 )
        self.wfs_fluxF= self.subapmap.copy()
        self.wfs_centF= np.zeros( (self.n_sub_tot, 2))

        for i, c in enumerate(sc):
            c0 = int(c[0]-szx/2) #corners
            c1 = int(c[1]-szy/2) #corners
            c2 = int(c[0]+szx/2) #corners
            c3 = int(c[1]+szy/2) #corners

            E   = lapy.cropCorn(cpup, [c0,c1,c2,c3] )*subapmask
            ph  = lapy.cropCorn(cphs, [c0,c1,c2,c3] )*subapmask
            psf = self.makePSF( E, ph, self.pad )

            self.wfs_img[c1:c3,c0:c2] = psf.copy()

            E   = lapy.cropCorn(fpup, [c0,c1,c2,c3] )*subapmask
            ph  = lapy.cropCorn(fphs, [c0,c1,c2,c3] )*subapmask
            psf = self.makePSF( E, ph, self.pad  )
            self.wfs_imgF[c1:c3,c0:c2] = psf.copy()
            # self.wfs_flux[c[2]][c[3]] = lapy.cropCorn( self.wfs_img, [c0,c1,c2,c3]).sum()

        self.wfs_img = self.wfs_img/self.wfs_img.max()*(2**self.bpp)
        j=0
        for i, c in enumerate(sc):
            c0 = int(c[0]-szx/2) #corners
            c1 = int(c[1]-szy/2) #corners
            c2 = int(c[0]+szx/2) #corners
            c3 = int(c[1]+szy/2) #corners

            psf = lapy.cropCorn(self.wfs_imgF, [c0,c1,c2,c3])
            self.wfs_fluxF[c[2]][c[3]] = psf.sum()
            self.wfs_centF[i] = aotools.centre_of_gravity( psf, self.cog_thrs ) - self.npxlSA/2


            psf = lapy.cropCorn(self.wfs_img, [c0,c1,c2,c3])
            self.wfs_flux[c[2]][c[3]] = psf.sum()
            if self.sa_coord[i][4]:
                self.wfs_cent[i] = aotools.centre_of_gravity( psf, self.cog_thrs ) - self.npxlSA/2
                j+=1

def mpSubapsImg(d, src='f'):
    ''' ATTENTION an attempt at parallelising '''

    fullOpt = ['f', 'F', 'full', 'Full']
    partOpt = ['p', 'P', 'part', 'Part']
    szx = szy = d.npxlSA
    sc        = d.sa_coord
    if src in fullOpt:
        pup      = d.fPup
        phs      = d.aber*1151
        flux     = d.subapmap.copy()
        # print('[ fullOpt ]')
    elif src in partOpt:
        pup      = d.cpup
        phs      = d.cpup*d.aber*1151
        flux     = d.fillMap.copy()
        # print('[ partOpt ]')
    else:
        print('ERR: option unknown, aborting')
        return

    subapmask = circle( d.npxlSA/2,d.npxlSA)

    img   = np.zeros( [d.wfs_size]*2 )
    cent  = np.zeros( (d.n_sub_tot, 2))# will try to filter the illuminted later

    for i, c in enumerate(sc):
        c0 = int(c[0]-szx/2) #corners
        c1 = int(c[1]-szy/2) #corners
        c2 = int(c[0]+szx/2) #corners
        c3 = int(c[1]+szy/2) #corners

        E   = lapy.cropCorn(pup, [c0,c1,c2,c3] )*subapmask
        ph  = lapy.cropCorn(phs, [c0,c1,c2,c3] )*subapmask
        psf = d.makePSF( E, ph, d.pad )

        img[c1:c3,c0:c2] = psf.copy()


    img = img/img.max()*(2**d.bpp)
    j=0
    for i, c in enumerate(sc):
        c0 = int(c[0]-szx/2) #corners
        c1 = int(c[1]-szy/2) #corners
        c2 = int(c[0]+szx/2) #corners
        c3 = int(c[1]+szy/2) #corners

        psf = lapy.cropCorn(img, [c0,c1,c2,c3])
        flux[c[2]][c[3]] = psf.sum()
        if d.sa_coord[i][4]:
            cent[i] = aotools.centre_of_gravity( psf, d.cog_thrs ) - d.npxlSA/2
            j+=1

    if src in fullOpt:
        d.wfs_imgF  =  img.copy()
        d.wfs_fluxF = flux.copy()
        d.wfs_centF = cent.copy()
    elif src in partOpt:
        d.wfs_img  =  img.copy()
        d.wfs_flux = flux.copy()
        d.wfs_cent = cent.copy()


class zaveDat:
    '''
    A class auxiliary to (any) WFS class
    '''
    def __init__(self, **kwargs):
        def_kw = {
            'wavelength': 633,
            'nSA1D'     : 12,
            'npxlSA'    : 40,
            # 'pupil'     : None,
            'nZern'     : 101,
            'spupsc'    : 0.833,
            'spupos':  [0.,0.],
            'paths':{
                'fpup':    None, #full pupil path
                'spup':   'PupilTracking/pupils/ngs_onAxis_mask.fits', # sub pupil path
                    }
            }
        # initialize all allowed keys to false
        self.__dict__.update((key, value) for key, value in def_kw.items())
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in def_kw)
        #print(self.__dict__)
        self.__dict__.update(kwargs)

        pprint.pprint(self.__dict__)

    def saveDat(self, datset=None, path=None):
        ''' saves the data set '''
        if datset is None:
            print('ERR: datset is None, dunno what to do, aborting...')
            return -1j
        t = time.localtime()
        self.start_at = '{:4d}{:02d}{:02d} {:02d}:{:02d}:{:02d}'.format( t[0], t[1], t[2], t[3], t[4], t[5] )

        if not os.path.exists( path ):
            os.makedirs( path )

        fN = path+'{:s}'

        tosave  = ['here goes list of objects to save: ']
        #print( tosave, datset.sa_imgs.shape )
        saveDic = {
                'COMMENT' : ' Save Dat Res ',
                'SAVED_AT': self.start_at,
                'PATHTERN': '{:s}'  .format(datset.det['pathrn']),
                'PHS_TYPE': '{:s}'  .format(datset.algo['typ']),
                'SA_PXLS' : '{:d}'  .format(datset.npxlSA),
                'SUBAPS'  : '{:d}'  .format(datset.nsaps),
                'TOTSUBAP': '{:d}'  .format(datset.sa_tot),
                'PUPSIZE' : '{:.2f}'.format(datset.Pupil),
                'THRESHLD': '{:.2f}'.format(datset.trshld),
                'WAVELNGT': '{:.3e}'.format(datset.wv),
                'PXLSCALE': '{:.3e}'.format(datset.pxl_scale),
                'R0'      : '{:s}'.format(' '.join(str(x) for x in datset.r0) ),
                }
        lapy.saveFits( {'DATA': datset.slopes_cg,'HDR':saveDic}, fName=fN.format('wfs_cents_CoG') )
        lapy.saveFits( {'DATA': datset.slopes_cc,'HDR':saveDic}, fName=fN.format('wfs_cents_CC') )
        lapy.saveFits( {'DATA': datset.sa_cent,'HDR':saveDic}, fName=fN.format('wfs_cents_TT') )

        lapy.saveFits( {'DATA': datset.sa_psf,'HDR':saveDic}, fName=fN.format('wfs_subap_PSF') )
        lapy.saveFits( {'DATA': datset.sa_img,'HDR':saveDic}, fName=fN.format('wfs_subap_IMG') )
        lapy.saveFits( {'DATA': datset.sa_phs,'HDR':saveDic}, fName=fN.format('wfs_subap_PHS') )

        lapy.saveFits( {'DATA': datset.phs,'HDR':saveDic}, fName=fN.format('phase_screen') )
        lapy.saveFits( {'DATA': datset.pattrn,'HDR':saveDic}, fName=fN.format('speckles') )

        lapy.saveFits( {'DATA': datset.subap_ref,'HDR':saveDic}, fName=fN.format('wfs_subap_reference') )

        lapy.saveFits( {'DATA': datset.zerns_tt,'HDR':saveDic}, fName=fN.format('zernike_TT') )
        lapy.saveFits( {'DATA': datset.zerns_cg,'HDR':saveDic}, fName=fN.format('zernike_CoG') )
        lapy.saveFits( {'DATA': datset.zerns_cc,'HDR':saveDic}, fName=fN.format('zernike_CC') )


