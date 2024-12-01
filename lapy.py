# 'My' collection of 'Py'thon functions
# mostly created by Lazar
# since 2023 another python module called mypy has become very popular
# therefore, this one is renamed lapy

import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import os
import pickle
#import numba
#import numba.targets.linalg as lan

from astropy.io import fits
from aotools import image_processing as imp
from matplotlib.colors import LogNorm
from numpy import mean, square
from numpy import sqrt as root

import PIL.Image as PIM

from scipy import ndimage
from scipy import interpolate
from scipy import misc #DEPRECATED use the imports underneath
from scipy.signal import convolve2d as c2d
from scipy.signal import fftconvolve as fftc

from skimage.transform import resize
#from skimage import data

from scipy.linalg import expm

from aotools import zernike_noll as zern
from aotools import circle

from ABBOT.gradientOperator import gradientOperatorType1 as gOT1

#import darc

listOfZernike   = None
nominalZlist    = None
Zlist46         = None
Zlist100        = None
Zlist150        = None

gM_i            = None

SH_scaling      = 138.8 # SH WFS calibration factor in [nm]
Mag_factor      = 2.86  # DM to SH magnification

def get_objs(parent=None, kwargs=None):
    '''
    Object oriented atomic union between a Class and a Dict. The beauty is in the dinamic nature of the union
    Input:
        parent  - class that can utilise the content of the dictionary
        kwargs  - dict that contains class initialization parameters
    Returns:
        object of the class populated with kwargs paramters
    '''
    if parent is None:
        print('inadequate parent, see usage for more details.')
        return -1j


    class objs(parent):
        def __init__(self, kwargs):
            '''
            object handler
            populate the given keys as members of this object and read their given values
            inherits from parent
            '''
            self.__dict__.update(kwargs)

    return objs(kwargs)

def polar2z(R,phi):
    return R*np.exp( 1j*phi )

def z2polar(z):
    return np.abs(z), np.angle(z)

def cart2pol(x, y):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return R, phi

def pol2cart(R, phi):
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y

def Gauss_2D( sz=None, sigma=1, muXY=[0,0]):
    '''
        Retunrs normalised 2D Gaussian
        Input:
            sz      - size of the returned 2D array
            sigma   - standard deviation normalised to sz
            muXY    - mean location relative to centre (sz//2)
    '''
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    if sz is None:
        print('ERR: size is None, aborting...')
        return -1j

    xm, ym = muXY
    x, y = np.meshgrid(np.linspace(-1, 1, sz), np.linspace(-1, 1, sz))
    dst = np.sqrt((x+xm)**2+(y+ym)**2)

    # lower normal part of gaussian
    normal = 1/(2 * np.pi * sigma**2)

    gauss = np.exp(-((dst)**2 / (2.0 * sigma**2))) * normal
    return gauss/gauss.max()

def aber_maker(Zlist, size=121, asize=None, DRAW=False, ex=1.):
    #Zlist = self.Zlist
    #if Zlist.ndim != 2 or Zlist.shape[1] != 2:
        #print('invalid Zlist format, aborting...')
        #return -1

    if Zlist.ndim == 1:
        nollL = np.linspace( 1, Zlist.shape[0], Zlist.shape[0] )
        zs = Zlist.copy()
        Zlist = np.zeros( (Zlist.shape[0], 2) )
        Zlist.T[0] = nollL.copy()
        Zlist.T[1] = zs.copy()

    nZ=Zlist.shape[0]
    #print(Zlist)

    if asize is None:
        asize = size #int( np.round(size/2) )

    pist=zern(1,size)
    aber=np.zeros((nZ, size, size))
    # print(Zlist)
    for i, z in enumerate( Zlist ):
        zz  = zern( int(z[0]), size)
        zz -= zz[pist>0].mean()
        if i > 0:
            zz  /= (zz[pist==1].std())
            # print( i, z, zz.std() )

        #print(zz.shape)

        # if asize < size:
        #     aber[i]  = insertImage( zz, endShape=[size]*2 )
        # else:
        #     aber[i] = zz.copy()
        #print(f'{zz[zern(1,asize)>0].std():.3f}, {(aber[i][circle(size/4,size)>0]).std():.3f}')
        aber[i] = zz*z[1]

    if DRAW:
        #print(Zlist)
        title = '$\sigma=${:.3f}'.format( aber.sum(0)[pist==1].std() )
        plt.figure('aber', clear=1);
        plt.imshow( aber.sum(0)/circle( size/2, size ), extent=[-ex,ex,-ex,ex] )
        plt.title(title)
        plt.colorbar()

    return aber.sum(0)

def cropCorn(inp=None, c=None):
    '''
    crops an image with the crop defined with it's corners
    '''
    if inp is None or c is None:
        print('ERR: input is None, naught to do, aborting...')
        return TypeError

    return inp[c[1]:c[3],c[0]:c[2]]

def eight_neighbor_average_convolve2d(x):
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    neighbor_sum = convolve2d(
        x, kernel, mode='same',
        boundary='fill', fillvalue=0)

    num_neighbor = convolve2d(
        np.ones(x.shape), kernel, mode='same',
        boundary='fill', fillvalue=0)

    return neighbor_sum / num_neighbor

def cent(img, modusop='space', confFile="/home/simulink/aocp/config/aocc_sh_wfschar.yaml" ):
    '''
    returns centroids from a single image using the AOCC WFS class

    Parameters:
        img         - the image to be analysed, can be a single image or an array
        modusop     - can be 'space' or 'earth' defining if CoG or crosscorrelation centroid algorithm will be used
        confFile    - the file with the SH_WFS configuration
    Returns:
        an array of centroids
    '''
    sh = centroider.Centroider( confFile )
    sh.modusop = modusop
    return sh.centroid( img )

def getFWHM_area(data):
    '''
    Very crude FWHM apporoximation.
        returns the Diameter of the circle with area the sum of pixels above 1/2 amplitude
    '''

    arr = data/data.max()

    arr[arr<.5]=0
    arr[arr>=.5]=1

    return 2*(arr.sum()/np.pi)**0.5


def getCent( data ):
    '''
    cycles throug multiple images and returns array of multiple centroids
    '''

    if data.ndim == 2:
        return cent( data, modusop='space' )
    elif data.ndim ==3:
        centroids = []
        for i in range( data.shape[0] ):
            centroids.append( cent( data[i], modusop ) )
            #if i%10:
                #print('cent# ', i)
        return np.asarray( centroids )
    else:
        print( 'datashape incompatible' )
        return

def update_animation(i):
    actmap[i] = drawSHmap( (cen[i] - cen[0]), DRAW=False )
    ax.set_data( actmap[i] )
    plt.title( str( i ) )
    plt.pause( 0.0001 )

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def getDate():
    '''
    returns time string used for data management e.g. filenames, dirnames etc..
    '''
    t = time.gmtime()
    return '{:4d}{:02d}{:02d}'.format( t[0], t[1], t[2] )

def pickleMein(obj, name ):
    '''
    dumps object to file using pickle, used to save dictionary with the result from the data analysis
    '''
    if name[-4:] != '.pkl':
        name = name + '.pkl'
    with open( name, 'wb') as f:
        pickle.dump(obj, f, 2)

def pickleMeout( name ):
    '''
    loads pickled data
    '''
    if name[-4:] != '.pkl':
        print( name[-4:] )
        name = name + '.pkl'
    with open( name, 'rb') as f:
        return pickle.load(f)

def deAlify(vec, N=4064):
    #N=nSA*nSA
    ret=vec.copy()*0.
    #for i in range( vec.shape[0]/2 ):
        #ret[i*2]   = vec[i]
        #ret[i*2+1] = vec[i+vec.shape[0]/2]
    ret[:N/2]=vec[::2]
    ret[N/2:]=vec[1::2]

    return ret

def circMask( X=100, Y=100, R=52, DRAW=False ):
    '''
    TASK: document this
    creates a spot with circular shape
    placed at coordinates (0, 0)
    input:
        X, Y  - dimensions of the output image
        R     - diameter of the mask
    '''
    x = np.linspace(-X, X, X+1, dtype=int)  # x values of interest
    y = np.linspace(-Y, Y, Y+1, dtype=int)[:,None]# y values of interest, as a "column" array
    truthmask = ((x)/R)**2 + ((y)/R)**2 <= 1  # True for points inside the ellipse
    print( truthmask.shape )
    img = np.zeros( (Y+1,X+1), dtype=int )
    img[truthmask]=1
    if DRAW:
        plt.figure('cm')
        plt.imshow( img.T )
        plt.title(str(img.sum()))
    return img.T

def ellipMask( a=100, b=100, X=100, Y=100, y0=0, x0=0 ):
    '''
    creates a spot with elliptical shape with semiaxes (a, b)
    placed at coordinates (x0, y0) defaults correspod to the PSF
    location on PD screen
    '''
    x = np.linspace(0, X, X)  # x values of interest
    y = np.linspace(0, Y, Y)[:,None]# y values of interest, as a "column" array
    truthmask = ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1  # True for points inside the ellipse

    img = np.zeros( (Y,X) )
    img[truthmask]=1
    return img.T

def rotFlip( img, rot=0.0, flipa=False ):
    '''
    rotate and flip an img, used to set the propper orientation for Zernike distributions to be displayed on DM1
    Parameters:
        img - the array to be manipulated
        rot - any angle in degrees
        flip - the axis (0 or 1) around which the flip will be
    Returns:
        an flipped/rotated image
    '''
    if flipa:
        img = np.flip( np.ma.masked_invalid(img), 0 )

    return ndimage.rotate( np.ma.masked_invalid(img), rot, reshape=False )

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def round_down_to_even(f):
    return math.floor(f / 2.) * 2

def convMasq(Img, kern=None, typ=None, Rk=3, Nk=8, Th=.5):
    '''
    uses convolution to create filter mask
        Rk  - Radius of kern
        Nk  - Size of kern
        Th  - Threshold for masqing
    '''

    if kern is None:
        kern = circle( Rk, Nk )

    img=Img.copy()
    if img.max()>1:
        img[img<0] = 0
        img[img>0] = 1

    masq = c2d( img, kern, 'same', 'fill', fillvalue=0)

    print( kern.sum(), masq.max() )
    masq[masq<kern.sum()*Th] = 0
    masq[masq> 0] = 1

    kern = circle( Rk/2, Nk )

    masq = c2d( masq, kern, 'same', 'fill', fillvalue=0)
    print( kern.sum(), masq.max() )
    masq[masq<kern.sum()] = 0
    masq[masq> 0] = 1

    plt.figure('masq')
    plt.clf()
    plt.imshow(masq, interpolation='none')
    plt.colorbar()

    return masq

#OPT_CENT=(0,0)
def imCrop(img, idx):
    return img[ idx[0]:idx[1], idx[-2]:idx[-1] ]

def img_cropper(img, centre, cropsize=64):
    '''
    crops an image down to a square of cropsize, used for the PD analysis, needs to be to the power of two
    Parameters:
        img         - image to be croped
        centre      - the centre around which the image will be cropped
        cropsize    - the size of the cropped image
    returns:
        a square sub array of img
    '''

    cropar = [int(centre[0] - cropsize/2),
              int(centre[0] + cropsize/2),
              int(centre[1] - cropsize/2),
              int(centre[1] + cropsize/2)]
    if img.ndim == 2:
        return img[cropar[0]:cropar[1], cropar[2]:cropar[3]]
    elif img.ndim ==3:
        tmp = []
        for i in range( img.shape[0] ):
            tmp.append( img[i, cropar[0]:cropar[1], cropar[2]:cropar[3]] )
        return np.asarray( tmp )
    else:
        print( 'datashape incompatible' )
        return

def insertImage( image, imdim=None, endShape=[128,128], vpos=0, hpos=0, angle = None, xy0=None, scale=1. ):
    '''
    returns an insert of an image centered around coordinates (x0, y0)
    '''
    imIns = np.zeros( ( endShape ) )
    if imdim is None:
        imdim = image.shape

    x_scale = round_up_to_even( imdim[0]*scale )
    y_scale = round_up_to_even( imdim[1]*scale )
    #ins = np.flip( np.flip( misc.imresize( image, (x_scale, y_scale), interp='nearest', mode='F'), 0), 1)
    if scale == 1.:
        ins = image.copy()
    else:
        ins = resize(image, (x_scale, y_scale), order=0 ) #mode = {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}

    x_ofs = int( x_scale/2 )
    y_ofs = int( y_scale/2 )

    if xy0 is None:
        xy0=[0,0]
        xy0[0], xy0[1]= int( endShape[0]/2. ), int( endShape[1]/2. )

    if angle is None:
        imIns[ xy0[0] - x_ofs+vpos:xy0[0] + x_ofs+vpos , xy0[1] - y_ofs+hpos:xy0[1] + y_ofs+hpos ] = ins.copy()
    else:
        imIns[ xy0[0] - x_ofs+vpos:xy0[0] + x_ofs+vpos , xy0[1] - y_ofs+hpos:xy0[1] + y_ofs+hpos ] = ndimage.rotate( ins, angle, reshape=False )

    return imIns

def half_max_x(x, y):
    '''
    returns FWHM using very crude procedure, works only on the highest of peaks
    '''
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def sqSum(zdc):
    if zdc.ndim == 1:
        return ((zdc**2).sum())**0.5
    elif zdc.ndim == 2:
        return ((zdc**2).sum(0))**0.5

def stdAmp( img, rr=52 ):
    '''
    DEPRECATED
    returns the RMS amplitude of img in nm, elMasq is applied
    '''
    elMasq  = SAflag( rr ).reshape( (rr,rr) )

    return (np.ma.masked_invalid(img[elMasq == 1]).std())

def pupStat( img, rr=52, mom=2 ):
    '''
    returns the requested central moment of a pupil img in nm, elMasq is applied
    input:
        img - 2D image array
        rr  - pupil diameter
        mom - which moment, 1-mean, 2-std, 3-skew, 4-kurtosis
    TODO implement skew and kurtosis
    '''
    elMasq  = SAflag( rr ).reshape( (rr,rr) )

    if mom == 1:
        return (np.ma.masked_invalid(img[elMasq == 1]).mean())
    elif mom ==2:
        return (np.ma.masked_invalid(img[elMasq == 1]).std())
    else:
        print('invalid moment requested, currently supported 1-mean, 2-std')
        raise TypeError


def drawSHmap( cent, rr=52, DRAW=True, ax='xy', fign=None ):
    '''
    draws a 2D map of the centroid  magnitude,
    WARNING the centroids must be Alified
    '''
    cm = SAflag( rr ).reshape( (rr,rr) )
    subapmap = np.zeros( (rr,rr) )
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
        if fign is None: plt.figure('SH map')
        else: plt.figure(fign)
        plt.clf()
        #plt.imshow( 10**np.log10( subapmap -0.0000001) )
        plt.imshow( ( subapmap/cm ) )#, clim=[0, .1] )
        plt.colorbar()
        plt.pause(0.01)

    return ( subapmap )

def DM_drawactval( dm1actval, DM=3, DRAW=True ):
    '''
    draws the actuator influence function for a set of mirrors:
    Input:
        dm1actval   - float vector containing the actuator values
        DM          - int reference to which mirror, 1) ALPAO-241, 2) TNO-57, 3) ALPAO-97
        DRAW        - Boolean
    Return:
        2D actuator map
    '''

    if DM==1:#ALPAO 241
        cm=ellipMask( 19/2, 19/2, 17, 17, 17/2, 17/2 ).astype(int)
        #plt.imshow( cm )

        act_map = np.zeros( (17,17)  )
        k=0
        for i in range( 17 ):
            for j in range( 17 ):
                if cm[i][j] == 1:
                    act_map[i][j] = dm1actval[k]
                    k+=1
        act_map = np.flipud( act_map ).T


    if DM==2:#TNO 57
        cm=ellipMask( 4.7, 4.7, 9, 9, 4.5, 4.5 ).astype(int)
        #plt.imshow( cm )

        act_map = np.zeros( ( 9, 9 )  )
        k=0
        for i in range( 9 ):
            for j in range( 9 ):
                if cm[i][j] == 1:
                    act_map[i][j] = dm1actval[k]
                    k+=1

    if DM==3:
        nAct=11
        cm=ellipMask( 6, 6, nAct, nAct, 5.5, 5.5 ).astype(int)
        #plt.imshow( cm )

        act_map = np.zeros( (nAct,nAct)  )
        k=0
        for i in range( nAct ):
            for j in range( nAct ):
                if cm[i][j] == 1:
                    act_map[i][j] = dm1actval[k]
                    k+=1
        act_map = np.flipud( act_map ).T


    if DRAW:
        plt.figure('DM1 act val')
        plt.clf()
        plt.imshow( act_map/cm )
        plt.colorbar()

    return ( act_map )

def SAflag(nSA=52):
    '''
    Input:
        nSA - number of SA across the pupil
    Output:
        returns SA flag as 1D array to be compatible with darc
    '''

    SAarray = ellipMask( nSA/2, nSA/2, nSA, nSA, nSA/2, nSA/2)

    return SAarray.ravel().astype(int)

def deslopify( slopes, size=98, TT=False ):
    '''
    conforms AOCC slopes to ABBOT requirement
    '''
    slopesWF = np.zeros( (size*2) ) #slopes compatible with ABBOT
    if slopes.ndim == 2:
        s = slopes.mean(0)
    elif slopes.ndim == 1:
        s = slopes
    else:
        print( 'slopes given have incompatible dimensions, ndim must be 1 or 2...')

    rem = int( ( len( slopes ) - size*2 )/2 )


    if TT:
        slopesWF[:size] = s[:size] - s[:size].mean()
        slopesWF[size:] = s[-size-rem:-rem] - s[-size-rem:-rem].mean()
    else:
        slopesWF[:size] = s[:size]
        slopesWF[size:] = s[-size-rem:-rem]

    return slopesWF

def gimmeLesky(dicta):
    '''
    !!! UNDER CONSTRUCTION !!!
    returns inverted Cholesky matrix for de-aliasing of results
    '''
    Cmm = np.zeros( (25, 25) )
    for i in range( 0, 23 ):# in range [Z4:Z26]
        Z = 'Z'+str(i+4)
        try:
            ref_SHZC = ( dicta[Z]['RESULTS']['0']['SH_DC'] )
            for K in dicta[Z]['RESULTS']:
                toAppnd = ( ( dicta[Z]['RESULTS'][K]['SH_DC'] ) - ref_SHZC )/int(K)
            plt.plot( ( dicta[Z]['RESULTS'][K]['SH_DC'] ) - ref_SHZC )
            plt.pause( 0.02 )
            print(Z, K, toAppnd.max())
            Cmm[i+2] =  ( ( ( dicta[Z]['RESULTS'][K]['SH_DC'] ) - ref_SHZC )/int(K) )
        except:
            pass

    C_MM = np.dot( Cmm, Cmm.T )
    for i in np.where( np.diag( C_mm ) == 0 )[0]:
        C_MM[i][i] = np.diag( C_MM ).mean()

    Lesky = np.linalg.inv( np.linalg.cholesky( C_MM ) )

    return Lesky

def drawWF( wf, N=53, fign=None, norm=None, obj=False ):
    '''
    draws the given WF and calculates the RMS amplitude
    input:
        wf      - 2D array with WF
        N       - transverse size of the WF
        fign    - name of figure
    '''
    if fign is None:
        plt.figure('WF')
    else:
        plt.figure(fign)
    plt.clf()
    rms = stdAmp(wf,N)
    plt.title('RMS={:3.1f}, PtV={:3.1f}'.format( rms, wf.ptp()))
    imsh=plt.imshow(wf)
    if norm is not None:
        plt.clim( -norm, norm )
    plt.colorbar()
    plt.pause(0.01)

    if obj:
        return imsh
    return rms

valid_subap_map=SAflag(52).reshape( (52,52) )
gO=gOT1( valid_subap_map )
gM=gO.returnOp()

def StoWF(cent, DRAW=False, name='', gM_i=None, gO=None, TT=False ):
    '''
    Slopes to Wavefront, gets centroids/slopes as input, returns wavefront, display is optional
    TT - when True T/T is subtracted from centroids
    NOTE works only with 1D TODO 2D
    '''
    if gO is None or gM_i is None:
        print('ERR: gradient operator cannot be None, aborting')
        return -1j

    if cent.ndim == 2:
        c = cent[:,:2]
        if TT:
            c -= c.mean(0)
        slopesV = np.ravel( c, order='F')[::-1]
    elif cent.ndim == 3:
        c = cent[:,:,:2]
        slopesV = c.reshape( (c.shape[0], c.shape[1]*2), order='F')[:,::-1]

    if slopesV.ndim == 1:
        wfV=gM_i.dot(slopesV)
        wfV-=wfV.mean()
        wf2D=np.zeros(gO.n_)
        wf2D.ravel()[gO.illuminatedCornersIdx]=wfV[::-1]
        if DRAW:
            plt.figure('wf'+name)
            plt.clf()
            plt.imshow( wf2D.T, interpolation='nearest')
            plt.title('RMS={:3.1f}, PtV={:3.1f}'.format( 1000*wfV.std(), 1000*wfV.ptp()))
            plt.colorbar()
            plt.pause(0.01)

        return wf2D.T, wfV

    elif slopesV.ndim == 2:
        wfV=np.matmul( gM_i, slopesV.T)
        return wfV# wf2D.T# gM

def WFtoS( wf, nSA=52, DRAW=False ):
    '''
    returns slopes from a wavefront
    '''
    #valid_subap_map=SAflag(nSA).reshape( (nSA,nSA) )
    #gO=gOT1( valid_subap_map )
    #gM=gO.returnOp()

    #this below matches the alpao test darc Sub Aperture settings in the gradient space
    a=26.7
    masq = ellipMask( a, a, 51, 51, 25.5, 25.5 )

    wfv     = wf[masq==1]
    synslop = ( wfv ).dot( gM.T )/SH_scaling/Mag_factor

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

def centZdc( centrs ):
    '''
    returns Zernike decomposition of centroids
    '''
    if centrs.ndim == 1:
        return Lesky.T.dot(wf.decompoZerg(waveFront(wf.deslopify(centrs,87,True))))
    elif centrs.ndim == 2:
        dc = np.zeros( ( centrs.shape[0], 25 ) )
        for i in range( centrs.shape[0] ):
            dc[i] = Lesky.T.dot(wf.decompoZerg(waveFront(wf.deslopify(centrs[i],87,True))))
        return dc

def decompoZergOld( zerg, Zorder=46, Zsize=51, mask=None, DRAW=False ):
    '''
    decomposition of Zernike
    '''

    listOz = []
    listOa = []
    global nominalZlist
    global Zlist46
    global Zlist100
    global Zlist150

    if Zorder == 46 and Zsize == 51:
        if Zlist46 is not None:
            z = Zlist46.copy()
        else:
            print('recalculating zList({:}, {:}), please have patience...'.format( Zorder, Zsize))
            Zlist46= zList(Zorder+1,Zsize)
            z = Zlist46.copy()
    elif Zorder == 100 and Zsize == 51:
        if Zlist100 is not None:
            z = Zlist100.copy()
        else:
            print('recalculating zList({:}, {:}), please have patience...'.format( Zorder, Zsize))
            Zlist100= zList(Zorder+1,Zsize)
            z = Zlist100.copy()
    elif Zorder == 150 and Zsize == 51:
        if Zlist150 is not None:
            z = Zlist150.copy()
        else:
            print('recalculating zList({:}, {:}), please have patience...'.format( Zorder, Zsize))
            Zlist150= zList(Zorder+1,Zsize)
            z = Zlist150.copy()
    else:
        print('recalculating zList({:}, {:}), please have patience...'.format( Zorder, Zsize))
        z = zList(Zorder+1,Zsize)

    if mask is None:
        sf_norm = (1/(z[0]*z[0]).sum())**0.5
        sf_cal  = 1/(sf_norm*z[0]).sum()
        for i in range( 1, Zorder +1):
            #z       = zern( i, Zsize)
            sf_norm = (1/(z[i-1]*z[i-1]).sum())**0.5 #for each zernike
            #if i is 1:
                #sf_cal  = 1/(sf_norm*z[i-1]).sum()
            listOa.append( ( sf_norm*z[i-1]*zerg ).sum()*sf_cal )
            #print( 'Z4={4.3f}'.format( np.asarray( listOa[3] ) ) )
    else:
        #plt.figure('mask')
        #plt.clf()
        #plt.imshow( mask )

        sf_norm = (1/(z[0]*z[0]*mask).sum())**0.5
        sf_cal  = 1/(sf_norm*z[0]*mask).sum()
        for i in range( 1, Zorder +1):
            #z       = zern( i, Zsize)
            sf_norm = (1/(mask*z[i-1]*z[i-1]).sum())**0.5 #for each zernike
            #if i is 1:
                #sf_cal  = 1/(sf_norm*z[i-1]).sum()
            listOa.append( ( (sf_norm*z[i-1]*zerg*mask ) ).sum()*sf_cal )
            #print( 'Z4={4.3f}'.format( np.asarray( listOa[3] ) ) )

    if DRAW:
        plt.figure('decomposed centroids')
        plt.clf()
        plt.plot( listOa, '.:')
        plt.pause(0.01)


    return np.asarray( listOa )

def decompoZerg(wf, Zorder=46, Zsize=51, mask=None):
    '''
    Orthonormalisation of pixelised Zernike distributions and decomposition of a wave front agains it
    Input parameters:
        wf      - the wavefron to be decomposed
        Zorder  - the highest order of Zernike to be included
        Zsize   - the size of the discretisation matrix
        mask    - which elements of the matrix to be included in the returned one

    returns the Zernike decomposition of a pixelised WF
    '''

    if mask is None:
        mask = zern.zernike_noll( 1, Zsize )

    zn      = zern.zernike_noll( 1, Zsize )
    sf_norm = 1#(1/(zn*zn).sum())**0.5

    zMatx = np.zeros( ( Zorder-1, int( mask.sum() ) ) )
    for i in range( 2, Zorder + 1):
        #zMatx[i-2] = sf_norm*( rotFlip( zern.zernike_noll( i, Zsize ), 180, True) )[ mask==1 ]
        zMatx[i-2] = sf_norm*( zern.zernike_noll( i, Zsize ) )[ mask==1 ]

    #print( mask.shape, sf_norm )
    return np.dot( np.dot( np.linalg.inv( np.dot( zMatx, zMatx.T) ), zMatx ), wf[ mask==1 ])

def decompoZerg_orth( zerg, Zorder=46, Zsize=53 ):
    '''
    Performs a Zernike decomposition of zerg, the dimensions of zerg must match Zsize
    Parameters:
        zerg    - input 2D array that needs to be decomposed
        Zorder  - the maximum Zernike order that will be decomposed
        Zsize   - the size of the arrays
    Returns
        array with the result from the decomposition
    TODO:
        calculate the list of Zernike polynomials only when needed
    '''

    listOa = []

    #if listOfZernike is None:
        #listOfZernike = zList(Zorder, Zsize)
    #elif listOfZernike is not None and (listOfZernike.shape[0] != Zorder or listOfZernike.shape[1] != Zsize ):
    listOfZernike = zList(Zorder, Zsize)

    z = listOfZernike.copy()

    sf_norm = (1/(z[0]*z[0]).sum())**0.5
    sf_cal  = 1/(sf_norm*z[0]).sum()
    for i in range( 1, Zorder +1):
        #z       = zern( i, Zsize)
        sf_norm = (1/(z[i-1]*z[i-1]).sum())**0.5 #for each zernike
        #if i is 1:
            #sf_cal  = 1/(sf_norm*z[i-1]).sum()
        listOa.append( ( sf_norm*z[i-1]*zerg ).sum()*sf_cal )

    #print( 'Z4={4.3f}'.format( np.asarray( listOa[3] ) ) )

    return np.asarray( listOa )

def zList(nZ=21, sz=100):
    '''
    Generic Zernike list defined over a circle with radius sz/2 and size of sz
    Zernike RMS amplitude is normalised to 1
    '''
    pist=zern(1,sz).astype(int)

    Zl=np.zeros( (nZ, pist.sum()) )
    for i in range(nZ):
        zz = zern( int(i+2), sz)
        zz -= zz[pist>0].mean()
        if i != 0:
            zz *= 1/(zz[pist>0].std())
        Zl[i]=zz[pist>0]

    return Zl

def zList_old(Zorder=34, Zsize=50):
    '''
    creates a list of Zernike distributions with the default parameters
    returns numpy array
    '''
    z = np.zeros( (Zorder,Zsize,Zsize) )
    for i in range( 1, Zorder+1):
        z[i-1] =  zern.zernike_noll( i, Zsize )

    return z

def Z_formater( z ):
    '''
    returns Zernike wave front in compliance with the AOCC SH
    '''
    r=50
    rr=27
    #z = zern(Z, N)*A*10000

    zz = misc.imresize( z, (nSA+2,nSA+2), interp='nearest', mode='F')[1:-1,1:-1]*ellipMask( rr, rr, r,r, r/2, r/2 )
    #zz[6][6]=0

    #rotFlip( zz, 180, True )
    #print( stdAmp(z, nSA), stdAmp(zz, nSA+2) )
    return zz[zz!=0]

listOfZernike = None

#from astropy.io import fits

def store_complex_array(filename, arr):
    real = arr.real
    imag = arr.imag
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(real), fits.ImageHDU(imag)])
    hdul.writeto(filename)
    print('DATA\tsave success!')

def read_complex_array(filename):
    with fits.open(filename) as hdul:
        return hdul[1].data + hdul[2].data * 1j

def readFits( fName=None ):
    if fName is None or data['DATA'] is None:
        print('fName or data[\'DATA\'] is None, dunno what to do...')
        return
    if not os.path.exists( fName ):
        raise IOError( fName+" file of the same name exists")

    data = fits.getdata( fName )
    hdr  = fits.getheader( fName )

    return data, hdr

#def saveFits( fName, data, data_dic ):
    #'''
    #save the fits file and adds the entries from data_dic to the FITS header
    #'''
    #if os.path.exists( fName ):
        #print('file of the same name exists')
        #return
    #else:
        #fits.writeto( fName, data )
        #for e in data_dic:
            #fits.setval( fName, e, value = str( data_dic[e] ) )

def saveFits( data, fName=None ):
    '''
    saves the data['DATA'] in a FITS file and adds the entries from data['HDR'] to the FITS header

    :input:
        *data (list, tuple or numpy array) needs to be homogeneous
        *fName (str) file name, if None then auto generated

    :output: a FITS file with header
    '''
    #print(data.keys())
    if type(data) is dict:
        if fName is None or data['DATA'] is None:
            print('fName or data[\'DATA\'] is None, dunno what to do...')
            return
        hdr  = data['HDR'].copy()
        data = data['DATA'].copy()

    #if type(data) is list:
        #data=np.asarray(data)

    if fName is None or data is None:
        print('fName or data is None, dunno what to do...')
        return

    try:
        if fName[-4] != 'fits':
            fName += '.fits'
    except:
        fName += '.fits'

    if os.path.exists( fName ):
        raise IOError( fName+" file of the same name exists")
    else:
        try:
            fits.writeto( fName, data )
            print('DATA\tsave success!')
        except KeyError:
            print('resorting to store_complex_array()')
            store_complex_array( fName, data )

        try:
            print(hdr)
            for e in hdr:
                print(e)
                fits.setval( fName, e, value = str( hdr[e] ) )
            print('HDR\tsave success!')
            print('PATH: '+fName)
        except:
            print('Header is None, saving ONLY data[\'DATA\']')

def getModeMatrix(profiles, npola = 1, shift = None, angle = None):
    '''
    TAKEN: from pyMMF
        Returns the matrix containing the mode profiles.
        Note that while you can set two polarizations, the modes profiles are obtained under a scalar apporoximation.

        Parameters
        ----------

        npola : int (1 or 2)
            number of polarizations considered. For npola = 2, the mode matrix will be a block diagonal matrix.

        shift : list or None
            (slow) value of a coordinate offset, allows to return the mode matrix for a fiber with the center shifted with regard
            to the center of the observation window.
            defaults to None

        angle: float or None
            (slow) angle in radians, allows to rotate the mode matrix with an arbitrary angle.
            Note that the rotation is applied BEFORE the transverse shift.
            defaults to None


        Returns
        -------

        M : numpy array
        the matrix representing the basis of the propagating modes.
    '''
    if profiles is None:
        print('profiles are None, dunno what to do')

    if shift is not None:
        assert(len(shift)==2)

    N = profiles.shape[1]
    number = profiles.shape[0]
    #print( profiles.shape, N , number )
    #return
    M = np.zeros((N*npola, npola*number), dtype = np.complex128)
    angle = angle/np.pi*180. if (angle is not None) else None


    for pol in range(npola):

        for ind, modeProfile in enumerate(profiles):

            if (shift is None and angle is None):
                M[pol*N:(pol+1)*N,pol*number+ind] = modeProfile#.reshape(1,self._npoints**2)
            else:
                mode2D = modeProfile.reshape([int(np.sqrt( N ))]*2)

                if angle is not None:
                    mode2D = \
                        ndimage.rotate(mode2D.real, angle, reshape=False) + \
                        complex(0,1)*ndimage.rotate(mode2D.imag, angle, reshape=False)

                if shift is not None:
                    mode2D = \
                        ndimage.shift(input=mode2D.real, shift=shift) \
                        + complex(0,1)*ndimage.shift(input=mode2D.imag, shift=shift)

                M[pol*N:(pol+1)*N,pol*number+ind] = mode2D.flatten()

    #self.modeMatrix = M
    return M

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def scale(im, new_shape):
  nR0 = len(im)     # source number of rows
  nC0 = len(im[0])  # source number of columns

  nR, nC = new_shape

  return np.asarray( [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)] for c in range(nC)] for r in range(nR)] )

INTERP_KIND = {1: 'linear', 3: 'cubic', 5: 'quintic'}
def zoom(array, newSize, order=3):
    """
    A Function to zoom 2-dimensional arrays using interpolation

    Uses the scipy `Interp2d` interpolation routine to zoom into an array. Can cope with real of complex data.

    Parameters:
        array (ndarray): 2-dimensional array to zoom
        newSize (tuple): the new size of the required array
        order (int, optional): Order of interpolation to use. default is 3

    Returns:
        ndarray : zoom array of new size.
    """

    if order not in INTERP_KIND:
        raise ValueError("Order can either be 1, 3, or 5 only")

    try:
        xSize = newSize[0]
        ySize = newSize[1]

    except (IndexError, TypeError):
        xSize = ySize = newSize

    coordsX = np.linspace(0, array.shape[0] - 1, xSize)
    coordsY = np.linspace(0, array.shape[1] - 1, ySize)

    # If array is complex must do 2 interpolations
    if array.dtype == np.complex64 or array.dtype == np.complex128:

        realInterpObj = interpolate;interp2d(np.arange(array.shape[0]),
                                 np.arange(array.shape[1]), array.real, copy=False,
                                 kind=INTERP_KIND[order])
        imagInterpObj = interpolate.interp2d(np.arange(array.shape[0]),
                                 np.arange(array.shape[1]), array.imag, copy=False,
                                 kind=INTERP_KIND[order])
        return (realInterpObj(coordsY, coordsX)
                + 1j * imagInterpObj(coordsY, coordsX))



    else:

        interpObj = interpolate.interp2d(np.arange(array.shape[0]),
                             np.arange(array.shape[1]), array, copy=False,
                             kind=INTERP_KIND[order])

        # return np.flipud(np.rot90(interpObj(coordsY,coordsX)))
        return interpObj(coordsY, coordsX)
