# collection of functions
# mostly created by Lazar
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import os

import pickle

from astropy.io import fits
from aotools import image_processing as imp
from matplotlib.colors import LogNorm
from numpy import mean, square
from numpy import sqrt as root
from scipy import misc
from scipy import ndimage
from scipy.signal import convolve2d
from aotools import zernike as zern

from ABBOT.gradientOperator import gradientOperatorType1 as gOT1

#import darc

listOfZernike   = None
nominalZlist    = None
Zlist46         = None

SH_scaling      = 138.8 # SH WFS calibration factor in [nm]
Mag_factor      = 1#2.86  # DM to SH magnification

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
    WFS_drawSHmap( (cen[i] - cen[0]), DRAW=False )
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
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
    #vec.take( range(0//N,2)+range(1//N,2) )
    #ret

def saveFits( fName, data, data_dic ):
    '''
    save the fits file and adds the entries from data_dic to the FITS header
    '''
    if os.path.exists( fName ):
        print('file of the same name exists')
        return
    else:
        fits.writeto( fName, data )
        for e in data_dic:
            fits.setval( fName, e, value = str( data_dic[e] ) )


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

#OPT_CENT=(0,0)
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

def insertImage( displ, imdim=[500,500], vpos=0, hpos=0, angle = -1.7, x0=int( 250 ), y0=int( 250 ), oled2pd=1. ):
    '''
    returns an insert of an image centered around coordinates (x0, y0)
    defaults correspod to the PSF location on PD screen
    can be used together with toOLED()
    '''
    calibOLED = np.zeros( ( OLED_shape ) )

    x_scale = round_up_to_even( imdim[0]/oled2pd )
    y_scale = round_up_to_even( imdim[1]/oled2pd )
    ins = np.flip( np.flip( misc.imresize( displ, (x_scale, y_scale), interp='nearest', mode='F'), 0), 1)

    x_ofs = int( x_scale/2 )
    y_ofs = int( y_scale/2 )

    calibOLED[ x0 - x_ofs+vpos:x0 + x_ofs+vpos , y0 - y_ofs+hpos:y0 + y_ofs+hpos ] = ndimage.rotate( ins, angle, reshape=False )

    return calibOLED

def stdAmp( img, rr=52 ):
    '''
    returns the RMS amplitude of img in nm, elMasq is applied
    '''
    elMasq  = SAflag( rr ).reshape( (rr,rr) )

    return (np.ma.masked_invalid(img[elMasq == 1]).std(), np.ma.masked_invalid(img[elMasq == 1]).ptp())

def drawSHmap( cent, rr=52, DRAW=True, ax='xy' ):
    '''
    draws a 2D centroid map, the centroids must be Alified
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
        plt.figure('SH map')
        plt.clf()
        #plt.imshow( 10**np.log10( subapmap -0.0000001) )
        plt.imshow( ( subapmap/cm ) )#, clim=[0, .1] )
        plt.colorbar()
        #plt.pause(0.01)

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
    '''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""|
    !!! UNDER CONSTRUCTION !!!                                  |
    returns inverted Cholesky matrix for de-aliasing of results |
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
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

def linFitre(x, A, B):

    return A+B*x

def StoWF( cent, nSA=52, DRAW=False ):
    '''
    gets centroids returns wavefront
    '''
    slopesV = cent*SH_scaling*Mag_factor
    valid_subap_map=SAflag(nSA).reshape( (nSA,nSA) )

    gO=gOT1( valid_subap_map )
    gM=gO.returnOp()

    if not os.path.exists( '/home/ls/dev/alpaoTST/oldData/gM_i_nSA4064.fits' ):
        print('gradient ma3x NOT FOUND, recalculating...')
        gM_i=np.linalg.pinv( gM )
    else:
        gM_i = fits.getdata( '/home/ls/dev/alpaoTST/oldData/gM_i_nSA4064.fits' )
        #print( 'gi 36 loaded' )
    wfV=gM_i.dot(slopesV)
    if slopesV.ndim == 1:
        wf2D=np.zeros(gO.n_)
        wf2D.ravel()[gO.illuminatedCornersIdx]=wfV
        if DRAW:
            plt.figure('wf')
            plt.clf()
            plt.imshow(wf2D)
            plt.title('RMS={:3.1f}, PtV={:3.1f}'.format( stdAmp(wf2D,53), wf2D.ptp()))
            plt.colorbar()
            plt.pause(0.01)
        return wf2D

    elif slopesV.ndim == 2:
        shpe = list(gO.n_)#.copy()
        tmp=np.zeros(gO.n_)
        (shpe).append( slopesV.shape[1] )
        wf2D=np.zeros( shpe )
        #print(wf2D.shape)
        #wf2D.reshape( shpe[0]*shpe[1], -1 ).ravel()[gO.illuminatedCornersIdx]=wfV
        for i in range( slopesV.shape[1] ):
            #print(wf2D[:,:,i].shape, wfV[i].shape )
            tmp.ravel()[gO.illuminatedCornersIdx]=wfV.T[i].copy()
            wf2D[:,:,i] = tmp.copy()
            #((wf2D[:,:,i]).ravel())[gO.illuminatedCornersIdx]=wfV.T[i].copy()

        return wf2D.T#, wfV#, gM


def WFtoS( wf, nSA=52, DRAW=False, valid_subap_map=None ):
    '''
    returns slopes from a wavefront
    '''
    #valid_subap_map=SAflag(nSA).reshape( (nSA,nSA) )
    if wf is None or valid_subap_map is None:
        print('insufficient input, aborting...')
        return -1

    gO=gOT1( valid_subap_map )
    gM=gO.returnOp()

    ##this below matches the darc Sub Aperture settings in the gradient space
    #a=26.7                                       # this
    #masq = ellipMask( a, a, 51, 51, 25.5, 25.5 ) # and this

    masq = wf**2
    masq[masq>0]=1

    wfv     = wf[masq==1]
    synslop = ( wfv ).dot( gM.T )/SH_scaling

    if DRAW:
        plt.figure('syn')
        plt.clf();
        plt.plot( synslop )
        plt.pause(0.1)

        plt.figure('wf')
        plt.clf()
        plt.subplot( 1,3,1 )
        plt.imshow( wf )
        plt.title('In, RMS{:3.2f}'.format( wf[masq>0].std() ) )
        plt.subplot( 1,3,2 )
        #recWF = StoWF( synslop, nSA=nSA )
        #plt.imshow( recWF  )
        #plt.title('Out, RMS{:3.2f}'.format( stdAmp( recWF[1:-1,1:-1], recWF.shape[0]-2 ) ) )
        #plt.subplot( 1,3,3 )
        #plt.imshow( recWF[1:-1,1:-1] - wf )
        #plt.title('Diff, RMS{:3.2f}'.format( stdAmp( recWF[1:-1,1:-1] - wf, wf.shape[0] ) ) )
        plt.pause(0.1)

        #plt.figure('decomposition')
        #plt.plot( ( decompoZergOld( StoWF( synslop ) ) ), '.:' )
        #plt.pause(0.1)

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

def decompoZergOld( zerg, Zorder=46, Zsize=53, mask=None, DRAW=False ):
    '''
    decomposition of Zernike
    '''

    listOz = []
    listOa = []
    global nominalZlist
    global Zlist46

    if Zorder == 46 and Zsize == 53:
        if Zlist46 is not None:
            z = Zlist46.copy()
        else:
            print('recalculating zList({:}, {:}), please have patience...'.format( Zorder, Zsize))
            Zlist46= zList(Zorder+1,Zsize)
            z = Zlist46.copy()
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

def decompoZerg(wf, Zorder=46, Zsize=53, mask=None):
    '''
    Orthonormalisation of pixelised Zernike distributions and decomposition of a wave front against it
    Input parameters:
        wf      - the wavefron to be decomposed
        Zorder  - the highest order of Zernike to be included
        Zsize   - the size of the discretisation matrix
        mask    - which elements of the matrix to be included in the returned one

    returns the Zernike decomposition of a pixelised WF
    '''

    if mask is None:
        mask = zern( 1, Zsize )

    zn      = zern( 1, Zsize )
    sf_norm = (1/(zn*zn).sum())**0.5

    zMatx = np.zeros( ( Zorder-1, int( mask.sum() ) ) )
    for i in range( 2, Zorder + 1):
        zMatx[i-2] = sf_norm*( rotFlip( zern( i, Zsize ), 180, True) )[ mask==1 ]

    print( mask.shape)
    return np.dot( np.dot( np.linalg.inv( np.dot( zMatx, zMatx.T) ), zMatx ), wf[ mask==1 ])

#def decompoZerg( zerg, Zorder=46, Zsize=53 ):
    #'''
    #Performs a Zernike decomposition of zerg, the dimensions of zerg must match Zsize
    #Parameters:
        #zerg    - input 2D array that needs to be decomposed
        #Zorder  - the maximum Zernike order that will be decomposed
        #Zsize   - the size of the arrays
    #Returns
        #array with the result from the decomposition
    #TODO:
        #calculate the list of Zernike polynomials only when needed
    #'''

    #listOa = []

    ##if listOfZernike is None:
        ##listOfZernike = zList(Zorder, Zsize)
    ##elif listOfZernike is not None and (listOfZernike.shape[0] != Zorder or listOfZernike.shape[1] != Zsize ):
    #listOfZernike = zList(Zorder, Zsize)

    #z = listOfZernike.copy()

    #sf_norm = (1/(z[0]*z[0]).sum())**0.5
    #sf_cal  = 1/(sf_norm*z[0]).sum()
    #for i in range( 1, Zorder +1):
        ##z       = zern( i, Zsize)
        #sf_norm = (1/(z[i-1]*z[i-1]).sum())**0.5 #for each zernike
        ##if i is 1:
            ##sf_cal  = 1/(sf_norm*z[i-1]).sum()
        #listOa.append( ( sf_norm*z[i-1]*zerg ).sum()*sf_cal )

    ##print( 'Z4={4.3f}'.format( np.asarray( listOa[3] ) ) )

    #return np.asarray( listOa )

def zList(Zorder=34, Zsize=50):
    '''
    creates a list of Zernike distributions with the default parameters
    returns numpy array
    '''
    z = np.zeros( (Zorder,Zsize,Zsize) )
    for i in range( 1, Zorder+1):
        z[i-1] =  zern( i, Zsize )

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
