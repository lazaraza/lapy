import time
import serial
try:
    import darc
except:
    print( 'no darc!!!!!' )

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class wire:
    '''
    generic WFS class
    '''
    def __init__(self, **kwargs):

        def_keys = set(['device', 'comment', 'I0','U0', 'spI', 'spV'])

        # initialize all allowed keys to false
        self.__dict__.update((key, None) for key in def_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in def_keys)

        self.__dict__.update(kwargs)

        if self.device is None:
            print('You need to provide a device to connect to')
            return None

        self.dev= serial.Serial(self.device, timeout=1)
        self.getStatus()
        self.dev.flushInput()

    def getStatus(self):
        self.dev.write("OUT?\r\n")
        time.sleep(0.1)
        self.state = self.read()[:-2]

    def toggleONOFF(self, action=None):
        self.getStatus()
        if action is None:
            if self.state == 'OUT ON':
                self.dev.write("OFF\r\n")
            elif self.state == 'OUT OFF':
                self.dev.write("ON\r\n")
            else:
                print('Unknown state')
        elif action=='ON':
            self.dev.write("ON\r\n")
        elif action=='OFF':
            self.dev.write("OFF\r\n")
        else:
            print('Error: action unknown. Only action=\'ON\' or action=\'OFF\' are recognised!')
        self.getStatus()

    def getOutput(self, PRINT=False):
        '''
        acquire the output values:
        outI - output current
        outV - output voltage
        '''
        self.dev.write("IO?\r\n")
        time.sleep(0.1)
        ret = (self.read())
        err = self.getErr()

        if err == 0:
            #print(ret[2:6])
            self.outI = float( ret[2:7] )
            if PRINT:
                print('outI: {:.2f}, [A]'.format( self.outI) )

        self.dev.write("VO?\r\n")
        time.sleep(0.1)
        ret = (self.read())
        err = self.getErr()

        if err == 0:
            #print(ret[2:6])
            self.outV = float( ret[2:7] )
            if PRINT:
                print('outV: {:.2f}, [V]'.format( self.outV) )

        return self.outI, self.outV

    def getSP(self):
        '''
        acquire the Set Point values:
        spI - set point current
        spV - set point voltage
        '''
        self.dev.write("I?\r\n")
        time.sleep(0.1)
        ret = (self.read())
        err = self.getErr()
        #print( err )
        if err == 0:
            #print(ret[2:6])
            self.spI = float( ret[2:7] )
            print('spI: {:.2f}, [A]'.format( self.spI) )

        self.dev.write("V?\r\n")
        time.sleep(0.1)
        ret = (self.read())
        err = self.getErr()
        #print( 'err: ',err )
        if err == 0:
            #print(ret[2:6])
            self.spV = float( ret[2:8] )
            print('spV: {:.2f}, [V]'.format( self.spV) )


    def getErr(self):
        self.dev.write("ERR?\r\n");
        err = int(self.read()[4:5])
        if err == 0:
            #print('passed')
            pass
        elif err == 1:
            print('ERROR: Command not recognised!')
        elif err == 2:
            print('ERROR: Value outside instrument limits!')
        else:
            print( 'Error: UNKNOWN' )
        return int(err)


    def read(self):
        return self.dev.readline()

    def setSP(self, spI=1., spV=1.):
        '''
        acquire the Set Point values:
        spI - set point current
        spV - set point voltage
        '''

        print("I {:.2f}\r\n".format(spI))
        print("V {:.2f}\r\n".format(spV))

        self.dev.write("I {:.2f}\r\n".format(spI))
        time.sleep(0.1)
        self.getErr()
        self.dev.write("V {:.2f}\r\n".format(spV))
        self.getErr()
        #print("V {:.2f}\r\n".format(spV))

        self.getSP()

class WFS():
    '''
    generic WFS class
    '''
    def __init__(self, name='turb', comment='no comment', wfs=0, refCfn=None, refAfn=None, confile=None):
        '''
        Input:
            name   - the DARC prefix
            comment- few words describin the purpose of this object
            refCfn - reference centroids file name
        '''
        #self.conf = conf
        self.darcList   = []
        self.ncam       = []

        self.comment=comment

        if refCfn is not None:
            try:
                self.refC = fits.getdata(refCfn).mean(0)
            except IOError:
                print('File error, reference centroids inaccessible, setting to zero')
                self.refC = 0 # TODO see how

        if confile is None:
            print( 'initialising the following DARC instance:', name)

            self.d          = darc.Control( name )
            self.ncam       = self.d.Get('ncam')
            self.ncent      = int(self.d.GetStream( "rtcCentBuf" )[0].shape[0]/self.d.Get('ncam'))
            self.cents      = self.getCents( 10 )['DATA']# (0,1,2,3))

            self.iniSAL     = self.d.Get("subapLocation")

            self.iniref     = self.d.Get('refCentroids')
            self.oldref     = None# self.iniref.copy()
            self.ONLINE     = True
            self.nsubaps    = self.d.Get("nsub1D")

        else:
            #import confile as d
            self.dc          = __import__(confile)
            self.ncam       = self.dc.ncam
            self.ncent      = int(self.dc.ncents/self.ncam)
            self.cents      = np.zeros( (0, self.ncent) )

            self.iniSAL     = self.dc.subapLocation
            self.ONLINE     = False
            self.nsubaps    = self.dc.xsubaps

        self.getSAcoor()
        self.getSAmask()
        print( 'NoCam: {:}, NoCents: {:}, {:}, STATUS: {:}'.format( self.ncam, self.ncent, self.cents.shape, self.ONLINE ) )


    def loadRefCents(self, refC=None):
        '''
        Loads reference centroids to DARC
        Input parameter:
                refC - reference centroids to be loaded:
                        refC='ini'       - the centroids acquired during initialisation of the WFS object are loaded
                        refC='last'      - the previously loaded centroids are loaded back
                        refC=numpy.array(10024,) - is uploaded to the RTC
        '''
        helps = 'unable to load reference centroids, None given as input.\n please provide the reference centroids as numpy array or as string \'ini\' to load the initial values, or \'last\' to load the previous values.'
        if refC == 'ini':
            self.d.Set('refCentroids', self.iniref)
        elif refC == 'last':
            self.d.Set('refCentroids', self.oldref)
        elif type(refC) is np.ndarray and refC.shape[0]==10024:
            self.d.Set('refCentroids', refC)
        elif refC is None:
            print( helps )
        else:
            print( helps )

    def getCents(self, nFr=1,  wfs_i=0, avg=False, ref=False):
        '''
        Acquires centroids from the designated RT instance.
        Parameters:
                nFr     - number of frames to be acquired
                wfs_i   - WFS index:
                        0    - TWFS
                        1..3 - LGS0..LGS2
                        4..6 - NGS0..NGS2
                avg     - should the reuslt be averaged
                ref     - are the data taken to be reference, if True then the RTC reference is set to Zero and then measurement is taken
        returns (X, Y).centroids and updated data header
        '''

        if ref:
            self.oldref = self.d.Get('refCentroids')
            print('setting DARC\'s reference centroids to ZERO!')
            d.Set('refCentroids', oldRefCents*0.)


        #hdr = dataMan(meas_type='centroid data', comment=self.comment)
        #self.centHdr = hdr.dataHdr

        data = self.d.GetStreamBlock('rtcCentBuf', nFr, asArray=1 )['rtcCentBuf'][0]
        if 0<= wfs_i <7:
            # slice with the centroids from only wfs_i
            return {'DATA': data[:,wfs_i*self.ncent: wfs_i*self.ncent+self.ncent] }
        elif wfs_i == 7:
            return {'DATA': data }

    def getFlux(self, nFr=1,  wfs_i=0, avg=False):
        '''
        self.explanatory
        returns (X, Y).flux
        '''
        #d=self.d
        #hdr = dataMan(meas_type='flux data', comment=self.comment)
        #self.fluxHdr = hdr.dataHdr

        data = self.d.GetStreamBlock('rtcFluxBuf', nFr, asArray=1 )['rtcFluxBuf'][0]
        if 0<= wfs_i <=3:
            # slice with the centroids from only wfs_i
            return {'DATA': data[:,wfs_i*self.ncent: wfs_i*self.ncent+self.ncent] }
        elif wfs_i == 4:
            return {'DATA': data }

    def cLoop( self, cmat, lGain=0.5 ):

        ref_c  = self.refC.copy()*0.
        ref_c[::2]=ref_c[::2].mean()
        ref_c[1::2]=ref_c[1::2].mean()

        curCen = self.getCents( 100 ).mean(0)
        runAct = self.dm.act0

        newAct  = (curCen - ref_c ).dot( cmat )
        runAct[:277] += newAct*lGain

        self.d.Set('actuators', runAct)

        print('closing the loop,  ^C to EXIT')
        begin = time.time()
        while True:
            now = time.time()
            curCen = self.getCents( 2 ).mean(0)
            newAct = (-curCen + ref_c ).dot( cmat )
            runAct[:277] += newAct*lGain
            if abs(runAct).max() >= 0.99:
                print('actuators saturated')
                self.d.Set( 'actuators', self.dm.act0 )
                break

            self.d.Set('actuators', runAct)
            try:
                time.sleep(0.01)
                print( now-begin, abs(runAct).max() )
            except KeyboardInterrupt:
                self.d.Set( 'actuators', self.dm.act0 )
                print('breaking the lopp, breaking the loop, TADADADA!!!!')
                return

    def salAdjust(self, CamNum = 0, ax0=0, ax1=0, Zax0=1., Zax1=1.):
        '''
        simple translation of the sub-app grid in the RTC
        parameters:
            CamNum  - which WFS to be adjusted

            ax0     - one of the translation axes
            ax1     - the other translation axis
            Zax0    - zoom in axis0, baby steps +/- 1.01 should do
            Zax1    - zoom in the other axis, baby steps +/- 1.01 should do
        '''

        if CamNum is None:
            print('You need to specify CamNum! Restoring the initial sub-ap grid!')
            self.d.Set("subapLocation", self.iniSAL)
        else:
            SAL=self.d.Get("subapLocation");
            SAL[716*CamNum:716*(CamNum+1),[3,4]]+=ax1
            SAL[716*CamNum:716*(CamNum+1),[0,1]]+=ax0
            if Zax0 > 0.9 and Zax0 < 1.1:
                SAL[716*CamNum:716*(CamNum+1),[0,1]]=( (SAL[716*CamNum:716*(CamNum+1),[0,1]]-488)*(Zax0) ).astype('i')+488
            if Zax1 > 0.9 and Zax1 < 1.1:
                print( 'zooming by: ', Zax1 )
                SAL[716*CamNum:716*(CamNum+1),[3,4]]=( (SAL[716*CamNum:716*(CamNum+1),[3,4]]-488)*(Zax1) ).astype('i')+488

            self.d.Set("subapLocation",SAL)
            print('moving by: ax0={:}, ax1={:}'.format(ax0,ax1) )


    def getSAcoor(self):
        '''
        returns the 2D pupil in terms of X,Y coordinates of the SA's
        '''
        nsa = int(self.ncent/2)
        sal = self.iniSAL
        sap = np.zeros( (2, nsa) )
        sap[0] = sal[:nsa,0:2].mean(1)
        sap[1] = sal[:nsa,3:5].mean(1)


        self.sacoor = sap.copy()
        self.sacoor[0] = ((sap[0]-sap[0][0])/15)
        self.sacoor[1] = ((sap[1]-sap[1].min())/15)


    def getSAmask(self):
        '''
        returns the 2D pupil in terms of X,Y coordinates of the SA's
        '''
        self.samask = np.zeros( (self.nsubaps, self.nsubaps) )
        ii=0
        for i in self.sacoor.T:
            self.samask[ int(i[0]), int(i[1])] = 1
            ii+=1


    def showFlux(self, flux=None):
        if flux is None:
            flux = (self.getFlux( nFr=10,  wfs_i=0, avg=True ))['DATA']

        print( flux.ndim )
        if flux.ndim == 2:
            flux = flux.mean(0)
            print('avg-ing Flux' )

        pupc = self.sacoor

        pupF = np.ones( (self.nsubaps, self.nsubaps) )*-1
        print( flux.shape )
        ii=0
        for i in pupc.T:
            pupF[ int(i[0]), int(i[1])] = flux[ii]
            ii+=1

        cmap = matplotlib.cm.Greens_r
        cmap.set_bad(color='red')

        plt.figure('flux')
        plt.clf()
        plt.imshow( np.ma.masked_where(pupF == -1, pupF), cmap=cmap )
        print('lux: ', flux.shape)
        plt.clim( flux[:716].min(), flux[:716].max() )
        plt.colorbar()
        return pupF

    def readTemp(self, nProbes=6, port='/dev/ttyUSB0' ):
        '''
        DEPRECATED: now handled by Pyro server

        reads temperature from /dev/ttyUSB0 returns array with the readings
            1: wire
            2: bench
            3: WFS focus
            4: WFS pole
            5: duct WFS
            6: duct wire
        '''
        self.tempLabels = ['wire',
                           'bench',
                           'WFS focus',
                           'WFS pole',
                           'duct WFS',
                           'duct wire'
                           ]

        ser = serial.Serial(port, 9600, bytesize = serial.SEVENBITS, parity = serial.PARITY_EVEN, stopbits = serial.STOPBITS_ONE, dsrdtr = False, timeout=0.1)
        ser.write("CRDG? 0\r\n")
        tempss = ( str(ser.readline())[:-2] ).split(',')

        temperatures = np.zeros( nProbes )
        for i in range( nProbes ):
            temperatures[i] = float( tempss[i] )

        return temperatures

'''

#this (alldist) is supposed to be all possible distances between the subapps

import turbV
wfs = turbV.WFS('turb')

alldist = []; countr = wfs.sacoor.T.shape[0]; begin=1
for i, sc in enumerate( wfs.sacoor.T ):
    print( begin )
    for j in range( begin, countr ):
         alldist.append( (sc**2+wfs.sacoor.T[j]**2)**0.5 )
    begin+=1

#OR

for i, sc in enumerate( wfs.sacoor.T ):
    print( begin )
    for j in range( begin, countr ):
        alldist.append( ( (sc**2).sum()+(wfs.sacoor.T[j]**2).sum())**0.5 )
    begin+=1

massdist = np.asarray( alldist )
plt.clf(); 
th = plt.hist( (massdist**2).sum(1)**0.5, bins=100 )
'''
