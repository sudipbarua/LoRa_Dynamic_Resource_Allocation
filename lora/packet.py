from numpy import zeros, random, where
from .loratools import getRXPower, dBmTomW, airtime
import numpy as np

class myPacket():
    """ LPWAN Simulator: packet
    Base station class
   
    |category /LoRa
    |keywords lora
    
    \param [IN] nodeid: id of the node
    \param [IN] bsid: id of the base station
    \param [IN] dist: distance between node and bs
    \param [IN] transmitParams: physical layer parameters
                [sf, rdd, bw, packetLength, preambleLength, syncLength, headerEnable, crc, pTX, period] 
    \param [IN] logDistParams: log shadowing channel parameters
    \param [IN] sensi: sensitivity matrix
    \param [IN] setActions: set of possible actions
    \param [IN] nrActions: number of actions
    \param [IN] sfSet: set of spreading factors
    \param [IN] prob: probability
    """

    def __init__(self, nodeid, bsid, dist, transmitParams, logDistParams, sensi, setActions, nrActions, sfSet, prob): 
        self.nodeid = nodeid
        self.bsid = bsid
        self.dist = dist
        
        # params
        self.sf = int(transmitParams[0])
        self.rdd = int(transmitParams[1])
        self.bw = int(transmitParams[2])
        self.packetLength = int(transmitParams[3])
        self.preambleLength = int(transmitParams[4])
        self.syncLength = transmitParams[5]
        self.headerEnable = int(transmitParams[6])
        self.crc = int(transmitParams[7])
        self.pTXmax = int(transmitParams[8])
        self.sensi = sensi
        self.sfSet = sfSet
        
        # learn strategy
        self.setActions = setActions
        self.nrActions = nrActions
        self.prob = [prob[x] for x in prob]
        #self.choosenAction = choosenAction
        #self.sf, self.freq, self.pTX = self.setActions[self.choosenAction]
        self.sf = None
        self.freq= None
        self.pTX = self.pTXmax
        
        #received params
        self.rectime = airtime(transmitParams[0:8])
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)
        self.signalLevel = None

        # measurement params
        self.packetNumber = 0
        self.isLost = False
        self.isCritical = False
        self.isCollision = False
                
    def computePowerDist(self, bsDict, logDistParams):
        """ Get the power distribution .
        Parameters
        ----------
        self : packet
            Packet.
        bsDict: dictionary
            Dictionary of BSs
        Returns
        -------
        signalLevel: dictionary
            The power contribution of a packet in various frequency buckets for each BS
    
        """
        signal = self.getPowerContribution()
        signalLevel = {x:signal[x] for x in signal.keys() & bsDict[self.bsid].signalLevel.keys()}
        return signalLevel
        
    def updateTXSettings(self, bsDict, logDistParams, prob):
        """ Update the TX settings after frequency hopping.
        Parameters
        ----------
        bsDict: dictionary
            Dictionary of BSs
        logDistParams: list
            Channel parameters, e.x., log-shadowing model: (gamma, Lpld0, d0)]
        
        Returns
        isLost: bool
            Packet is lost ot not by compare the pRX with RSSI
        -------
    
        """
        self.packetNumber += 1
        self.prob = prob
        self.choosenAction = random.choice(self.nrActions, p=self.prob)
        self.sf, self.freq, self.pTX = self.setActions[self.choosenAction]
        print("[myPacket updateTXSettings] Node " + str(self.nodeid) + " chose action: " + str(self.choosenAction) + " with SF: " + str(self.sf) + ", Freq: " + str(self.freq) + ", pTX: " + str(self.pTX))
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)
        # print("[myPacket updateTXSettings] probability of node " +str(self.nodeid)+" is: " +str(self.prob))

        self.signalLevel = self.computePowerDist(bsDict, logDistParams)

        if self.pRX >= self.sensi[self.sf-7, 1+int(self.bw/250)]:
            self.isLost = False
        else:
            self.isLost = True
            print("[myPacket updateTXSettings] pRX", self.pRX)
            print("[myPacket updateTXSettings] Node " + str(self.nodeid) + ": packet is lost (smaller than RSSI theshold)!")
   
        self.isCritical = False
        
    def getAffectedFreqBuckets(self):
        """ Get the list of affected frequency buckets from [fc-bw/2 fc+bw/2].
        Parameters
        ----------
        
        Returns
        fRange: list
            List of frequencies that effected by the using frequency
        -------
        """
        low = self.freq - self.bw/2 # Note: this is approx due to integer division for 125
        high = self.freq + self.bw/2 # Note: this is approx due to integer division for 125
        lowBucketStart = int(low - (low % 200) + 100)
        highBucketEnd = int(high + 200 - (high % 200) - 100)

        # the +1 ensures that the last value is included in the set
        return range(lowBucketStart, highBucketEnd + 1, 200)
            
    def getPowerContribution(self):
        """ Get the power contribution of a packet in various frequency buckets.
        Parameters
        ----------

        Returns
        powDict: dic
            Power distribution by frequency
        -------
    
        """
        freqBuckets = self.getAffectedFreqBuckets()
        powermW = dBmTomW(self.pRX)
        #print(self.pRX, powermW)
        signal = zeros((6,1))
        full_setSF = [7, 8, 9, 10, 11, 12]
        idx = full_setSF.index(self.sf)
        #print(idx)
        signal[idx] = powermW
        #print(signal)
        return {freqBuckets[0]:signal}
    
    def getPktAirtime(self):
        # Airtime of the packet in miliseconds
        return airtime([self.sf, self.rdd, self.bw, self.packetLength, 
                        self.preambleLength, self.syncLength, self.headerEnable, 
                        self.crc])


class rlPacket(myPacket):
    """ LPWAN Simulator: ddqnPacket
    DDQN packet class, inherits from myPacket
    
    |category /LoRa
    |keywords lora
    
    \param [IN] nodeid: id of the node
    \param [IN] bsid: id of the base station
    \param [IN] dist: distance between node and bs
    \param [IN] transmitParams: physical layer parameters
                [sf, rdd, bw, packetLength, preambleLength, syncLength, headerEnable, crc, pTX, period] 
    \param [IN] logDistParams: log shadowing channel parameters
    \param [IN] sensi: sensitivity matrix
    \param [IN] setActions: set of possible actions
    \param [IN] nrActions: number of actions
    \param [IN] sfSet: set of spreading factors
    \param [IN] prob: probability
    """
    def __init__(self, nodeid, bsid, dist, transmitParams, logDistParams, sensi, setActions, nrActions, sfSet, agent, prob={'dummy': 'dummy'}):
        super().__init__(nodeid, bsid, dist, transmitParams, logDistParams, sensi, setActions, nrActions, sfSet, prob)
        self.agent = agent  # Reference to the RL agent from RL node 

    def updateTXSettings(self, bsDict, logDistParams, state):  # add state parameters like rssiHistory, snrHistory
        """ Update the TX settings after frequency hopping.
        Parameters
        ----------
        bsDict: dictionary
            Dictionary of BSs
        logDistParams: list
            Channel parameters, e.x., log-shadowing model: (gamma, Lpld0, d0)]
        
        Returns
        isLost: bool
            Packet is lost ot not by compare the pRX with RSSI
        -------
    
        """
        self.packetNumber += 1
        self.chosenAction = self.agent.act(np.array(state).reshape(1, -1))  # reshaping reqired in case state is a 1D array
        self.sf, self.freq, self.pTX = self.setActions[self.chosenAction]
        print(f"[{self.__class__.__name__} updateTXSettings] Node " + str(self.nodeid) + " chose action: " + str(self.chosenAction) + " with SF: " + str(self.sf) + ", Freq: " + str(self.freq) + ", pTX: " + str(self.pTX))
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)

        self.signalLevel = self.computePowerDist(bsDict, logDistParams)

        if self.pRX >= self.sensi[self.sf-7, 1+int(self.bw/250)]:
            self.isLost = False
        else:
            self.isLost = True
            print(f"[{self.__class__.__name__} updateTXSettings] pRX", self.pRX)
            print(f"[{self.__class__.__name__} updateTXSettings] Node " + str(self.nodeid) + ": packet is lost (smaller than RSSI theshold)!")
   
        self.isCritical = False


class rlPacketFreqHop(myPacket):
    def __init__(self, nodeid, bsid, dist, transmitParams, logDistParams, sensi, setActions, nrActions, sfSet, agent, prob={'dummy': 'dummy'}):
        super().__init__(nodeid, bsid, dist, transmitParams, logDistParams, sensi, setActions, nrActions, sfSet, prob)
        self.agent = agent  # Reference to the RL agent from RL node    

    def updateTXSettings(self, bsDict, logDistParams, state):
        self.packetNumber += 1
        self.chosenAction = self.agent.act(np.array(state).reshape(1, -1))  # reshaping reqired in case state is a 1D array
        self.freq = random.choice(self.agent.channels)
        self.sf, self.pTX = self.setActions[self.chosenAction]
        print(f"[{self.__class__.__name__} updateTXSettings] Node " + str(self.nodeid) + " chose action: " + str(self.chosenAction) + " with SF: " + str(self.sf) + ", Freq: " + str(self.freq) + ", pTX: " + str(self.pTX))
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)

        self.signalLevel = self.computePowerDist(bsDict, logDistParams)

        if self.pRX >= self.sensi[self.sf-7, 1+int(self.bw/250)]:
            self.isLost = False
        else:
            self.isLost = True
            print(f"[{self.__class__.__name__} updateTXSettings] pRX", self.pRX)
            print(f"[{self.__class__.__name__} updateTXSettings] Node " + str(self.nodeid) + ": packet is lost (smaller than RSSI theshold)!")
   
        self.isCritical = False


class basicPacket(myPacket):
    def __init__(self, nodeid, bsid, dist, transmitParams, logDistParams, sensi, sfSet, freqSet, adrEbable): 
        self.nodeid = nodeid
        self.bsid = bsid
        self.dist = dist
        
        # params
        self.sf = int(transmitParams[0])
        self.rdd = int(transmitParams[1])
        self.bw = int(transmitParams[2])
        self.packetLength = int(transmitParams[3])
        self.preambleLength = int(transmitParams[4])
        self.syncLength = transmitParams[5]
        self.headerEnable = int(transmitParams[6])
        self.crc = int(transmitParams[7])
        self.pTXmax = int(transmitParams[8])
        self.sensi = sensi
        self.sfSet = sfSet
        self.freqSet = freqSet
        
        self.sf = None
        self.freq= None
        self.pTX = self.pTXmax
        
        #received params
        self.rectime = airtime(transmitParams[0:8])
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)
        self.signalLevel = None

        # measurement params
        self.packetNumber = 0
        self.isLost = False
        self.isCritical = False
        self.isCollision = False

        self.adrEnable = adrEbable
        self.adrAckReq = 0

    def updateTXSettings(self, bsDict, logDistParams, adrAckCnt, oldSF, oldPtx):
        self.packetNumber += 1
        self.freq = random.choice(self.freqSet)
        if self.adrEnable:
            ############ ED ADR algorithm ############
            print("ED ADR algorithm in use.")
            adrAckLimit = 16 
            adrAckDelay = 8
            if adrAckCnt >= adrAckLimit:
                print("ADR acknowledgement limit is exceeded.")
                if self.pTX < self.pTXmax and self.sf < max(self.sfSet):
                    self.adrAckReq = 1
                    if adrAckCnt > (adrAckLimit + adrAckDelay):
                        if self.pTX < self.pTXmax:
                            self.pTX += 2
                        else:
                            self.sf += 1
                        adrAckCnt = adrAckLimit
                else:
                    self.pTX, self.sf = self.pTXmax, max(self.sfSet)
            else:
                # ADR acknowledgement limit is not exceeded so we keep the previous settings
                self.sf, self.pTX = oldSF, oldPtx
            ###########################################
        else:
            self.sf, self.pTX = random.choice(self.sfSet), self.pTXmax 
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)

        self.signalLevel = self.computePowerDist(bsDict, logDistParams)
        print(f"[{self.__class__.__name__} updateTXSettings] Packet " + str(self.nodeid) + " Old SF: " + str(oldSF) + " Old TxPow: " + str(oldPtx) + " chose SF: " + str(self.sf) + ", pTX: " + str(self.pTX))
        if self.pRX >= self.sensi[self.sf-7, 1+int(self.bw/250)]:
            self.isLost = False
        else:
            self.isLost = True
            print(f"[{self.__class__.__name__} updateTXSettings] pRX", self.pRX)
            print(f"[{self.__class__.__name__} updateTXSettings] Packet " + str(self.nodeid) + ": packet is lost (smaller than RSSI theshold)!")
   
        self.isCritical = False
