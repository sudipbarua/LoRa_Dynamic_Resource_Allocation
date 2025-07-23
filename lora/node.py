from __future__ import division
import numpy as np
from .loratools import getDistanceFromPower
from .packet import myPacket, rlPacket
from .Agent import LoRaDRL, AdaptiveResourceAllocation, LoRaQLAgent, DQNAgent

class myNode():
    """ LPWAN Simulator: node
    Base station class
   
    |category /LoRa
    |keywords lora
    
    \param [IN] nodeid: id of the node
    \param [IN] position: position of the node in format [x y]
    \param [IN] transmitParams: physical layer's parameters
    \param [IN] bsList: list of BS
    \param [IN] interferenceThreshold: interference threshold
    \param [IN] logDistParams: log shadowing channel parameters
    \param [IN] sensi: sensitivity matrix
    \param [IN] nSF: number of spreading factors
    
    """
    def __init__(self, nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, bsList,
                 interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, simu_dir, fname):
        print(f"[myNode __init__] nodeid={nodeid}, position={position}, node_mode={node_mode}, info_mode={info_mode}, algo={algo}")
        # print(f"[myNode __init__] transmitParams: {transmitParams}, bsList: {bsList}, interferenceThreshold: {interferenceThreshold}, logDistParams: {logDistParams}, sensi: {sensi}")
        self.nodeid = nodeid # id
        self.x, self.y = position # location
        if node_mode == 0:
            self.node_mode = initial
        else:
            self.node_mode = "SMART"
        print(f"[myNode __init__] Set node_mode: {self.node_mode}")

        self.info_mode = info_mode # 'no', 'partial', 'full'
        self.bw = int(transmitParams[2])
        self.period = float(transmitParams[9])
        self.pTXmax = max(powSet) # max pTX
        self.sensi = sensi

        print(f"[myNode __init__] bw={self.bw}, period={self.period}, pTXmax={self.pTXmax}")

        # generate proximateBS
        self.proximateBS = self.generateProximateBS(bsList, interferenceThreshold, logDistParams)
        # print(f"[myNode __init__] Generated proximateBS: {self.proximateBS}")

        # set of actions
        self.freqSet = freqSet
        self.powerSet = powSet

        if self.info_mode == "NO":
            self.sfSet = sfSet
        else:
            self.sfSet = self.generateHoppingSfFromDistance(sfSet, logDistParams)
        print(f"[myNode __init__] sfSet: {self.sfSet}")

        self.setActions = [(self.sfSet[i], self.freqSet[j], self.powerSet[k]) for i in range(len(self.sfSet)) for j in range(len(self.freqSet)) for k in range(len(self.powerSet))]
        self.nrActions = len(self.setActions)
        self.initial = initial
        print(f"[myNode __init__] setActions: {self.setActions}")
        print(f"[myNode __init__] nrActions: {self.nrActions}")

        # learning algorithm
        if algo == 'exp3':
            self.learning_rate = np.minimum(1, np.sqrt((self.nrActions*np.log(self.nrActions))/((horTime)*(np.exp(1.0)-1))))
            self.alpha = None
        elif algo == 'exp3s':
            self.learning_rate = np.minimum(1, np.sqrt((self.nrActions*np.log(self.nrActions*horTime))/horTime))
            self.alpha = 1/horTime
        print(f"[myNode __init__] learning_rate: {self.learning_rate}, alpha: {self.alpha}")

        # weight and prob for learning
        self.weight = {x: 1 for x in range(0, self.nrActions)}
        if self.initial=="RANDOM":
            prob = np.random.rand(self.nrActions)
            prob = prob/sum(prob)
        else:
            prob = (1/self.nrActions) * np.ones(self.nrActions)
        self.prob = {x: prob[x] for x in range(0, self.nrActions)}
        # print(f"[myNode __init__] Initial weights: {self.weight}")
        # print(f"[myNode __init__] Initial probs: {self.prob}")

        # generate packet and ack
        self.packets = self.generatePacketsToBS(transmitParams, logDistParams)
        self.ack = {}

        # measurement params
        self.packetNumber = 0
        self.packetsTransmitted = 0
        self.packetsSuccessful = 0
        self.transmitTime = 0
        self.energy = 0
        self.energyConsumedByThisPacket = 0  # Energy consumed for the last packet

    def generateProximateBS(self, bsList, interferenceThreshold, logDistParams):
        """ Generate dictionary of base-stations in proximity.
        Parameters
        ----------
        bsList : list
            list of BSs.
        interferenceThreshold: float
            Interference threshold
        logDistParams: list
            Channel parameters
        Returns
        -------
        proximateBS: list
            List of proximated BS
        """

        maxInterferenceDist = getDistanceFromPower(self.pTXmax, interferenceThreshold, logDistParams)
        dist = np.sqrt((bsList[:,1] - self.x)**2 + (bsList[:,2] - self.y)**2)
        index = np.nonzero(dist <= maxInterferenceDist)

        proximateBS = {} # create empty dictionary
        for i in index[0]:
            proximateBS[int(bsList[i,0])] = dist[i]
        print(f"[myNode generateProximateBS] ProximateBS result: {proximateBS}")
        return proximateBS

    def generatePacketsToBS(self, transmitParams, logDistParams):
        """ Generate dictionary of base-stations in proximity.
        Parameters
        ----------
        transmitParams : list
            Transmission parameters.
        logDistParams: list
            Channel parameters
        Returns
        -------
        packets: packet
            packets at BS
        """
        packets = {} # empty dictionary to store packets originating at a node
        for bsid, dist in self.proximateBS.items():
            packets[bsid] = myPacket(self.nodeid, bsid, dist, transmitParams, logDistParams, self.sensi, self.setActions, self.nrActions, self.sfSet, self.prob)
        print(f"[myNode generatePacketsToBS] Packets generated: {packets}")
        return packets

    def generateHoppingSfFromDistance(self, sfSet, logDistParams):
        """ Generate the sf hopping sequence from distance
        Parameters
        ----------
        logDistParams: list in format [gamma, Lpld0, d0]
            Parameters for log shadowing channel model.
        Returns
        -------
    
        """
        sfBuckets = []
        gamma, Lpld0, d0 = logDistParams
        dist = self.proximateBS[0] if 0 in self.proximateBS else 0
        print(f"[myNode generateHoppingSfFromDistance] Distance for SF hopping: {dist}")

        if self.bw == 125:
            bwInd = 0
        else:
            bwInd = 1
        Lpl = self.pTXmax - self.sensi[:, bwInd+1]

        LplMatrix = Lpl.reshape((6,1))
        distMatrix =np.dot(d0, np.power(10, np.divide(LplMatrix - Lpld0, 10*gamma)))

        for i in range(6):
            if dist <= distMatrix[0, 0]:
                minSF = 7
            elif distMatrix[i, 0 ]<= dist < distMatrix[i+1, 0]:
                minSF = (i + 1) + 7
        tempSF = [sf for sf in sfSet if sf >= minSF]
        sfBuckets.extend(tempSF)
        print(f"[myNode generateHoppingSfFromDistance] SF Buckets: {sfBuckets}")
        return sfBuckets

    def updateProb(self, algo):
        """ Update the probability of each action by using EXP3 algorithm.
        Parameters
        ----------
       
        Returns
        -------
    
        """
        prob = [self.prob[x] for x in self.prob]
        weight = [self.weight[x] for x in self.weight]
        reward = np.zeros(self.nrActions)
        # compute reward
        if self.node_mode == "SMART":
            # no and partial information case:
            if self.info_mode in ["NO", "PARTIAL"]:
                # with ACK -> 1, no ACK -> 0
                if self.ack:
                    reward[self.packets[0].choosenAction] = 1/prob[self.packets[0].choosenAction]
                else:
                    reward[self.packets[0].choosenAction] = 0
            # full information case:
            else:
                if self.ack:
                    if not self.ack[0].isCollision:
                        reward[self.packets[0].choosenAction] = 1/prob[self.packets[0].choosenAction]
                    else:
                        reward[self.packets[0].choosenAction] = 0.5/prob[self.packets[0].choosenAction]
                else:
                    reward[self.packets[0].choosenAction] = 0

        # update weight
        for j in range(0, self.nrActions):
            if algo == "exp3":
                weight[j] *= np.exp((self.learning_rate * reward[j])/self.nrActions)
            elif algo == "exp3s":
                weight[j] *= np.exp((self.learning_rate * reward[j])/self.nrActions)
                weight[j] += ((np.exp(1) * self.alpha)/self.nrActions) * sum(weight)

        # update prob
        if self.node_mode == "SMART":
            for j in range(0, self.nrActions):
                prob[j] = (1-self.learning_rate) * (weight[j]/sum(weight)) + (self.learning_rate/self.nrActions)
        elif self.node_mode == "RANDOM":
            prob = np.random.rand(self.nrActions)
            prob = prob/sum(prob)
        else:
            prob = (1/self.nrActions) * np.ones(self.nrActions)

        # trick: force the small value (<1/5000) to 0 and normalize
        prob = np.array(prob)
        prob[prob<0.0005] = 0
        prob = prob/sum(prob)
        self.weight = {x: weight[x] for x in range(0, self.nrActions)}
        self.prob = {x: prob[x] for x in range(0, self.nrActions)}
        # print(f"[myNode updateProb] Updated weights: {self.weight}")
        # print(f"[myNode updateProb] Updated probs: {self.prob}")

    def resetACK(self):
        print(f"[myNode resetACK] nodeid={self.nodeid}")
        self.ack = {}

    def addACK(self, bsid, packet):
        print(f"[myNode addACK] nodeid={self.nodeid}, bsid={bsid}")
        self.ack[bsid] = packet

    def updateTXSettings(self):
        print(f"[myNode updateTXSettings] nodeid={self.nodeid}")
        pass


class rlNode(myNode):
    """ LPWAN Simulator: node
    |category /LoRa
    |keywords lora
    
    \param [IN] nodeid: id of the node
    \param [IN] position: position of the node in format [x y]
    \param [IN] transmitParams: physical layer's parameters
    \param [IN] bsList: list of BS
    \param [IN] interferenceThreshold: interference threshold
    \param [IN] logDistParams: log shadowing channel parameters
    \param [IN] sensi: sensitivity matrix
    \param [IN] nSF: number of spreading factors
    
    """
    def __init__(self, nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, bsList,
                 interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, simu_dir, fname):
        print(f"[myNode __init__] nodeid={nodeid}, position={position}, node_mode={node_mode}, info_mode={info_mode}, algo={algo}")
        self.nodeid = nodeid # id
        self.x, self.y = position # location
        if node_mode == 0:
            self.node_mode = initial
        else:
            self.node_mode = "SMART"
        print(f"[myNode __init__] Set node_mode: {self.node_mode}")

        self.info_mode = info_mode # 'no', 'partial', 'full'
        self.bw = int(transmitParams[2])
        self.period = float(transmitParams[9])
        self.pTXmax = max(powSet) # max pTX
        self.sensi = sensi

        print(f"[myNode __init__] bw={self.bw}, period={self.period}, pTXmax={self.pTXmax}")

        # generate proximateBS
        self.proximateBS = self.generateProximateBS(bsList, interferenceThreshold, logDistParams)
        # print(f"[myNode __init__] Generated proximateBS: {self.proximateBS}")

        # set of actions
        self.freqSet = freqSet

        if self.info_mode == "NO":
            self.sfSet = sfSet
            self.powerSet = powSet
        else:
            self.sfSet, self.powerSet = self.generateHoppingSfFromDistance(sfSet, powSet, logDistParams)
        print(f"[myNode __init__] sfSet: {self.sfSet}")

        self.setActions = [(self.sfSet[i], self.freqSet[j], self.powerSet[k]) for i in range(len(self.sfSet)) for j in range(len(self.freqSet)) for k in range(len(self.powerSet))]
        self.nrActions = len(self.setActions)
        self.initial = initial
        print(f"[myNode __init__] setActions: {self.setActions}")
        print(f"[myNode __init__] nrActions: {self.nrActions}")
        
        # Storing the last 'n' states of a node 
        self.rssiHistory = []  # Store RSSI history for the node
        # self.snrHistory = []  # Store SNR history for the node
        self.packetsSuccessfulHistory = []  # Store successful packets history
        self.packetsTransmittedHistory = []  # Store transmitted packets history
        self.previousState = [1, 0, 5, 1.0, 0]  # Initial state: [Normalized RSSI, PRR, airtime, battery level]

        # initialize LoRaDRL agent
        self.algo = algo
        if self.algo == 'DDQN_LORADRL':
            self.loraDrlAgent = LoRaDRL(state_size=len(self.previousState), action_size=self.nrActions, sfSet=self.sfSet, powSet=self.powerSet, freqSet=self.freqSet)
        elif self.algo == 'DDQN_ARA':
            self.loraDrlAgent = AdaptiveResourceAllocation(state_size=len(self.previousState), action_size=self.nrActions, sfSet=self.sfSet, powSet=self.powerSet, freqSet=self.freqSet)
        self.targetUpdateInterval = 100  # Update target model every 100 successful packets

        # generate packet and ack
        # Unlike the parent class, we only initialize the packets attribute here and initialize in the sim function of utils.py
        self.packets = {}
        self.ack = {}

        # measurement params
        self.packetNumber = 0
        self.packetsTransmitted = 0
        self.packetsSuccessful = 0
        self.transmitTime = 0
        self.energy = 0
        self.energyConsumedByThisPacket = 0  # Energy consumed for the last packet in joules
        
    
    def generateHoppingSfFromDistance(self, sfSet, powSet, logDistParams):
        """ Generate the sf hopping sequence and power set from distance
        Parameters
        ----------
        logDistParams: list in format [gamma, Lpld0, d0]
            Parameters for log shadowing channel model.
        Returns
        -------
    
        """
        sfBuckets = []
        powBuckets = []
        gamma, Lpld0, d0 = logDistParams
        dist = self.proximateBS[0] if 0 in self.proximateBS else 0
        print(f"[myNode generateHoppingSfFromDistance] Distance for SF hopping: {dist}")

        if self.bw == 125:
            bwInd = 0
        else:
            bwInd = 1
        Lpl = self.pTXmax - self.sensi[:, bwInd+1]

        LplMatrix = Lpl.reshape((6,1))
        distMatrix =np.dot(d0, np.power(10, np.divide(LplMatrix - Lpld0, 10*gamma)))

        for i in range(6):
            if dist <= distMatrix[0, 0]:
                minSF = 7
                minPower = min(powSet)
            elif distMatrix[i, 0 ]<= dist < distMatrix[i+1, 0]:
                minSF = (i + 1) + 7
                minPower = powSet[i] if i < len(powSet) else powSet[-1]
        tempSF = [sf for sf in sfSet if sf >= minSF]
        tempPow = [pow for pow in powSet if pow >= minPower]
        sfBuckets.extend(tempSF)
        powBuckets.extend(tempPow)
        print(f"[myNode generateHoppingSfFromDistance] SF Buckets: {sfBuckets} and Power Buckets: {powBuckets}")
        return sfBuckets, powBuckets

    
    def generatePacketsToBS(self, transmitParams, logDistParams):
        """ Generate dictionary of base-stations in proximity.
        Parameters
        ----------
        transmitParams : list
            Transmission parameters.
        logDistParams: list
            Channel parameters
        Returns
        -------
        packets: packet
            packets at BS
        """
        packets = {} # empty dictionary to store packets originating at a node
        for bsid, dist in self.proximateBS.items():
            packets[bsid] = rlPacket(self.nodeid, bsid, dist, transmitParams, logDistParams, self.sensi, self.setActions, self.nrActions, self.sfSet, agent=self.loraDrlAgent)
        print(f"[rlNode generatePacketsToBS] Packets generated: {packets}")
        return packets
    
    def updateAgent(self):
        # Get the last state from the memory or use a default state
        if self.loraDrlAgent.memory.__len__() > 0:
            # Get the sate from the memory if available otherwise use the default state [1, 0, 5, 1.0]
            self.previousState = self.loraDrlAgent.memory[-1][3]           
        current_state = self.get_network_state()
        prr = current_state[1]  # PRR from the current state
        if self.algo == 'DDQN_LORADRL':
            reward = self.loraDrlAgent.calculate_reward(prr, self.packets[0].rectime/1000)  # PRR (in percentage) and airtime(converted to seconds)
        else:
            reward = self.loraDrlAgent.calculate_reward(prr, self.packets[0].rectime/1000, self.packets[0].isLost)
        print(f"[{self.__class__.__name__} updateAgent] Previous state: {self.previousState}, Current state: {current_state}, Reward: {reward}")

        if self.get_battery_level() <= 0:
            done = True  # Episode is done if battery is empty
        else:
            done = False  # Assuming the episode is not done yet

        self.loraDrlAgent.remember(self.previousState, self.packets[0].chosenAction, reward, current_state, done)
        self.loraDrlAgent.replay()  # Train the agent with the replay memory
        # Update target model when the number of successful packets reaches the target update interval
        if self.packetsSuccessful % self.targetUpdateInterval == 0: 
            print(f"[{self.__class__.__name__} updateAgent] Updated target model at packetsSuccessful={self.packetsSuccessful}")
            self.loraDrlAgent.update_target_model()

        # # Update target model when the number of total tranmitted packets reaches the target update interval
        # if self.packetsTransmitted % self.targetUpdateInterval == 0: 
        #     self.loraDrlAgent.update_target_model()
        #    print(f"[{self.__class__.__name__} updateAgent] Updated target model at packetsTransmitted={self.packetsTransmitted}")

    def get_network_state(self):
        """ Get the current state of the network.
        Returns
        -------
        state: np.array
            Current state of the network.
        """
        rssi = self.packets[0].pRX  # RSSI of the packet
        normalized_rssi = (rssi - (-137)) / (-70 - (-137))  # Normalize RSSI to [0, 1] range
        # snr = self.packets[0].snr  # SNR of the packet
        # prr = self.packetsSuccessful / self.packetsTransmitted if self.packetsTransmitted > 5 else 0  # Packet Reception Rate
        prr = sum(self.packetsSuccessfulHistory[-100:]) / sum(self.packetsTransmittedHistory[-100:]) if len(self.packetsTransmittedHistory) > 5 else 0  # PRR from the last 100 packets
        airtime = self.packets[0].rectime/1000  # Airtime of the packet in seconds
        self.battery_level = self.get_battery_level()
        state = [normalized_rssi, prr, airtime, self.battery_level, self.energyConsumedByThisPacket]  
        self.rssiHistory.append(rssi)  # Store the RSSI in the history
        # self.snrHistory.append(snr)  # Store the SNR in the history
        if len(self.rssiHistory) > 10:
            self.rssiHistory.pop(0)
            # self.snrHistory.pop(0)  # Keep only the last 10 SNR values
        return state

    def get_battery_level(self):
        """ Get the current battery level of the node.
        Returns
        -------
        battery_level: float
            Current battery level of the node.
        """
        # Assuming a AA battery with 2500 mAh suppying constantly at 3.7V 
        # The battery capacity in Joules is 2500 mAh * 3.7V = 9250 mWh = 33,300 J
        battery_capacity = 33300  # in Joules
        # print(f"rlNode get_battery_level] Current energy consumed: {self.energy} J")
        remaining_energy = battery_capacity - self.energy # Remaining energy in Joules
        return remaining_energy / battery_capacity  # Return the battery level as a fraction of the total capacity
    

class qlNode(rlNode):
    def __init__(self, nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, bsList, interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, simu_dir, fname):
        super().__init__(nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, bsList, interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, simu_dir, fname)
        self.loraDrlAgent = LoRaQLAgent(state_size=len(self.previousState), action_size=self.nrActions, sfSet=self.sfSet, powSet=self.powerSet, freqSet=freqSet)

    def updateAgent(self):
        # Get the last state from the memory or use a default state
        if self.loraDrlAgent.memory.__len__() > 0:
            # Get the sate from the memory if available otherwise use the default state [1, 0, 5, 1.0]
            self.previousState = self.loraDrlAgent.memory[-1][3]           
        current_state = self.get_network_state()
        prr = current_state[1]  # PRR from the current state
        reward = self.loraDrlAgent.calculate_reward(prr, self.packets[0].rectime/1000, self.packets[0].isLost)
        # print(f"[{self.__class__.__name__} updateAgent] Previous state: {self.previousState}, Current state: {current_state}, Reward: {reward}")

        if self.get_battery_level() <= 0:
            done = True  # Episode is done if battery is empty
        else:
            done = False  # Assuming the episode is not done yet

        self.loraDrlAgent.remember(self.previousState, self.packets[0].chosenAction, reward, current_state, done)
        self.loraDrlAgent.replay()  # Train the agent with the replay memory


class dqnNode(rlNode):
    def __init__(self, nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, bsList, interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, simu_dir, fname):
        super().__init__(nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, bsList, interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, simu_dir, fname)
        self.loraDrlAgent = DQNAgent(state_size=len(self.previousState), action_size=self.nrActions, sfSet=self.sfSet, powSet=self.powerSet, freqSet=self.freqSet)