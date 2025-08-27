""" LPWAN Simulator: Hepper functions
============================================
Utilities (:mod:`lora.bsFunctions`)
============================================
.. autosummary::
   :toctree: generated/
   transmitPacket           -- Transmission process with discret event simulation.
   cuckooClock              -- Notify the simulation time (for each 1k hours).
   saveProb                 -- Save the probability profile of each node.
"""    
import os
import random
import numpy as np
from os.path import join
from .loratools import airtime, dBmTomW

def nsAdrAlgorithm(sf, txpow, m_snr, tpMax, tpMin):
    """
    Implements the ADR algorithm for a given node.
    Adjusts the transmission power (TxPower) and data rate (DR) based on the flowchart.
    """
    print("NS ADR algorithm in use.")
    min_sf = 7  
    snr_threshold = [-7.5, -10.0, -12.5, -15.0, -17.5, -20.0]
    req_snr = snr_threshold[sf - 7]  # SNR threshold for the current SF
    margin_snr = m_snr - req_snr - 10  # Margin to be added to the SNR threshold
    nstep = margin_snr / 3

    while nstep > 0 and sf > min_sf:
        sf -= 1
        nstep -= 1

    while nstep > 0 and txpow > tpMin:
        txpow -= 3
        nstep -= 1

    while nstep < 0 and txpow < tpMax:
        txpow += 3
        nstep += 1

    if txpow > tpMax:
        txpow = tpMax
    if txpow < tpMin:
        txpow = tpMin
        
    return sf, txpow

# Transmit
def transmitPacket(env, node, bsDict, logDistParams, algo, ergMonitor=None, prrMonitor=None):
    """ Transmit a packet from node to all BSs in the list.
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    node: my Node
        LoRa node.
    bsDict: dict
        list of BSs.
    logDistParams: list
        channel params
    algo: string
        learning algorithm
    Returns
    -------
    """
    while True:
        # The inter-packet waiting time. Assumed to be exponential here.
        yield env.timeout(random.expovariate(1/float(node.period)))
        
        # update settings if any
        node.updateTXSettings()
        node.resetACK()
        node.packetNumber += 1
        
        # send a virtual packet to each base-station in range and those we may affect
        for bsid, dist in node.proximateBS.items():
            if algo=="exp3" or algo=="exp3s": 
                prob_temp = [node.prob[x] for x in node.prob]
                node.packets[bsid].updateTXSettings(bsDict, logDistParams, prob_temp)
            elif algo == "basicAdr":
                if node.adrAckReq == 1:
                ################# Calling the NS ADR algorithm ##################
                    nsAdrAlgorithm(node.oldSF, node.oldTxp, node.m_snr, max(node.powerSet), min(node.powerSet))
                node.packets[bsid].updateTXSettings(bsDict, logDistParams, node.adrAckCnt, node.oldSF, node.oldTxp)
                node.adrAckReq = node.packets[bsid].adrAckReq
                print('oldSF:', node.oldSF, 'oldTxp:', node.oldTxp, 'newSF:', node.packets[bsid].sf, 'newTxp:', node.packets[bsid].pTX)
                #################################################################
            else:
                # in case of DQN, the choice of action depends on the (previous) states 
                # So this is where we pass the preceding state information to the agent via the packet object
                node.packets[bsid].updateTXSettings(bsDict, logDistParams, node.previousState)
            bsDict[bsid].addPacket(node.nodeid, node.packets[bsid])
            bsDict[bsid].resetACK()
        
        print("[bsFunctions transmitPacket] Start transmitting packet at t= {}".format(int(1+env.now/(node.period))) + " from node {}".format(node.nodeid))
        # print(node.prob)
        # print(node.weight)
        for pkid in bsDict[bsid].packets.keys():    
            print(
                f"packetId: {pkid}, sf: {bsDict[bsid].packets[pkid].sf}, "
                f"freq: {bsDict[bsid].packets[pkid].freq}, "
                f"pTX: {bsDict[bsid].packets[pkid].pTX}"
            )
        
        # wait until critical section starts
        Tcritical = (2**node.packets[0].sf/node.packets[0].bw)*(node.packets[0].preambleLength - 5) # time until the start of the critical section
        yield env.timeout(Tcritical)
        
        # make the packet critical on all nearby basestations
        for bsid in node.proximateBS.keys():
            bsDict[bsid].makeCritical(node.nodeid)
            
        Trest = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw, node.packets[0].packetLength, node.packets[0].preambleLength, node.packets[0].syncLength, node.packets[0].headerEnable, node.packets[0].crc)) - Tcritical # time until the rest of the message completes
        yield env.timeout(Trest)
        
        successfulRx = False
        ACKrest = 0
        
        # transmit ACK
        for bsid in node.proximateBS.keys():
            # print("[bsFunctions transmitPacket]=====> eval bs {}".format(bsid))
            if bsDict[bsid].removePacket(node.nodeid):
                bsDict[bsid].addACK(node.nodeid, node.packets[bsid])
                ACKrest = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw, node.packets[0].packetLength, node.packets[0].preambleLength, node.packets[0].syncLength, node.packets[0].headerEnable, node.packets[0].crc))# time until the ACK completes
                yield env.timeout(ACKrest)
                node.addACK(bsDict[bsid].bsid, node.packets[bsid])
                successfulRx = True
                
        # update parameters        
        node.packetsTransmitted += 1
        node.energyConsumedByThisPacket = node.packets[0].getPktAirtime() * dBmTomW(node.packets[0].pTX) * (3.0) /1e6 # V = 3.0    
        node.energy += node.energyConsumedByThisPacket
        # updating the overall average energy per packet by finding the mean based on the energy consumed by the incoming packet  
        ergMonitor.avgErgPerPkt = (ergMonitor.avgErgPerPkt + node.energyConsumedByThisPacket) / 2  
        prrMonitor.sysWidePktTx += 1

        if successfulRx:
            if node.info_mode in ["NO", "PARTIAL"]:
                node.packetsSuccessful += 1
                node.transmitTime += node.packets[0].getPktAirtime()
                prrMonitor.sysWideSuccessfulPkt += 1
                prrMonitor.prrSys = prrMonitor.sysWideSuccessfulPkt / prrMonitor.sysWidePktTx
            elif node.info_mode == "FULL": 
                if not node.ack[0].isCollision:   # ack is stored in the form of the whole packet object in the dict. So here we check the isCollision attribute 
                    node.packetsSuccessful += 1
                    node.transmitTime += node.packets[0].getPktAirtime()
                    prrMonitor.sysWideSuccessfulPkt += 1
                    prrMonitor.prrSys = prrMonitor.sysWideSuccessfulPkt / prrMonitor.sysWidePktTx
            if algo=='exp3' or algo=='exp3s':
                node.updateProb(algo)
            elif algo=="basicAdr":
                node.adrAckCnt = 0
                node.adrAckReq = 0
                # Like we did update agent in the RL algos or updated probability for the exp3 algo, we update the node in case of the basic ADR algo
                # This will update the SNR history and also update the SF and TxPower value of the node   
                node.updateNode()
        else:
            if algo=='basicAdr':
                node.adrAckCnt += 1

        if algo not in ('exp3', 'exp3s', 'basicAdr'):
            node.packetsTransmittedHistory.append(1)
            node.packetsSuccessfulHistory.append(1 if successfulRx else 0)
            if algo=="DDQN_sysOptim":
                node.updateAgent(prrMonitor.prrSys, ergMonitor.avgErgPerPkt)
            elif algo=="masterAgent":
                node.updateAgent(prrMonitor.prrSys, ergMonitor.avgErgPerPkt, prrMonitor.sysWideSuccessfulPkt)
            else:
                node.updateAgent()
            
        # print("[bsFunctions transmitPacket]Probability of action from node " +str(node.nodeid)+ " at (t+1)= {}".format(int(1+env.now/(6*60*1000))))
        # print(node.prob)
        # print(node.weight)
        # wait to next period
        yield env.timeout(float(node.period)-Tcritical-Trest-ACKrest)
        #input()

def cuckooClock(env):
    """ Notifies the simulation time.
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    Returns
    -------
    """
    while True:
        yield env.timeout(1000 * 3600000)
        print("Running {} kHrs".format(env.now/(1000 * 3600000)))

def saveAvgEnergyPerPacket(env, ergMonitor, fname, simu_dir):
    """ Save average energy per packet to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    ergMonitor: ergMonitor
        Energy monitor object.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 3600000)
        # write average energy per packet to file
        filename = join(simu_dir, str('avgEnergyPerPacket_'+ fname) + '.csv')
        if os.path.isfile(filename):
            res = "\n" + str(ergMonitor.avgErgPerPkt)
        else:
            res = str(ergMonitor.avgErgPerPkt)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()

def saveProb(env, nodeDict, fname, simu_dir):
    """ Save probabilities every to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 3600000)
        # write prob to file
        for nodeid in nodeDict.keys():
             if nodeDict[nodeid].node_mode != "UNIFORM":
                filename = join(simu_dir, str('prob_'+ fname) + '_id_' + str(nodeid) + '.csv')
                save = str(list(nodeDict[nodeid].prob.values()))[1:-1]
                if os.path.isfile(filename):
                    res = "\n" + save
                else:
                    res = save
                with open(filename, "a") as myfile:
                    myfile.write(res)
                myfile.close()

def saveTxParams(env, nodeDict, fname, simu_dir):
    """ Save transmission parameters to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    txParamDir = join(simu_dir, "txParams")
    if not os.path.exists(txParamDir):
        os.makedirs(txParamDir)
    while True:
        yield env.timeout(100 * 3600000)
        # write tx params to file
        for nodeid in nodeDict.keys():
            filename = join(txParamDir, str('txParams_'+ fname) + '_id_' + str(nodeid) + '.csv')
            save = str(nodeid) + " " + str(nodeDict[nodeid].packets[0].dist) + " " + str(nodeDict[nodeid].packets[0].sf) + " " + str(nodeDict[nodeid].packets[0].freq) + " " + str(nodeDict[nodeid].packets[0].pTX)
            if os.path.isfile(filename):
                res = "\n" + save
            else:
                res = save
            with open(filename, "a") as myfile:
                myfile.write(res)
            myfile.close()

def savePRRlastFew(env, nodeDict, fname, simu_dir):
    """ Save packet reception ratio of last 100 or 1000 packets 
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(24 * 3600000)
        # write packet reception ratio to file
        PacketReceptionRatioLastFew = 0
        nTransmitted = sum(
            sum(nodeDict[nodeid].packetsTransmittedHistory[-100:]) 
            for nodeid in nodeDict.keys()
            )
        nRecvd = sum(
            sum(nodeDict[nodeid].packetsSuccessfulHistory[-100:]) 
            for nodeid in nodeDict.keys()
            )
        if nTransmitted > 0:
            PacketReceptionRatioLastFew = nRecvd/nTransmitted
        filename = join(simu_dir, str('PRR_last_few'+ fname) + '.csv')
        if os.path.isfile(filename):
            res = "\n" + str(PacketReceptionRatioLastFew)
        else:
            res = str(PacketReceptionRatioLastFew)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()

def saveRatio(env, nodeDict, fname, simu_dir):
    """ Save packet reception ratio to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 3600000)
        # write packet reception ratio to file
        nTransmitted = 0
        nRecvd = 0
        PacketReceptionRatio = 0
        nTransmitted = sum(nodeDict[nodeid].packetsTransmitted for nodeid in nodeDict.keys())
        nRecvd = sum(nodeDict[nodeid].packetsSuccessful for nodeid in nodeDict.keys())
        PacketReceptionRatio = nRecvd/nTransmitted
        filename = join(simu_dir, str('ratio_'+ fname) + '.csv')
        if os.path.isfile(filename):
            res = "\n" + str(PacketReceptionRatio)
        else:
            res = str(PacketReceptionRatio)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()

def saveEnergy(env, nodeDict, fname, simu_dir):
    """ Save energy to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 3600000)
        # compute and wirte energy consumption to file
        totalEnergy = sum(nodeDict[nodeid].energy for nodeid in nodeDict.keys())
        nTransmitted = sum(nodeDict[nodeid].packetsTransmitted for nodeid in nodeDict.keys())
        nRecvd = sum(nodeDict[nodeid].packetsSuccessful for nodeid in nodeDict.keys())
        filename = join(simu_dir, str('energy_'+ fname) + '.csv')
        if os.path.isfile(filename):
            res = "\n" + str(totalEnergy) + " " + str(nTransmitted) + " " + str(nRecvd)
        else:
            res = str(totalEnergy) + " " + str(nTransmitted) + " " + str(nRecvd)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()

def saveTraffic(env, nodeDict, fname, simu_dir, sfSet, freqSet, lambda_i, lambda_e):
    """ Save norm traffic and throughput to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    sfSet: list
        set of possible sf
    freqSet: list
        set of possible freq
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 3600000)
        # compute and wirte traffic and throughtput to file
        # total_Ts = sum(nodeDict[nodeid].transmitTime for nodeid in nodeDict.keys())
        Gsc = np.zeros((len(sfSet),len(freqSet)))
        Tsc = np.zeros((len(sfSet),len(freqSet)))
        Gsc += lambda_e

        for nodeid in nodeDict.keys():
            if nodeDict[nodeid].packets[0].sf != None:
                if nodeDict[nodeid].packets[0].freq != None:
                    si = sfSet.index(nodeDict[nodeid].packets[0].sf) 
                    ci = freqSet.index((nodeDict[nodeid].packets[0].freq))
                    Gsc[si, ci] += lambda_i
        
        for i in range(len(sfSet)):
            Gsc[i, :] *= airtime((sfSet[i], nodeDict[0].packets[0].rdd, nodeDict[0].packets[0].bw, nodeDict[0].packets[0].packetLength, nodeDict[0].packets[0].preambleLength, nodeDict[0].packets[0].syncLength, nodeDict[0].packets[0].headerEnable, nodeDict[0].packets[0].crc))

        for i in range(len(sfSet)):
            for j in range(len(freqSet)):
                Tsc[i][j] = Gsc[i][j] * np.exp(-2* Gsc[i][j]) 

        filename = join(simu_dir, str('traffic_'+ fname) + '.csv')
        if os.path.isfile(filename):
            res = "\n" + str(sum(sum(Gsc))) + " " + str(sum(sum(Tsc)))
        else:
            res = str(sum(sum(Gsc))) + " " + str(sum(sum(Tsc)))
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()