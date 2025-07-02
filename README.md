# IOT-MAB: Decentralized Intelligent Resource Allocation Approach for LoRaWAN Networks

## Introduction
IoT-MAB is a discrete-event simulator based on SimPy for simulating intelligent distributed resource allocation in LoRa networks and to analyse scalability. We also combine the classed and functions for Physical layer of LoRA. 

## How to cite?
```latex
@misc{LoRa_MAB,
author =   {Duc-Tuyen Ta, Kinda Khawam, Samer Lahoud},
title =    {{LoRaWAN Network Simulator with Reinforcement Learning-based Algorithms}},
howpublished = {\url{https://github.com/tuyenta/IoT-MAB}},
}
```

## Installation
It is recommend to use virtualenv to keep your Python environment isolated, together with virtualenvwrapper to make working with virtual environments much more pleasant, e.g.:

```python
$ pip install virtualenvwrapper
...
$ export WORKON_HOME=~/.virtualenvs
$ mkdir -p $WORKON_HOME
$ source /usr/local/bin/virtualenvwrapper.sh
$ mkvirtualenv -p python3 iot_mab
```

You can install the required packages using the provided requirements.txt file:

```python
(lorasim)$ pip install -r requirements.txt
```

## Usage

### Synopsis

```python
python3 IoT_MAB.py <nrNodes> <nrIntNodes> <nrBS> <initial> <radius> <distribution> <AvgSendTime> <horizonTime>
<packetLength> <freqSet> <sfSet> <powerSet> <captureEffect> <interSFInterference> <infoMode> <logdir> <exp_name>
```

Example:

```python
python3 IoT_MAB.py --nrNodes 5 --nrIntNodes 5 --nrBS 1 --initial UNIFORM --radius 2000 --distribution '0.1 0.1 0.3 0.4 0.05 0.05' --AvgSendTime 360000 --horizonTime 10  --packetLength 50 --freqSet '867300' --sfSet '7 8'  --powerSet "14"  --captureEffect 1  --interSFInterference 1 --infoMode NO --algo 'exp3s' --logdir logs --exp_name exp1
```
### Description
**nrNodes**

Total number of end devices (nodes) to simulate in the network.

**nrIntNodes**

Number of intelligent (smart) nodes using learning-based algorithms. Must be less than or equal to `nrNodes`.

**nrBS**

Number of LoRaWAN gateways (base stations) in the simulation.

**initial**

Initialization method for the learning algorithm's action probabilities. Use `UNIFORM` for equal probability distribution or `RANDOM` for random initialization.

**radius**

Radius (in meters) of the simulated network area. All nodes are placed within this circular region.

**distribution**

Proportion of nodes assigned to each region within the network. Values should sum to 1.0. For example, `0.5 0.3 0.2` means 50% of nodes in region 1, 30% in region 2, and 20% in region 3.

**AvgSendTime**

Average interval (in milliseconds) between consecutive transmissions from each node.

**horizonTime**

Number of simulation cycles (iterations). The total simulated time is `horizonTime × AvgSendTime`.

**packetLength**

Size of each data packet (in bytes) sent by nodes.

**sfSet**

Set of Spreading Factors (SF) to be considered, specified as space-separated values between 7 and 12 (e.g., `7 8 9`).

**freqSet**

Set of carrier frequencies (in kHz) available for transmission, specified as space-separated values (e.g., `867300 868100`).

**powerSet**

Set of transmission power levels (in dBm) available to nodes, specified as space-separated values (e.g., `14 17`).

**captureEffect**

Enable (1) or disable (0) the capture effect, which allows a stronger signal to be received in the presence of interference.

**interSFInterference**

Enable (1) or disable (0) inter-spreading factor interference, which models imperfect orthogonality between different SFs.

**infoMode**

Level of information output during simulation. Options may include `NO`, `PARTIAL`, or `FULL` (refer to code for supported values).

**algo**

Specifies the learning algorithm used by intelligent nodes for resource allocation. Supported values include `exp3s`, `exp3`, and others as implemented in the code.

**logdir**

Directory name where simulation logs and output files will be stored.

**exp_name**

Name of the experiment/scenario, used to organize output files and results.

### Output

The result of every simulation run will be appended to a file named prob..._X.csv, ratio....csv, energy....csv and traffic....csv, whereby

* prob..._X is the probability of device X.

* ratio... is the packet reception ration of the network.

* energy... is the energy consumption of the network.

* traffic... is the normalized traffic and normalized throughput of the network.

The data file is then plotted into .png file by using matplotlib.

## Changelogs

## Contact
**Duc-Tuyen Ta**

Postdoc, ROCS, LRI, Paris-Sud University.
ta@lri.fr

**Kinda Khawam**

Associate Professor at the University of Versailles.
Associated to the ROCS team in LRI, Paris-Sud University.
kinda.khawam@gmail.com

**Samer Lahoud**

Faculté d’ingénierie ESIB, Université Saint-Joseph de Beyrouth, Lebanon
samer.lahoud@usj.edu.lb
