from lora.parse import get_args
from lora.utils import print_params, sim
from types import SimpleNamespace

def main(args):
    # import agruments
    nrNodes = int(args.nrNodes)
    nrIntNodes = int(args.nrIntNodes)
    nrBS = int(args.nrBS)
    initial = str(args.initial)
    radius = int(args.radius)
    distribution = list(map(float, args.distribution.split()))
    avgSendTime = int(args.AvgSendTime)
    horTime = int(args.horizonTime)
    packetLength = int(args.packetLength)
    sfSet = list(map(int, args.sfSet.split()))
    freqSet = list(map(int, args.freqSet.split()))
    powSet = list(map(int, args.powerSet.split()))
    captureEffect = bool(args.captureEffect)
    interSFInterference = bool(args.interSFInterference)
    info_mode = str(args.infoMode)
    algo = str(args.Algo)
    exp_name = str(args.exp_name)
    logdir = str(args.logdir)
    
    # print simulation parameters
    print("\n=================================================")
    print_params(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, horTime, packetLength, 
                sfSet, freqSet, powSet, captureEffect, interSFInterference, info_mode, algo)
    
    assert initial in ["UNIFORM", "RANDOM"], "Initial mode must be UNIFORM, RANDOM."
    assert info_mode in ["NO", "PARTIAL", "FULL"], "Initial mode must be NO, PARTIAL, or FULL."
    assert algo in ["exp3", "exp3s"], "Learning algorithm must be exp3 or exp3s."
    
    
    # running simulation
    bsDict, nodeDict = sim(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, horTime,
    packetLength, sfSet, freqSet, powSet, captureEffect, interSFInterference, info_mode, algo, logdir, exp_name)

    return bsDict, nodeDict

if __name__ == '__main__':
    # # import agruments
    # args = get_args()
    # # print args and run simulation

    args = SimpleNamespace(
        nrNodes=100,
        nrIntNodes=50,
        nrBS=1,
        initial='UNIFORM',
        radius=4500,
        distribution='0.1 0.1 0.3 0.4 0.05 0.05',
        AvgSendTime= 1*60*10e3,  # 1 minutes in ms
        horizonTime= 140000,
        packetLength=50,
        sfSet='7 8 9 10 11 12',  # Example spreading factors
        # sfSet='7 9 12',
        freqSet='868100',  # Example frequencies in kHz
        # freqSet='867300',
        powerSet='2 4 6 8 10 12 14',  # Example power levels in dBm
        # powerSet='2 6 10 14',
        captureEffect=1,
        interSFInterference=1,
        infoMode='FULL',  # NO, PARTIAL, FULL
        Algo='exp3s',
        exp_name='sc2',
        logdir='./exp3s_logs' 
    )
    main(args)