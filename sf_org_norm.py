
# In[1]:
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
import os
from numba import njit
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from loguru import logger
from datetime import datetime

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

analysisdir = os.getcwd()
basedir = os.path.dirname(os.getcwd())+'/'
os.chdir(basedir)

for arg in sys.argv:
    run_num=str(arg).zfill(5)
    # print(run_num)

run_start=str(sys.argv[1]).zfill(5)
run_end=str(sys.argv[2]).zfill(5)
run_num=str(sys.argv[3]).zfill(5)

chan_enab = [25,26,27,28,29]

datadir = 'D:/LANSCE_FP12_2023/data/' ## add directory of hard drive
uniquefolder = "runs" + str(run_start) + "-" + str(run_end) +"/"
savefilename = 'SF_Norm_files/'+uniquefolder+run_num
if not os.path.exists(os.getcwd() + '/' +'SF_Norm_files/'+uniquefolder):
    # Create the directory
    os.makedirs(os.getcwd() + '/' +'SF_Norm_files/'+uniquefolder)
    print("Directory created successfully!")
else:
    pass

logger.add(basedir+"/SF_Norm_files/" + uniquefolder + '0_ErrorLog_'+run_start+'_'+run_end+'.txt', delay = True)

print('processing data: ' + uniquefolder + 'run' + run_num)

statefileloc = basedir+'\SF_Norm_files\TR_R_expected_avgs_stds_afterclip.csv'

start = time.time()
fullstart = time.time()

read_data = []
fileLength = []

####### try reading all channels. If no ch 29, try reading ch 25-28. If no 26-28 (coils) try just ch 25. If still an error, log it.
try: ## adding attempt to read La Puck info, Ch. 29
    for el in chan_enab:
        f = open(datadir+uniquefolder + 'run' + str(run_num) + "_ch" +str(el) + ".bin", 'rb')
        read_data.append(np.fromfile(file=f, dtype=np.uint16))
        f.close()
        fileLength.append(len(read_data[-1]))
except:
    print('No La Puck state Ch. 29')
    chan_enab = chan_enab[:-1]  ## get rid of ch 29 puck state if there is no ch 29
    try:
        for el in chan_enab:
            f = open(datadir+uniquefolder + 'run' + str(run_num) + "_ch" +str(el) + ".bin", 'rb')
            read_data.append(np.fromfile(file=f, dtype=np.uint16))
            f.close()
            fileLength.append(len(read_data[-1]))
    except: 
        print('Also no coils')
        chan_enab = [chan_enab[0]]  ## get rid of ch 29 puck state if there is no ch 29
        try:
            for el in chan_enab:
                f = open(datadir+uniquefolder + 'run' + str(run_num) + "_ch" +str(el) + ".bin", 'rb')
                read_data.append(np.fromfile(file=f, dtype=np.uint16))
                f.close()
                fileLength.append(len(read_data[-1]))
        except Exception as e:
            logger.error('run '+run_num + ' failed during data load. Unknown reason')
            logger.exception(e)

fileLength = np.asarray(fileLength)

print("Channel is " + str(chan_enab))
print('saving to state & norm information to ' + savefilename)
end = time.time()

# Store the big header for each channel in arrays
# In[2]:
BoardID = []
recordLength = []
numSamples = []
eventCounter = []
decFactor = []
chanDec = []
postTrig = []
groupStart = []
groupEnd = []
timestamp= []
sizeFirstEvent = []
TTT = []

targetDict = {0: "La", 1: "Tb2O3", 2: "Yb2O3", 3: "Sm2O3", 4: "Er2O3", 5: "Ho2O3", 6: "other"}
foilDict = {0: "TBD", 1: "TBD", 2: "TBD", 3: "TBD", 4: "TBD", 5: "TBD", 6: "other"}

target=(read_data[0][5]&0x00F0)>>4
foil=read_data[0][5]&0x000F
targetFlag = read_data[0][5]>>8&1
foilFlag = read_data[0][5]>>9&1
spinFiltFlag = read_data[0][5]>>10&1
spinFlipFlag = read_data[0][5]>>11&1
shutterFlag = read_data[0][5]>>12&1
facilityTrigFlag = read_data[0][5]>>13&1

if targetFlag:
    target=targetDict[(read_data[0][5]&0x00F0)>>4]
else:
    target = "empty"
    
    
if foilFlag:
    foil=foilDict[read_data[0][5]&0x000F]
else:
    foil = "empty"

for i in range(0,len(chan_enab)):
    BoardID.append(read_data[i][9]>>8)
    recordLength.append(((read_data[i][9]&0x00FF)<<16)+read_data[i][8])
    numSamples.append(((read_data[i][11]&0x00FF)<<16)+read_data[i][10])
    eventCounter.append(read_data[i][6]+(read_data[i][7]<<16))
    BoardID.append(read_data[i][9]>>8)  
    decFactor.append(read_data[i][11]>>8)
    chanDec.append(read_data[i][13]>>8)
    postTrig.append(read_data[i][15]>>8)
    groupStart.append(((read_data[i][13]&0x00FF)<<16)+read_data[i][12])
    groupEnd.append(((read_data[i][15]&0x00FF)<<16)+read_data[i][14])
    
    timestamp.append(read_data[i][16]+(read_data[i][17]<<16)+(read_data[i][18]<<32)+(read_data[i][19]<<40))  
    sizeFirstEvent.append(read_data[i][0]+(read_data[i][1]<<16))
    TTT.append(read_data[i][2]+(read_data[i][3]<<16)+(read_data[i][4]<<32))
    
#     print("For channel " + str(chan_enab[i]) + ", BoardID is " + str(BoardID[i])
#           + "; record length is " + str(recordLength[i]) + "; num Samples is " 
#           + str(numSamples[i]) + "; event counter is " + str(eventCounter[i]) + "; dec factor is " + str(decFactor[i]) + "; chan dec is " 
#           + str(chanDec[i]) + "; postTrig is " + str(postTrig[i]) + "; group start is " + str(groupStart[i]) + "; group end is " + str(groupEnd[i])
#           + "; epoch time is " + str(timestamp[i]) +  "; first event size is " + str(sizeFirstEvent[i]) + "; and ETTT is " + str(TTT[i]) + "\n")

numSamples = np.asarray(numSamples)

eventCounter = np.asarray(eventCounter)
TTT = np.asarray(TTT)

print("Target is " + target)
# print("Foil is " + foil)
# print("Shutter is open: " + str(bool(shutterFlag)))
# print("Facility t0 is on: " + str(bool(facilityTrigFlag)))
# print("Spin flipper is on: " + str(bool(spinFlipFlag)))
# print("Spin filter is on: " + str(bool(spinFiltFlag)))
# print("Target is present: " + str(bool(targetFlag)))
# print("Foil is present: " + str(bool(foilFlag)))


# Determine the time axis for each channel
# In[3]:
preTime = []
startTime = []
endTime = []
resolution = []
xs = [] 

for i in range(0,len(chan_enab)):
    preTime.append((100-postTrig[i])*recordLength[i]/100)
    startTime.append((-1*preTime[i]*16*decFactor[i] + groupStart[i]*16*decFactor[i]))
    endTime.append((-1*preTime[i]*16*decFactor[i] + groupEnd[i]*16*decFactor[i]))
    resolution.append(16*chanDec[i]*decFactor[i])
#     print("Pretime for channel", chan_enab[i],"is " + str(preTime[i]) + "; start time is " + str(startTime[i]) + "; end time is " + str(endTime[i]) 
#           + "; resolution is " + str(resolution[i]) + "ns")
    xs.append(np.arange(startTime[i],(numSamples[i])*resolution[i]+startTime[i], resolution[i]))


# In[4]:
start=time.time()
if chan_enab[0] != 25:
    emessage = ('3He not in first channel loaded. Cannot normalize')
    logger.error(run_num + emessage)
    raise Exception(emessage)
else:
    pass

@njit
def dataread(data, channels, fileLen, numSamps):
    numRuns = int((fileLen[0]-20-numSamps[0])/(numSamps[0]+6)+1)
    ys_arr = np.zeros((len(channels), numRuns,numSamps[0]), dtype=np.uint16)
    ETTT_arr = np.zeros((len(channels), numRuns), dtype=np.intc)
    eventcount_arr = np.zeros((len(channels), numRuns), dtype=np.intc)
    for i in range(0,len(channels)):
        eventCount = 0
        byteCounter = 0
        while byteCounter < fileLen[i]:
            if byteCounter == 0:
                ETTT_arr[i]=TTT[i]
                #ETTT_arr[i].append(TTT[i])
                eventcount_arr[i]=(eventCounter[i])
                byteCounter = 20
            else:
                ETTT_arr[i]=(data[i][byteCounter]+(data[i][byteCounter+1]<<16)+(data[i][byteCounter+2]<<32))
                eventcount_arr[i]=(data[i][byteCounter+4]+(data[i][byteCounter+5]<<16))
                byteCounter += 6
            for j in range(0, numSamps[i]):
                ys_arr[i][eventCount][j]=data[i][byteCounter]
                byteCounter += 1
            eventCount += 1
    return ys_arr, ETTT_arr, eventcount_arr

if chan_enab[-1] == 29:
    ys_arrHe, ETTT_arrHe, eventcount_arrHe  = dataread(read_data, np.array([25]), fileLength, numSamples) ##hardcoded channel 25 for He
    ys_arr, ETTT_arr, eventcount_arr  = dataread(read_data[1:], np.array([26,27,28]), fileLength[1:], numSamples[1:]) ##hardcoded channels for coils
    ys_arrpuck, ETTT_arrpuck, eventcount_arrpuck  = dataread(read_data[4:], np.array([29]), fileLength[4:], numSamples[4:]) ##hardcoded channel for La Puck
    ETTT_arr = np.vstack([ETTT_arrHe,ETTT_arr,ETTT_arrpuck]) ## ordering makes sure that first array of new ETTT_arr is ETTT_arr of He
    eventcount_arr = np.vstack([eventcount_arrHe,eventcount_arr,eventcount_arrpuck])
elif chan_enab[-1] == 25: ## if no coils and no La Puck
    ys_arrHe, ETTT_arr, eventcount_arr  = dataread(read_data, np.array([25]), fileLength, numSamples) ##hardcoded channel 25 for He
else:
    ys_arrHe, ETTT_arrHe, eventcount_arrHe  = dataread(read_data, np.array([25]), fileLength, numSamples) ##hardcoded channel 25 for He
    ys_arr, ETTT_arr, eventcount_arr  = dataread(read_data[1:], np.array([26,27,28]), fileLength[1:], numSamples[1:]) ##hardcoded channels for coils
    ETTT_arr = np.vstack([ETTT_arrHe,ETTT_arr]) ## ordering makes sure that first array of new ETTT_arr is ETTT_arr of He
    eventcount_arr = np.vstack([eventcount_arrHe,eventcount_arr])

end = time.time()
print('dataread from binary time: ' + str(end-start))  

# Put ADC values in arrays for each channel (one array per event, an array of events per channel) and put the miniheader information in an array


# In[5]:

timeDif=[]
for i in range(0,len(chan_enab)):
    timeDif.append([])
    for j in range(len(ETTT_arr[i])-1):
        timeDif[i].append((ETTT_arr[i][j+1]-ETTT_arr[i][j])*8)
#     print("Min time difference for channel", chan_enab[i], "is", min(timeDif[i]), "ns")
#     print("Max time difference for channel", chan_enab[i], "is", max(timeDif[i]), "ns \n")
#print(timeDif)

# In[6]:
baseL = 0
baseRHe = int(((preTime[0]-groupStart[0])*0.70)/chanDec[0])
numRuns = int((fileLength[0]-20-numSamples[0])/(numSamples[0]+6)+1)

if spinFlipFlag == 0:
    pass
if spinFlipFlag == 1:
    baseRCoil = int(((preTime[1]-groupStart[1])*0.70)/chanDec[1])

if chan_enab[-1] != 25:
    if numSamples[0] != 45000:
        emessage = ('He channel wrong size')
        logger.error(run_num + ' ' + emessage)
        raise Exception(emessage)
    elif numSamples[1] != 351:
        emessage = ('Coil channel wrong size')
        logger.error(run_num + ' ' + emessage)
        raise Exception(emessage)
    elif numSamples[2] != 351:
        emessage = ('Coil channel wrong size')
        logger.error(run_num + ' ' + emessage)
        raise Exception(emessage)
else:
    if numSamples[0] != 45000:
        emessage = ('He channel wrong size')
        logger.error(run_num + ' ' + emessage)
        raise Exception(emessage)
    
legend =  ['LO', 'TR', 'R']
transitions = ['111->101', '101->100', '100->110', '110->101','101->110','110->111','111->100','100->111']
switchpulses = np.arange(255,614,45) 
p=0

s = switchpulses[p]
t=s+1

start = time.time()

def plotter(ys, xs, baseR, numpoints):
    tempys_basesub = np.zeros((len(ys), numRuns,numpoints[0]), dtype=float)
    for i in range((len(ys))):
        for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
            tempys_basesub[i][pulse]=np.subtract(ys[i][pulse], np.mean(ys[i][pulse][baseL:baseR]))
        for j in range(s, t): ## plot only interested pulses
            plt.plot(xs[i], tempys_basesub[i][j] , label=legend[i]) #+str(sums[1][j])) ## sums[j] will not work for more than just TR   
            plt.axvline(xs[0][baseL], ls = '--')
            plt.axvline(xs[0][baseR], ls = '--')
            plt.axvline(xs[0][baseR+5], ls = '--', c ='r') ## BaseR+5 line marks the beginning of the integral, until the end of samples.
            plt.legend()
            
#plotter(ys_arrHe, xs, baseRHe,numSamples) ##plot 3He
#plotter(ys_arr, xs[1:], baseRCoil, numSamples[1:]) ##plot coils

#@njit(nopython = True) ## Actually JIT seems to be slower here!
def basesub_sum(ys, baseR, numpoints): ## for coils, could be used for He but below does that
    tempys_basesub = np.zeros((len(ys), numRuns,numpoints[0]), dtype=np.float64)
    tempsums = np.zeros((len(ys), numRuns), dtype=np.float64)
    for i in range((len(ys))):
        for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
            tempys_basesub[i][pulse]=np.subtract(ys[i][pulse], np.mean(ys[i][pulse][baseL:baseR]))
            tempsums[i][pulse] = np.sum(tempys_basesub[i][pulse][baseR+5:-1])
    return tempys_basesub, tempsums

if spinFlipFlag == 0:
    print('Spin Flipper is off. No state info.')
    ys_basesub = np.zeros((3,numRuns,351), dtype=np.float64) ##3 channels for what would have been ys_arr
    sums = np.zeros((3,numRuns), dtype=np.float64)
if spinFlipFlag == 1:
    ys_basesub, sums = basesub_sum(ys_arr, baseRCoil, numSamples[1:])

@njit ## separate function for He because this checks every point, can take a while
def basesub_normHe(ys, baseRegion, intgrRegion):
    tempys_basesub = np.zeros((len(ys), numRuns,45000), dtype=np.float64) #hardcode numSamples[0] = 45000
    tempsums = np.zeros((len(ys), numRuns), dtype=np.float64)
    for i in range((len(ys))): ## i is pretty much always 0 for 3He. Left general.
        for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
            for j in range(intgrRegion[0]+1000, intgrRegion[1]): ## checking for saturation in He channel. restrict to slightly smaller range
                if ys[i][pulse][j] > 4060:  ## cutoff adc value (real is 4092)
                    err = ((('3He is saturating in normalization region at pulse,point: ' + str(pulse) + ', '+ str(j))))
                    print(err)
                else:
                    err = ''
                    pass
            tempys_basesub[i][pulse]=np.subtract(ys[i][pulse], np.mean(ys[i][pulse][baseRegion[0]:baseRegion[1]]))
            tempsums[i][pulse] = np.sum(tempys_basesub[i][pulse][intgrRegion[0]:intgrRegion[1]])
    return tempys_basesub, tempsums, err

baseRHe = int(((preTime[0]-groupStart[0])*0.70)/chanDec[0]) #redefined for clarity

HeBaseReg = np.array([0, baseRHe])
HeIntgrReg = np.array([baseRHe+700, 15999]) ## hardcoded begin/end region for integral over NaI and 6Li regions

if shutterFlag == 0:
    print('Shutter closed. HeNorms set to 1')
    HeNorms = np.ones((1,numRuns), dtype=np.float64)
    emessage = '' 
if shutterFlag == 1:
    ys_basesubHe, HeNorms, emessage = basesub_normHe(ys_arrHe, HeBaseReg, HeIntgrReg)

## can't use logger in JIT, so do it outside
if len(emessage) >1 :
    logger.error(emessage)
    raise Exception(emessage)
else:
    pass

end = time.time()
# print('plotting and/or base subtraction time: ' + str(end-start))            

# In[7]:

start = time.time()

statefile = pd.read_csv(statefileloc)
transitions = statefile['transition'].to_numpy() 
expectedSumsTR_R = statefile[' averages'].to_numpy()
expectedStdsTR_R = statefile[' standard dev'].to_numpy()

AllSwitches = []
tolerance = 6000 ## see comments below

if spinFlipFlag == 0:
    pass
if spinFlipFlag == 1:
    ## can't use pre-existing np array because usually one array of unequal length
    for i in range(len(expectedSumsTR_R)):
        diff_arr = np.absolute(np.add(sums[1],sums[2]) - expectedSumsTR_R[i])
        found_sums =[]
        for j in range(len(diff_arr)):
            if diff_arr[j] < tolerance: ## this uses a constant "tolerance"
                found_sums.append(j)
        AllSwitches.append(np.array(found_sums))

end = time.time()
# print('find switches time: ' + str(end-start)) 

# In[8]:
if spinFlipFlag == 0:
    #### i.e. no spin flip saving
    transitions = ['bogus']
    AllSwitches = ['bogus']
    transitionSumsTR = ['bogus']
    transitionSumsR = ['bogus']
    transitionTR_RAvgs = ['bogus']
    transitionTR_Rstds = ['bogus']
    transitionSumsTR_R = ['bogus']
    cols = ['transition', 'transition_locations', 'sumsTR_R', 'TR_R_avgs', 'TR_R_stds']
    transSumsData = [transitions, AllSwitches, transitionSumsTR_R, transitionTR_RAvgs, transitionTR_Rstds]

    df_SF = pd.DataFrame({cols[0]: transSumsData[0],            
                    cols[1]: transSumsData[1],
                    cols[2]: transSumsData[2],
                    cols[3]: transSumsData[3],
                    cols[4]: transSumsData[4]})
if spinFlipFlag == 1:
    start = time.time()
    transitions = ['111->101', '101->100', '100->110', '110->101','101->110','110->111','111->100','100->111']
    transitionSumsTR = []
    transitionSumsR = []
    transitionTR_RAvgs = []
    transitionTR_Rstds = []
    transitionSumsTR_R = []
    for i in range(0,len(transitions)):
        tempTR = []
        tempR = []
        for j in range(0,len(AllSwitches[i])):
            tempTR.append(sums[1][AllSwitches[i][j]])
            tempR.append(sums[2][AllSwitches[i][j]])
        transitionSumsTR.append(tempTR)
        transitionSumsR.append(tempR)
        transitionSumsTR_R.append(np.add(tempTR,tempR))
        transitionTR_RAvgs.append(np.average(np.add(tempTR,tempR)))
        transitionTR_Rstds.append(np.std(np.add(tempTR,tempR)))

    cols = ['transition', 'transition_locations', 'sumsTR_R', 'TR_R_avgs', 'TR_R_stds']
    transSumsData = [transitions, AllSwitches, transitionSumsTR_R, transitionTR_RAvgs, transitionTR_Rstds]

    df_SF = pd.DataFrame({cols[0]: transSumsData[0],            
                    cols[1]: transSumsData[1],
                    cols[2]: transSumsData[2],
                    cols[3]: transSumsData[3],
                    cols[4]: transSumsData[4]})

    df_SF = df_SF.explode(['transition_locations', 'sumsTR_R']).reset_index().rename(columns={'index' : 'nicknames'}) #turn the 'index' of the exploded df_SF into a column, then reassign indices

    ## now sort by the transition location and rearrange dataframe indices, only matters for looping (??)
    df_SF = df_SF.sort_values(by=['transition_locations'])
    df_SF = df_SF.reset_index(drop=True)

    for ind in df_SF.index[:-1]:
    #     print('transition: '+ str(df_SF['nicknames'][ind]) + ' location: ' + str(df_SF['transition_locations'][ind]))
        if (df_SF['nicknames'][ind+1])-1 != df_SF['nicknames'][ind]: ## if next transition 'nickname' is not next in sequence, failure
            if (df_SF['nicknames'][ind+1])-1 == -1: ## special condition for end of sequence where (0-1) != 7
                if (df_SF['transition_locations'][ind+1]-df_SF['transition_locations'][ind]) != 45: ## changed to 45 pulses!
                    emessage = '# pulses error: ' + str(df_SF['transition_locations'][ind+1]-df_SF['transition_locations'][ind])
                    logger.error(run_num + ' ' + emessage)
                    raise Exception(emessage)
                if (df_SF['transition_locations'][ind+1]-df_SF['transition_locations'][ind]) == 45:
                    pass
                    # print('# pulses correct, end of sequence')
            else:
                ## checks that the sequence follows 0-> 1-> 2-> 3... etc order
                emessage = 'sorting failure, ' + str((df_SF['nicknames'][ind+1])-1) + '!=' + str(df_SF['nicknames'][ind])
                logger.error(run_num + ' ' + emessage)
                raise Exception(emessage)
        elif (df_SF['nicknames'][ind+1])-1 == df_SF['nicknames'][ind]:
            if (df_SF['transition_locations'][ind+1]-df_SF['transition_locations'][ind]) != 45: ## error if =/= 45 pulses between each
                emessage = ('# pulses error: ' + str(df_SF['transition_locations'][ind+1]-df_SF['transition_locations'][ind]))
                logger.error(run_num + ' ' + emessage)
                raise Exception(emessage)
            if (df_SF['transition_locations'][ind+1]-df_SF['transition_locations'][ind]) == 45:
                pass
        else:
            emessage = ('Unknown failure in sorting')
            logger.error(run_num + ' ' + emessage)
            raise Exception(emessage)
            
    print('SF # pulses and sequence success')  
    end = time.time()

# print('SF dataframe time: ' + str(end-start))  

# In[9]:
## puck info.

if chan_enab[-1] == 29:
    puckstates = []
    for i in range(0, len(ys_arrpuck[0])):
        puckval = st.mode(ys_arrpuck[0][i])
        puckstates.append(puckval)           
    puckstates = np.asarray(puckstates)
else:
    puckstates = np.zeros((numRuns),dtype=np.int32)  ## if no chan 29, then just fill w 0s (false)

## dataframe for He Norms
cols = ['pulse', 'norms','puckval']
pulses = range(numRuns)
normsData = [pulses, HeNorms[0], puckstates]

df_HE = pd.DataFrame({cols[0]: normsData[0], ## this is actually He and puckstate
                      cols[1]: normsData[1],
                      cols[2]: normsData[2]})

# In[10]:

if spinFlipFlag == 0:
    pass
if spinFlipFlag == 1:
    df_SF.to_hdf(savefilename + '.h5', f'df_0', mode='w') ## this "deletes" any previous data in the file name

for idx, df in enumerate([df_SF, df_HE]):
    df.to_hdf(savefilename + '.h5', f'df_{idx}', mode='a') # rerunning this without the above 'w' code will keep increasing file size

# ## end of file creation ##

fullend = time.time()
print('full processing time: ' + str(fullend-fullstart))  
print('finished ' + str(datetime.now())) 
print('\n')

