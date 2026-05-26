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
import h5py

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='RuntimeWarning: overflow encountered in multiply')

analysisdir = os.getcwd()
basedir = os.path.dirname(os.getcwd())+'/'
os.chdir(basedir)

############## real running ##############

for arg in sys.argv:
    run_num=str(arg).zfill(5)
    # print(run_num)

chan_enab = int(sys.argv[-1])
run_start=str(sys.argv[1]).zfill(5)
run_end=str(sys.argv[2]).zfill(5)
run_num=str(sys.argv[3]).zfill(5)

datadir = 'D:/LANSCE_FP12_2023/data/' ## add directory of hard drive
uniquefolder = "runs" + str(run_start) + "-" + str(run_end) +"/"
SFNormFile = 'SF_Norm_files/'+uniquefolder+run_num

processedasymfolder = '/processed_data/'+uniquefolder+'asym_U/'
AsymSavename = os.getcwd()+processedasymfolder+run_num+'_U'
logger.add(basedir+"/processed_data/" + uniquefolder + '0_ErrorLog_'+run_start+'_'+run_end+'_U.txt', delay = False)

# cannot handle all 24 detectors at once
# chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) ## all
# chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,24]) ## downstream _D
chan_enab = np.array([12,13,14,15,16,17,18,19,20,21,22,23,24]) ## upstream _U

if not os.path.exists(os.getcwd()+processedasymfolder):
    # Create the directory
    # os.makedirs(os.getcwd()+processedONOFFfolder)
    os.makedirs(os.getcwd()+processedasymfolder)
    # os.makedirs(os.getcwd()+processedasymfolder_bg)
    print("Directory created successfully")
else:
    pass

##########################################

start = time.time()
fullstart = time.time()
read_data = []
fileLength = []

def open_file():
    for el in chan_enab:
        # f = open(datadir + folder + 'run' + run_num + "_ch" + str(el) + ".bin", 'rb')
        f = open(datadir+uniquefolder + 'run' + str(run_num) + "_ch" +str(el) + ".bin", 'rb')
        read_data.append(np.fromfile(file=f, dtype=np.uint16))
        f.close()
        fileLength.append(len(read_data[-1]))
    return read_data, fileLength

open_file()

fileLength = np.asarray(fileLength)
read_data = np.asarray(read_data) ## in detector's case, all are the same size samples, so can do read_data as np array

if chan_enab[-1] != 24:
    emessage = ('last channel is not 6Li detector')
    logger.error('run '+run_num + emessage)
    raise Exception(emessage)

end = time.time()

print('\n saving processed data to ' + AsymSavename)
print("Channel is " + str(chan_enab))

# In[2]:
# Store the big header for each channel in arrays
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

targetDict = {0: "La", 1: "Tb2O3", 2: "Yb2O3", 3: "Sm2O3", 4: "Er2O3", 5: "Ho2O3", 6: "Pr", 7: "other"}
foilDict = {0: "TBD", 1: "TBD", 2: "TBD", 3: "TBD", 4: "TBD", 5: "TBD", 6: "TBD", 7: "other"}

target=(read_data[0][5]&0x00F0)>>4
foil=read_data[0][5]&0x000F
targetFlag = read_data[0][5]>>8&1
foilFlag = read_data[0][5]>>9&1
spinFiltFlag = read_data[0][5]>>10&1
spinFlipFlag = read_data[0][5]>>11&1
shutterFlag = read_data[0][5]>>12&1
facilityTrigFlag = read_data[0][5]>>13&1

if targetFlag:
    ## CATCHES MISLABELED PR RUNS 08.20.25
    if uniquefolder[4:9] in ('12686', '13280', '13564'): ## Pr run #s for 2023 LANSCE run
        target = targetDict[6] ## corresponds to Pr after change
    else:
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

BoardID      = np.asarray(BoardID,dtype =np.uint32) ## after changing to Numpy ver >2.0, dtype should be explicit. uint32 is the smallest size that does not ruin the xs below
recordLength = np.asarray(recordLength,dtype =np.uint32)
numSamples   = np.asarray(numSamples,dtype =np.uint32)
eventCounter = np.asarray(eventCounter,dtype =np.uint32)
decFactor    = np.asarray(decFactor,dtype =np.uint32)
chanDec      = np.asarray(chanDec, dtype=np.uint32)
postTrig     = np.asarray(postTrig,dtype =np.uint32)
groupStart   = np.asarray(groupStart, dtype=np.uint32)
groupEnd     = np.asarray(groupEnd, dtype=np.uint32)
timestamp    = np.asarray(timestamp, dtype=np.uint32)
sizeFirstEvent = np.asarray(sizeFirstEvent, dtype=np.uint32)
TTT = np.asarray(TTT, dtype=np.uint32)

# print("Target is " + target)
# print("Foil is " + foil)
# print("Shutter is open: " + str(bool(shutterFlag)))
# print("Facility t0 is on: " + str(bool(facilityTrigFlag)))
# print("Spin flipper is on: " + str(bool(spinFlipFlag)))
# print("Spin filter is on: " + str(bool(spinFiltFlag)))
# print("Target is present: " + str(bool(targetFlag)))
# print("Foil is present: " + str(bool(foilFlag)))

if target == 'La':
    vparamsfileloc = basedir+'/processed_data/'+'0_vparams_La.h5'
elif target == 'Pr':
    vparamsfileloc = basedir+'/processed_data/'+'1_vparams_Pr.h5'
elif target == 'Tb2O3':
    vparamsfileloc = basedir+'/processed_data/'+'2_vparams_Tb.h5'
elif target == 'Ho203':
    vparamsfileloc = basedir+'/processed_data/'+'3_vparams_Ho.h5'
elif target == 'Tm203':
    vparamsfileloc = basedir+'/processed_data/'+'4_vparams_Tm.h5'
elif target == 'Yb203':
    vparamsfileloc = basedir+'/processed_data/'+'5_vparams_Yb.h5'
else:
    vparamsfileloc = 'missing'
    
print("Target is " + target + ', voigt fit params: ' + vparamsfileloc)
# In[3]:

# Determine the time axis for each channel
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

np.asarray(preTime)
np.asarray(startTime)
np.asarray(endTime)
np.asarray(resolution)

xs = np.asarray(xs) ## can convert xs to np array here because all detectors same numsamples

# In[4]:

start=time.time()

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
                eventcount_arr[i]=(eventCounter[i])
                byteCounter = 20
            else:
                ETTT_arr[i]=(data[i][byteCounter]+(data[i][byteCounter+1]<<16)+(data[i][byteCounter+2]<<32))
                eventcount_arr[i]=(data[i][byteCounter+4]+(data[i][byteCounter+5]<<16))
                byteCounter += 6
            for j in range(0, numSamps[i]):
                #if j == 0:
                    #ys_arr[i].append([])
                #print(byteCounter)
                ys_arr[i][eventCount][j]=data[i][byteCounter]
                byteCounter += 1
            eventCount += 1
    return ys_arr, ETTT_arr, eventcount_arr

start=time.time()
ys_arr, ETTT_arr, eventcount_arr  = dataread(read_data, chan_enab, fileLength, numSamples) ##hardcoded channels for coils

end = time.time()
print('dataread from binary time: ' + str(end-start))

# In[5]:

timeDif=[]
for i in range(0,len(chan_enab)):
    timeDif.append([])
    for j in range(len(ETTT_arr[i])-1):
        timeDif[i].append((ETTT_arr[i][j+1]-ETTT_arr[i][j])*8)
#     print("Min time difference for channel", chan_enab[i], "is", min(timeDif[i]), "ns")
#     print("Max time difference for channel", chan_enab[i], "is", max(timeDif[i]), "ns \n")

# In[6]:

# Load in SF and He normalization information ##
try:
    df_SF = pd.read_hdf(SFNormFile + '.h5', key='df_0')
    df_HE = pd.read_hdf(SFNormFile + '.h5', key='df_1')
except Exception as e:
    logger.error('run '+run_num + ' failed during SFNormFile load')
    logger.exception(e)

SF_Sort_arr = df_SF[['nicknames', 'transition_locations']].to_numpy().T
He_Norm_arr = df_HE[['pulse', 'norms','puckval']].to_numpy().T

if shutterFlag == 0:
    print('Shutter closed. NormFactor set to 1')
    NormFactor = 1
if shutterFlag == 1:
    NormFactor = 1000000 ## He integrals are huge, this normalizes all of those by a constant value for ease of use

HeNorms= (He_Norm_arr[1])/NormFactor

# In[7]:
# basesub and plotting ##
start = time.time()

baseL = 0
baseR = int(((preTime[0]-groupStart[0])*0.70)/chanDec[0])  ##70% before the trigger
numRuns = int((fileLength[0]-20-numSamples[0])/(numSamples[0]+6)+1)
legend =  ['NaI', 'R']

s = 20 ## pulse to look at
t=s+1

def plotter(ys, xs, baseR, numpoints):
    tempys_basesub = np.zeros((len(ys), numRuns,numpoints[0]), dtype=float)
    for i in range((len(ys))):
        for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
            tempys_basesub[i][pulse]=np.subtract(ys[i][pulse], np.mean(ys[i][pulse][baseL:baseR]))
        for j in range(s, t): ## plot only interested pulses
            plt.plot(xs[i], tempys_basesub[i][j]) #label=legend[i]) #+str(sums[1][j])) ## sums[j] will not work for more than just TR   
            plt.axvline(xs[0][baseL], ls = '--')
            plt.axvline(xs[0][baseR], ls = '--')
            plt.axvline(xs[0][baseR+5], ls = '--', c ='r') ## BaseR+5 line marks the beginning of the integral, until the end of samples.
            plt.legend()
            
# plotter(ys_arr[9:], xs[9:], baseR, numSamples) ##plot coils

@njit ## jit is faster for large # channels, slower for small # channels
def basesub(ys, baseRight, numpoints): 
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
    return tempys_basesub

@njit ## jit is faster for large # channels, slower for small # channels
def basesub_norm(ys, baseRight, numpoints): 
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
        tempys_basesub[pulse]=tempys_basesub[pulse]/HeNorms[pulse] 
    return tempys_basesub

ys_basesub = np.zeros((len(ys_arr), numRuns,numSamples[0]), dtype=np.float64)

for i in range(len(ys_basesub)): ## feeding y arrays into function 1 channel at  a time is faster than all at once
    ys_basesub[i] = basesub(ys_arr[i], baseR, numSamples)

ys_basesub[-1] = ys_basesub[-1]*-1 ## invert 6Li to positive signal. Comment out if not using

end = time.time()
# print('plotting and/or base subtraction time: ' + str(end-start))            

# In[8]:
# use 6Li t0 for all instead of for themselves individually ##
start = time.time()

NaIthresh=2000
Li6thresh=1000
threshold_array = (np.full(len(ys_basesub), NaIthresh))
threshold_array[-1] = Li6thresh

# njit ## numba does not support reversed, but this could be changed if it's slow
def find_offset(ys, thresharr):
    xCrosses = np.zeros((len(ys), numRuns)) #outer array is crossing arrays for given channel, inner array is crossing for each event
    offset = np.zeros((len(ys), numRuns), dtype=np.int32) ##offset in bins for each channel, each pulse
    modeCrosses = np.zeros((len(ys)), dtype=np.float64)
    for i in reversed(range(len(ys))):
        for p in range(len(ys[i])):
            xing = np.argmax(ys[i][p] > thresharr[i])
            xCrosses[i][p] = xing
        modeCrosses[i] = (st.mode(xCrosses[i])) #find the most typical crossing value for each channel
        for p in range(len(xCrosses[i])):
            offset[i][p] = (modeCrosses[-1] - xCrosses[i][p]) ## make sure this is the correct sign
    if (np.all(xCrosses[-1])) == False:
        emessage = ('ERROR: 6Li threshold was not reached for at least one pulse')
        logger.error('run '+run_num + emessage)
        raise Exception(emessage)
    return offset, xCrosses, modeCrosses
                           
offset, xCrosses, modeCrosses = find_offset(ys_basesub, threshold_array)

end = time.time()
# print('finding offset time: ' + str(end-start))  

# In[9]:
## numba versions for array shifting
@njit
def pad_numba(array, pad_width, constant_value=0.0):
    padded = np.empty(len(array) + 2 * pad_width, dtype=array.dtype)
    for i in range(pad_width):
        padded[i] = constant_value
    for i in range(len(array)):
        padded[i + pad_width] = array[i]
    for i in range(pad_width):
        padded[len(array) + pad_width + i] = constant_value
    return padded

@njit
def roll_numba(array, shift):
    n = len(array)
    result = np.empty_like(array)
    for i in range(n):
        result[(i + shift) % n] = array[i]
    return result

# In[9]:
# extend all arrays by a value, check that the max number of offset on 6Li is less than that value ##
del read_data 
start = time.time()

n_channels, n_pulses, og_length =  ys_basesub.shape[0], ys_basesub.shape[1], ys_basesub.shape[2]
extendedRange = 3 ## must be a positive value which to extend ys_arr
if abs(max(offset[-1], key = abs)) > extendedRange: ## if the max offset of 6Li is >extendedRange, something is wrong
    emessage = ('ERROR: largest offset greater than extended range')
    logger.error('run '+run_num + emessage)
    raise Exception(emessage)
new_length = og_length - 2*extendedRange - 2  ##should be the new length of the array

try:
    ys_cut = np.zeros((n_channels, n_pulses, new_length))
    xs_cut = np.zeros((len(ys_cut), len(ys_cut[0][0])))
except Exception as e:
    logger.error('run '+run_num + ' failed during ys_cut array creation')
    logger.exception(e)

# change np.pad and .roll to be numba applicable##
@njit
def align_cut_numba(ys, xs_arr, extendedr):
    tempys_ext = np.zeros((len(ys), len(ys[0])+extendedr*2), dtype=np.float64)
    tempys_cut = np.zeros((len(ys), (len(tempys_ext[0])-((extendedr*2)+1)*2)))
    tempxs_cut = np.zeros(len(tempys_cut[0]))
    for p in range(len(ys)):
        tempys_ext[p] = pad_numba(ys[p], extendedr) ## asumes constant fill = 0
        tempys_ext[p] = roll_numba(tempys_ext[p],offset[-1][p]) ## assumes 6Li at -1 position
        tempys_cut[p] = tempys_ext[p][((extendedr*2)+1):-((extendedr*2)+1)].copy() ## cut by 7 (if extRange == 3)
        tempys_cut[p] = tempys_cut[p]/HeNorms[p] ## normalize by 3He integral  ## comment out if using basesub_norm
    x_cut_amt = int((len(ys[0]) - len(tempys_cut[0]))/2)
    tempxs_cut = xs_arr[x_cut_amt:-x_cut_amt].copy()
    return tempys_cut, tempxs_cut

# looping every channel through function is 5x faster ##
try:
    for i in range(len(ys_basesub)):
        ys_cut[i], xs_cut[i] = align_cut_numba(ys_basesub[i], xs[i], extendedRange)
except Exception as e:
    logger.error('run '+run_num + ' failed aligning and cutting')
    logger.exception(e)

del ys_basesub

end = time.time()
print('aligning and cutting time: ' + str(end-start))            

# begin SF organization ##
# In[11]:

def organize_SF(SFsort_info): ## sometimes pulse 0 has the state switch. In that case, need to account by if clauses below
    counter = 0
    seq = 0
    seq_arr = ([[],[],[]])
    smallerseq = []
    smallerstateis = []
    for i in range(len(SFsort_info[1])-(np.mod((len(SFsort_info[1])), 8))):  ##111 mod 8 = 7, so essentially 111-7 = 104
        counter = counter+1
        if counter < 8:
            if (SF_Sort_arr[1][i]) == 0: ## catches state switches at pulse 0
                smallerstateis.append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
                smallerseq.append(SFsort_info[0][i+1])
                seq = seq+1
                continue
            smallerstateis.append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
            smallerseq.append(SFsort_info[0][i+1])
        elif counter == 8:
            if ((SF_Sort_arr[1][i])+45) >= 5000: ## breaks for state switches at pulse 0
                print(((SF_Sort_arr[1][i])+5))
                seq = seq+1
                seq_arr[0].append(seq)
                seq_arr[1].append(smallerseq)   
                seq_arr[2].append(smallerstateis)
                seq_arr[0] = [x-1 for x in seq_arr[0]] ## reset so sequences are 1-14 instead of 2-15
                break
            seq = seq+1 ## otherwise continue regular sorting
            smallerstateis.append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
            smallerseq.append(SFsort_info[0][i+1])
            seq_arr[0].append(seq)
            seq_arr[1].append(smallerseq)   
            seq_arr[2].append(smallerstateis)
            smallerseq = []
            smallerstateis = []
            counter  = 0
    return seq_arr

def find_leftover(SFsort_info, seq_arr): ## in case we want to use the other 6 states left over
    left = [[seq_arr[0][-1]+1],[],[]]
    counter = 0
    for i in range((len(SFsort_info[1])-(np.mod((len(SFsort_info[1])), 8))), len(SFsort_info[1])-1):
        counter = counter+1
        if counter < 8:
            left[1].append(SFsort_info[0][i+1])
            left[2].append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
    return left

try:
    sequence = organize_SF(SF_Sort_arr)
    if len(sequence[0]) == 14: ## catches state switches at pulse 0, leftovers are at the end of the regular sequence
        leftovers = [[sequence[0][-1]],[sequence[1][-1]],[sequence[2][-1]]]
        for i in range(len(sequence)):
            sequence[i].pop(-1) ## deletes the leftovers sequence for state switches at pulse 0
    else:
        leftovers = find_leftover(SF_Sort_arr, sequence) ## otherwise can use normal function
except Exception as e:
    logger.error('run '+run_num + ' failed during sequencing')
    logger.exception(e)

# print('sequences '+str(sequence[0]))
print(str(len(sequence[0]))+' sequences with sequence order: '+str(sequence[1][0]))
# print(leftovers)

# In[12]:
#  add up pulses for their respective state, in each 8 step sequence ##

start = time.time()
sequence = np.asarray(sequence, dtype = object)

ON_sums = np.zeros((len(ys_cut), len(sequence[0]), len(ys_cut[0][0])), dtype=np.float64) ## 13 channels, 13 sequences, added pulses for ON
OFF_sums = np.zeros((len(ys_cut), len(sequence[0]), len(ys_cut[0][0])), dtype=np.float64) ## 13 channels, 13 sequences, added pulses for OFF

# @njit
def add_pulse(ys, SFarr):
    temp_ON = np.zeros((len(SFarr[0]), len(ys[0])), dtype=np.float64)
    temp_OFF = np.zeros((len(SFarr[0]), len(ys[0])), dtype=np.float64)
    for seq in range(0, len(SFarr[0])): ## for every sequence
        for state in range(0, len(SFarr[1][0])): ## for every state in the sequence
            s = SFarr[1][seq][state] ## try this to condense code. Basically, the state currently at
            if s==0 or s==3 or s==5 or s==6: ## these are ON states
                for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])): ##From 20-60 for example. SFarr[2] is the array of start to end pulses to sum
                    temp_ON[seq] = np.add(temp_ON[seq],ys[p]) ## start with zeros, add to each iteratively
            if s==1 or s==2 or s==4 or s==7: ## these are OFF states
                for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])):
                    temp_OFF[seq] = np.add(temp_OFF[seq],ys[p])
    return temp_ON, temp_OFF

for i in range(len(ys_cut)):
    ON_sums[i], OFF_sums[i] = add_pulse(ys_cut[i], sequence)

end = time.time()
# print('summing pulses into their states time: ' + str(end-start))  

# In[13]:
# incorporating new polyfit functions
@njit
def polyN_fit_errors(x, y, degree):
    n = x.shape[0]
    A = np.empty((n, degree + 1))
    for i in range(n):
        xi = x[i]
        A[i, degree] = 1.0
        for j in range(degree - 1, -1, -1):
            A[i, j] = A[i, j + 1] * xi

    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = np.linalg.solve(ATA, ATy)

    # Calculate residuals and variance
    y_fit = A @ coeffs
    residuals = y - y_fit
    dof = n - (degree + 1)
    residual_variance = np.sum(residuals**2) / dof

    # Covariance matrix and standard errors
    cov_matrix = np.linalg.inv(ATA) * residual_variance
    stderr = np.sqrt(np.diag(cov_matrix))

    return coeffs, stderr

def polyN_predict(x, coeffs):
    n = x.shape[0]
    y = np.empty(n)
    degree = coeffs.shape[0]
    for i in range(n):
        xi = x[i]
        yi = coeffs[0]
        for j in range(1, degree):
            yi = yi * xi + coeffs[j]
        y[i] = yi
    return y

# In[15]:
# bg fitting

## load in voigt values and regions
v_sigmas = []
v_gammas = []
v_reslocs = []
with h5py.File(vparamsfileloc, 'r') as f: ## new arr_sizer
    channels_all = list(f.keys())
    bg_reg1 = f.attrs.get('bg_reg_bef')
    bg_reg2 = f.attrs.get('bg_reg_aft')
    res_reg = f.attrs.get('res_reg')
    for i in range(chan_enab[0], chan_enab[-1]):
        v_sigmas.append(f[channels_all[i]].attrs.get('sigma_[mean,mode,std]')) ## each of these for each channel
        v_gammas.append(f[channels_all[i]].attrs.get('gamma_[mean,mode,std]'))
        v_reslocs.append(f[channels_all[i]].attrs.get('res_loc_[mean,mode,std]')) ## in ms, not bins
    f.close()

v_sigmas = np.asarray(v_sigmas)
v_gammas = np.asarray(v_gammas)
v_reslocs = np.asarray(v_reslocs)
fullrange = bg_reg2[1]-bg_reg1[0]

## one f_g/l/v for each channel
f_g = 2*(v_sigmas[:,0])*np.sqrt(2*np.log(2)) ## Gaussian FWHM
f_l = 2*(v_gammas[:,0]) ##Lorentzian FWHM
f_v = 0.5343*f_l + np.sqrt(0.2169*f_l**2+f_g**2)  ## this is the FWHM from wikipedia. # one FWHM value for each channel

# In[18]:

from scipy.special import voigt_profile
from scipy.optimize import curve_fit
start = time.time()

def bg_fitsubtract(bef_res_reg, aft_res_reg, ys, order): ## before/after resonance region [start:end] respectively, ys[ch] to fit
    binstot = aft_res_reg[1]-bef_res_reg[0]  ## total number of bins in whole region
    x1 = np.arange(bef_res_reg[0], bef_res_reg[1],1)
    x2 = np.arange(aft_res_reg[0], aft_res_reg[1],1)
    x = np.append(x1,x2)
    fullx = np.arange(bef_res_reg[0], aft_res_reg[1],1) ## an array of every x bin in entire region
    ys_bgsub = []
    for seq in range(0, len(ys)): ## number of sequences, usually 13
        fitdata1 = ys[seq][bef_res_reg[0]: bef_res_reg[1]]
        fitdata2 = ys[seq][aft_res_reg[0]: aft_res_reg[1]]
        datasplice = np.append(fitdata1, fitdata2)
        y = datasplice
        
        coeffs, errs = polyN_fit_errors(x, y,order) 
        y_fit = polyN_predict(x, coeffs) ## run the function with the coeff just found
        fullpoly_y2 = polyN_predict(fullx, coeffs)

        bgsubtracted = ys[seq][bef_res_reg[0]:aft_res_reg[1]] - fullpoly_y2  ## subtracts RealData-BackgroundFit
        ys_bgsub.append(bgsubtracted)
    return ys_bgsub

# background subtraction currently only for NaI detectors
num_fittingchs = len(ON_sums)-1  ## removes the Li detector, assumes it is there
ON_bgsub = np.zeros((num_fittingchs,len(ON_sums[0]),fullrange), dtype = np.float64) ## channels, sequences, range of bg_ subtraction
OFF_bgsub = np.zeros((num_fittingchs,len(ON_sums[0]),fullrange), dtype = np.float64) ## channels, sequences, range of bg_ subtraction
fitorder = 4
for i in range(0, len(ON_sums)-1):
    if chan_enab[i] == 24:
        emessage = ('bg_ fit does not work for 6Li yet')
        logger.error('run '+run_num + emessage)
        raise Exception(emessage)
    ON_bgsub[i]  = bg_fitsubtract(bg_reg1,bg_reg2, ON_sums[i], fitorder)
    OFF_bgsub[i] = bg_fitsubtract(bg_reg1,bg_reg2, OFF_sums[i], fitorder)

# end = time.time()
# print('bg_ fitting time: ' + str(end-start))  

# In[19]:

## define new voigt functions using constants loaded in
def voigt_c(x, amp, c): ## takes sig, gamma, res_loc as constants from file  
    fit = voigt_profile(x-xloc[0], sig[0], gam[0])*amp + c  # ** changed to Vfit + constant
    return fit

def voigt_fitting_c(ogbin_res_reg, aftrebin_res_reg,xs,ys):
    fit_curves = [] ## now this function uses the res region as opposed to the not-background-region
    parameters = []
    constants = []
    for seq in range(0, len(ys)): ## number of sequences, usually 13
        ydata = ys[seq][aftrebin_res_reg[0]:aftrebin_res_reg[1]]
        popt, pcov = curve_fit(
            voigt_c, xs, ydata,
            p0=[np.max(ydata), 0],  
            bounds = ([0, -np.inf], [np.inf, np.inf])  
        )
        fitted_curve = voigt_c(xs, popt[0], popt[1])  
        fit_curves.append(fitted_curve)

        amp_err = np.sqrt(np.diagonal(pcov))[0]
        c_err = np.sqrt(np.diagonal(pcov))[1]

        parameters.append([popt[0], amp_err])
        constants.append([popt[1], c_err])   
    return fit_curves, parameters, constants

## go back to original voigt fitting but this time with suggested guesses and bounds on possible param

def voigt2(x, s, g, amp, c): 
    fit = voigt_profile(x-xloc[0], s, g)*amp + c  
    return fit

def voigt_fitting(ogbin_res_reg, aftrebin_res_reg,xs,ys):
    fit_curves = []
    parameters = []
    constants = []
    for seq in range(0, len(ys)): ## number of sequences, usually 13
        ydata = ys[seq][aftrebin_res_reg[0]:aftrebin_res_reg[1]]
        a_guess, s_guess, g_guess = np.max(ydata), sig[0], gam[0]
        s_1std, g_1std = sig[2]/2, gam[2]/2  ## allow for half of the std
        ## set the guesses as the loaded values, and the bounds on sigma,gamma as +/- 1 std
        popt, pcov = curve_fit(
            voigt2, xs, ydata,
            p0=[s_guess,g_guess,a_guess,0],  
            bounds = ([s_guess-s_1std,g_guess-g_1std,0,-np.inf], [s_guess+s_1std,g_guess+g_1std,np.inf,np.inf]) 
        )
        fitted_curve = voigt2(xs, popt[0],popt[1],popt[2],popt[3]) 
        fit_curves.append(fitted_curve)

        fit_params = np.array([popt[0], popt[1], popt[2]]) 
        fit_errs = np.sqrt(np.diagonal(pcov))[:3]          
        const_val = popt[3]                                
        const_err = np.sqrt(np.diagonal(pcov))[3]          

        parameters.append([fit_params,fit_errs])
        constants.append([const_val, const_err])
    return fit_curves, parameters, constants

# In[21]:

if res_reg[0] < bg_reg1[1] or res_reg[1]>bg_reg2[0]:
    emessage = ('Declared Background region and Resonance region have overlapping fitting regions')
    logger.error('run '+run_num + emessage)
    raise Exception(emessage)

# In[22]:

# start = time.time()

res_size    = res_reg[1]-res_reg[0] 
newresstart = res_reg[0]-bg_reg1[0]
newresend   = (res_reg[1]-res_reg[0])+newresstart
new_res_reg = [newresstart, newresend]  ## this just realigns in terms of bins

xdata = xs_cut[0][res_reg[0]:res_reg[1]]*1e-6  ## just change all xs to ms and one array

ON_vfit = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),res_size), dtype = np.float64) ## channels, sequences, range of V_ subtraction
OFF_vfit = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),res_size), dtype = np.float64) ## channels, sequences, range of V_ subtraction

ON_vfit_params  = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2), dtype = np.float64) ## channels, sequences, [amp thing, amp thing err]
OFF_vfit_params = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2), dtype = np.float64)

ON_vfit_const  = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2), dtype=np.float64)  ## [const, const_err]
OFF_vfit_const = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2), dtype=np.float64)

sig_ch = np.zeros((len(ON_vfit),2), dtype=np.float64) ## 1 sigma +/- error for each channel
gam_ch = np.zeros((len(ON_vfit),2), dtype=np.float64) ## 1 gamma +/- error for each channel

ON_vfit_params_varied  = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2,3), dtype = np.float64) ## channels, sequences,[params, param_errs], [sigma, gamma, amp thing] NO xSHIFT
OFF_vfit_params_varied = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2,3), dtype = np.float64)

ON_vfit_const_varied  = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2), dtype=np.float64)  ## [const, const_err]
OFF_vfit_const_varied = np.zeros((len(ON_bgsub),len(ON_bgsub[0]),2), dtype=np.float64)

for ch in range(0, len(ON_bgsub)):
    sig  = v_sigmas[ch] ## the definitions of these changes per channel for func voigt_c
    gam  = v_gammas[ch]
    xloc = v_reslocs[ch]
    if chan_enab[ch] == 24:
        emessage = ('bg_ fit does not work for 6Li yet')
        logger.error('run '+run_num + emessage)
        raise Exception(emessage)
    try:
        ON_vfit[ch],  ON_vfit_params[ch],  ON_vfit_const[ch]  = voigt_fitting_c(res_reg,new_res_reg,xdata, ON_bgsub[ch])
        OFF_vfit[ch], OFF_vfit_params[ch], OFF_vfit_const[ch] = voigt_fitting_c(res_reg,new_res_reg,xdata, OFF_bgsub[ch])

        trash, ON_vfit_params_varied[ch],  ON_vfit_const_varied[ch]  = voigt_fitting(res_reg,new_res_reg,xdata, ON_bgsub[ch])  ## these 2 are the original w varying sig,gam
        trash, OFF_vfit_params_varied[ch], OFF_vfit_const_varied[ch] = voigt_fitting(res_reg,new_res_reg,xdata, OFF_bgsub[ch])

    except Exception as e:
        logger.error('run '+run_num +' ch '+ str(chan_enab[ch])+' failed during Voigt fitting')
        logger.exception(e)

end = time.time()
print('bg_ fitting time: ' + str(end-start)) 

# In[23]:
# sum up all pulses, and then calculate asymmetry using "amplitude"
# new asym integrating over a curve and using FWHM of voigt function

start = time.time()

xs_full_res = xs_cut[0][bg_reg1[0]:bg_reg2[1]]*1e-6 ## this is in ms, not bins
fwhm_bins = []
for ch in range(0, len(f_v)):
    fwhm_bin_left = np.where(xs_full_res>=v_reslocs[ch][0]-f_v[ch]/2)[0][0]
    fwhm_bin_right = np.where(xs_full_res>=v_reslocs[ch][0]+f_v[ch]/2)[0][0]
    fwhm_bins.append([fwhm_bin_left, fwhm_bin_right])
# print(fwhm_bins)

asym_int = np.zeros((len(ON_vfit)), dtype=np.float64) ## 1 Asym for each channel, and its error
def asym_integral(ON_data, OFF_data, integral_bins): 
    tempasym = 0
    temperr  = []
    fwhm_l = integral_bins[0]
    fwhm_r = integral_bins[1]
    for seq in range(0, len(ON_data)): ## number of sequences
        A_plus = np.sum(ON_data[seq][fwhm_l:fwhm_r]) 
        A_min  = np.sum(OFF_data[seq][fwhm_l:fwhm_r])
        seqasym = ((A_plus-A_min) / (A_plus+A_min))
        tempasym = np.add(seqasym,tempasym)
    return tempasym  

for i in range(len(ON_vfit)):
    asym_int[i] = asym_integral(ON_bgsub[i], OFF_bgsub[i], fwhm_bins[i]) 

asym_raw = np.zeros((len(ON_sums), len(ON_sums[0][0])), dtype=np.float64) ## 1 Asym for each channel, not for each sequence
def asymraw(ON_arr, OFF_arr):
    tempasym = np.zeros((len(ON_arr[0])), dtype=np.float64)
    for seq in range(0, len(ON_arr)): ## number of sequences
        seqasym = ((ON_arr[seq]-OFF_arr[seq]) / (ON_arr[seq]+OFF_arr[seq]))
        tempasym = np.add(seqasym,tempasym)
    normedasym = tempasym #/len(ON_arr) ## Normalize later by # sequences
    return normedasym

for i in range(len(ON_sums)):
    asym_raw[i] = asymraw(ON_sums[i], OFF_sums[i])

# In[25]:

## add and avg all sigma/gamma/xloc for ON and OFF per channel. Should not be dpendent on spin state.
def sum_param(ON_params_arr, OFF_params_arr, key):  ## key choose between sigma gamma NOT xloc
    if key != 'sigma' and key != 'gamma':
        emessage = ('not a valid Voigt fit parameter to sum')
        logger.error('run '+run_num + emessage)
        raise Exception(emessage)
    if key == 'sigma':
        param = 0
    if key == 'gamma':
        param = 1
#     if key == 'xloc':
#         param = 2
    tempsum = 0
    temperr  = []
    for seq in range(0, len(ON_params_arr)): ## number of sequences
        seqsum = (ON_params_arr[seq][0][param]+OFF_params_arr[seq][0][param])/2  # "normalize" with /2 but not for #sequences
        ## [0] above corresponds to the real values of the paramters, as opposed to their errors
        ON_err = ON_params_arr[seq][1][param]
        OFF_err = OFF_params_arr[seq][1][param]
        ## above [1] is used which corresponds to the error in "parameter"
        ON_deriv  =  1  ## left over from asym error to keep error prop consistent
        OFF_deriv =  1
        ## use err prop to get below
        seq_err = np.sqrt((ON_deriv**2)*(ON_err**2)+(OFF_deriv**2)*(OFF_err**2))/2
        temperr.append(seq_err) ## collect all errors for each sequence
        tempsum = np.add(seqsum,tempsum)
    ## add error or each sequence in quad.
    toterr = np.sqrt(sum([i**2 for i in temperr]))  ## use list comprehension for sum of squares
    out = [tempsum,toterr]
    return out

for i in range(len(ON_vfit)):
    sig_ch[i] = sum_param(ON_vfit_params_varied[i], OFF_vfit_params_varied[i], key='sigma')
    gam_ch[i] = sum_param(ON_vfit_params_varied[i], OFF_vfit_params_varied[i], key='gamma') 

# In[27]:

# asym using "amplitude" parameter, with error prop
asym_ch_err = np.zeros((len(ON_vfit),2), dtype=np.float64) ## 1 Asym for each channel, and its error
asym_ch_err_varied = np.zeros((len(ON_vfit),2), dtype=np.float64) ## 1 Asym for each channel, and its error

def asym3_err2(ON_params_arr, OFF_params_arr):
    tempasym = 0
    temperr  = []
    for seq in range(0, len(ON_params_arr)):
        A_plus = ON_params_arr[seq][0]  # amplitude
        A_min  = OFF_params_arr[seq][0]  # amplitude

        denom = A_plus + A_min
        if denom == 0:
            continue

        seqasym = ((A_plus-A_min) / denom)
        ON_err = ON_params_arr[seq][1]  # amp error
        OFF_err = OFF_params_arr[seq][1]  # amp error
        ON_deriv  =  2*A_min/ (denom**2)
        OFF_deriv = -2*A_plus/(denom**2)
        seq_err = np.sqrt((ON_deriv**2)*(ON_err**2)+(OFF_deriv**2)*(OFF_err**2))
        temperr.append(seq_err)
        tempasym = np.add(seqasym,tempasym)

    toterr = np.sqrt(sum([i**2 for i in temperr])) if len(temperr) > 0 else 0.0
    return [tempasym,toterr]

for i in range(len(ON_vfit)):
    asym_ch_err[i] = asym3_err2(ON_vfit_params[i], OFF_vfit_params[i])
    asym_ch_err_varied[i] = asym3_err2(ON_vfit_params_varied[i,:,:,2], OFF_vfit_params_varied[i,:,:,2]) ## 2 here is amplitude [ch, all sequences, parameter & error, amplitude]

    
end = time.time()
print('calc asyms: ' + str(end-start)) 

# In[28]:

puck_thresh = 2500

if np.mean(He_Norm_arr[2])==0: ## catches runs with no ch. 29. 0/1 for True/False key
    puck_in = -1  ## to differentiate from the other 2 true/false states
    puck_in_pulses = ['No ch.29 or no puck information'] 
else:
    puck_in_pulses = np.where(He_Norm_arr[2]>=puck_thresh)[0] ## empty array for runs w/out puck
    if len(puck_in_pulses) == 0: ## puck_in state is false for empty array
        puck_in = 0
    else:
        puck_in = 1 

# In[29]
# save all on and off values and asymm + paramters. save out.
start = time.time()

with h5py.File(AsymSavename+'.h5', 'w') as hdf5_file:
    hdf5_file.create_dataset('xs ', data=xs_cut[0]) ## all xs are the same, even though they are per channel...
    hdf5_file.attrs['puck_state'] = puck_in ## adding the puck information
    hdf5_file.attrs['puck_pulses'] = puck_in_pulses ## will be empty if puck is not in (usually)
    hdf5_file.attrs['num_pulses'] = numRuns ## some runs do not have 5000 pulses! Usually the ones at the end
    hdf5_file.attrs['sequences'] = len(sequence[0])
    for i in range(0,len(asym_ch_err)): 
        Ch_grp = hdf5_file.create_group('ch_'+str(np.char.zfill(str(chan_enab[i]), 2)))
        Ch_grp.attrs['asym_amp'] = asym_ch_err[i]
        Ch_grp.attrs['asym_amp_varied'] = asym_ch_err_varied[i]
        Ch_grp.attrs['asym_integral'] = asym_int[i]
        Ch_grp.attrs['used_xloc']  = v_reslocs[i]
        Ch_grp.attrs['used_sigma'] = v_sigmas[i]
        Ch_grp.attrs['used_gamma'] = v_gammas[i]
        Ch_grp.attrs['FWHM_bins'] = fwhm_bins[i]
        Ch_grp.create_dataset('asym_raw', data=asym_raw[i]) ## raw asym for plotting etc 
        ON_subgrp = Ch_grp.create_group('ON')
        OFF_subgrp = Ch_grp.create_group('OFF')
        ON_subgrp.attrs['for_each_sequence'] = ['parameter ("amplitude")', 'and its error']
        OFF_subgrp.attrs['for_each_sequence'] = ['parameter ("amplitude")', 'and its error']

        ON_subgrp.create_dataset('parameters', data=ON_vfit_params[i])
        OFF_subgrp.create_dataset('parameters', data=OFF_vfit_params[i])

        ON_subgrp.create_dataset('constant', data=ON_vfit_const[i])
        OFF_subgrp.create_dataset('constant', data=OFF_vfit_const[i])

        ON_subgrp.create_dataset('parameters_varied', data=ON_vfit_params_varied[i])   
        OFF_subgrp.create_dataset('parameters_varied', data=OFF_vfit_params_varied[i]) 

        ON_subgrp.create_dataset('constant_varied', data=ON_vfit_const_varied[i])   
        OFF_subgrp.create_dataset('constant_varied', data=OFF_vfit_const_varied[i]) 

        ON_subgrp.create_dataset('Vfit_curves',data=ON_vfit[i])
        OFF_subgrp.create_dataset('Vfit_curves',data=OFF_vfit[i])
    Ch_grp = hdf5_file.create_group('ch_'+str(np.char.zfill(str(chan_enab[-1]), 2))) ## this is for 6Li data
    Ch_grp.create_dataset('asym_raw', data=asym_raw[-1]) ## 6Li channel is always last...

end = time.time()
# print('saving hdf5: ' + str(end-start))

# In[31]:

fullend = time.time()
print('full processing time: ' + str(fullend-fullstart))  
print('finished ' + str(datetime.now())) 
print('\n')

# ## end of data processing ##
