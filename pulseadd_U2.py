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
statefileloc = basedir+'\SF_Norm_files\TR_R_expected_avgs_stds_afterclip.csv'
processedpolfolder = '/processed_data/'+uniquefolder+'pulseadd_U/'
polSavename = os.getcwd()+processedpolfolder+run_num+'_pulseadd_U'
logger.add(basedir+"/processed_data/" + uniquefolder + '0_ErrorLog_'+run_start+'_'+run_end+'pulseadd_U.txt', delay = False)

if not os.path.exists(os.getcwd()+processedpolfolder):
    # Create the directory
    os.makedirs(os.getcwd()+processedpolfolder)
    print("Directory created successfully")
else:
    pass

# cannot handle all 24 detectors at once, memory issue... can look into np.empty and deleting variables if needed ##
# chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) ## all
# chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,24]) ## downstream _D
chan_enab = np.array([12,13,14,15,16,17,18,19,20,21,22,23,24]) ## upstream _U

start = time.time()
fullstart = time.time()

read_data = []
fileLength = []

def open_file():
    for el in chan_enab:
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

print('saving processed data to ' + polSavename)
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

BoardID = np.asarray(BoardID) 
recordLength = np.asarray(recordLength)
numSamples = np.asarray(numSamples)
eventCounter = np.asarray(eventCounter)
decFactor = np.asarray(decFactor)
chanDec = np.asarray(chanDec)
postTrig = np.asarray(postTrig)
groupStart = np.asarray(groupStart)
groupEnd = np.asarray(groupEnd)
timestamp = np.asarray(timestamp)
sizeFirstEvent = np.asarray(sizeFirstEvent)
TTT = np.asarray(TTT)

print("Target is " + target)
# print("Foil is " + foil)
# print("Shutter is open: " + str(bool(shutterFlag)))
# print("Facility t0 is on: " + str(bool(facilityTrigFlag)))
# print("Spin flipper is on: " + str(bool(spinFlipFlag)))
# print("Spin filter is on: " + str(bool(spinFiltFlag)))
# print("Target is present: " + str(bool(targetFlag)))
# print("Foil is present: " + str(bool(foilFlag)))

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
                #ETTT_arr[i].append(TTT[i])
                eventcount_arr[i]=(eventCounter[i])
                byteCounter = 20
            else:
                ETTT_arr[i]=(data[i][byteCounter]+(data[i][byteCounter+1]<<16)+(data[i][byteCounter+2]<<32))
                eventcount_arr[i]=(data[i][byteCounter+4]+(data[i][byteCounter+5]<<16))
                byteCounter += 6
            for j in range(0, numSamps[i]):
                #if j == 0:
                    #ys_arr[i].append([])
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
Li6thresh=400
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

if shutterFlag == 0:
    print('Shutter closed. No t0 offsets')
    offset = np.zeros((len(ys_basesub), numRuns), dtype=np.int32)
if shutterFlag == 1:
    offset, xCrosses, modeCrosses = find_offset(ys_basesub, threshold_array)

end = time.time()
# print('finding offset time: ' + str(end-start))  

# In[9]:

# extend all arrays by a value, check that the max number of offset on 6Li is less than that value ##
start = time.time()

extendedRange = 3 ## must be a positive value which to extend ys_arr
if abs(max(offset[-1], key = abs)) > extendedRange: ## if the max offset of 6Li is >extendedRange, something is wrong
    emessage = ('ERROR: largest offset greater than extended range')
    logger.error('run '+run_num + emessage)
    raise Exception(emessage)
try:
    ys_ext = np.zeros((len(ys_basesub), len(ys_basesub[0]), len(ys_basesub[0][0])+extendedRange*2), dtype=np.float64)
    ys_cut = np.zeros((len(ys_basesub), len(ys_basesub[0]), (len(ys_ext[0][0])-((extendedRange*2)+1)*2)))
    xs_cut = np.zeros((len(ys_cut), len(ys_cut[0][0])))
except Exception as e:
    logger.error('run '+run_num + ' failed during ys_cut array creation')
    logger.exception(e)

# cant use jit because np.pad is not supported ##
def align_cut(ys, xs_arr, extendedr):
    tempys_ext = np.zeros((len(ys), len(ys[0])+extendedr*2), dtype=np.float64)
    tempys_cut = np.zeros((len(ys), (len(tempys_ext[0])-((extendedr*2)+1)*2)))
    tempxs_cut = np.zeros(len(tempys_cut[0]))
    for p in range(len(ys)):
        tempys_ext[p] = np.pad(ys[p], extendedr, 'constant', constant_values=(0))
        tempys_ext[p] = np.roll(tempys_ext[p],offset[-1][p]) ## assumes 6Li at -1 position
        tempys_cut[p] = tempys_ext[p][((extendedr*2)+1):-((extendedr*2)+1)].copy() ## cut by 7 (if extRange == 3)
        tempys_cut[p] = tempys_cut[p]/HeNorms[p] ## normalize by 3He integral  ## comment out if using basesub_norm
    x_cut_amt = int((len(ys[0]) - len(tempys_cut[0]))/2)
    tempxs_cut = xs_arr[x_cut_amt:-x_cut_amt].copy()
    return tempys_cut, tempxs_cut

try:
    for i in range(len(ys_basesub)):
        ys_cut[i], xs_cut[i] = align_cut(ys_basesub[i], xs[i], extendedRange)
except Exception as e:
    logger.error('run '+run_num + ' failed aligning and cutting')
    logger.exception(e)

del ys_ext
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

######################## SF organization commented out for pol ########################
print('skipped SF organization')

# In[13]
start = time.time()
added_pulses = np.zeros((len(ys_cut), len(ys_cut[0][0])), dtype=np.float64)  ## channels, 8992 data points each

def add_pulse_noSF(ys):
    pulses_sum  = np.zeros((len(ys[0])), dtype=np.float64)
    for p in range(0, len(ys)):
        pulses_sum = np.add(pulses_sum,ys[p])
    return pulses_sum

for i in range(len(ys_cut)):
    added_pulses[i] = add_pulse_noSF(ys_cut[i])

end = time.time()
# print('summing pulses time: ' + str(end-start))  

# In[28]:

puck_thresh = 2500

if np.mean(He_Norm_arr[2])==0: ## catches runs with no ch. 29. 0/1 for True/False key
    puck_in = -1  
    puck_in_pulses = ['No ch.29 or no puck information'] 
else:
    puck_in_pulses = np.where(He_Norm_arr[2]>=puck_thresh) ## empty array for runs w/out puck
    if len(puck_in_pulses) == 0: ## puck_in state is false for empty array
        puck_in = 0
    else:
        puck_in = 1 ## if any pulse crosses threshold.

# In[29]:
# save summed pulses to be analyzed for polarization
start = time.time()

with h5py.File(polSavename+'.h5', 'w') as hdf5_file:
    hdf5_file.create_dataset('xs ', data=xs_cut[0]) ## all xs are the same, even though they are per channel...
    hdf5_file.attrs['puck_state'] = puck_in ## adding the puck information
    hdf5_file.attrs['puck_pulses'] = puck_in_pulses ## will be empty if puck is not in (usually)
    hdf5_file.attrs['num_pulses'] = numRuns ## some runs do not have 5000 pulses! Usually the ones at the end
    for i in range(0,len(ys_cut)):
        Ch_grp = hdf5_file.create_group('ch_'+str(np.char.zfill(str(chan_enab[i]), 2)))
        added_pulses_subgrp = Ch_grp.create_group('added_pulses')
        added_pulses_subgrp.create_dataset('ch_'+str(np.char.zfill(str(chan_enab[i]), 2)), data=added_pulses[i])  
        
end = time.time()
# print('saving hdf5: ' + str(end-start))    

# In[31]:

fullend = time.time()
print('full processing time: ' + str(fullend-fullstart))  
print('finished ' + str(datetime.now())) 
print('\n')

# ## end of data processing ##
