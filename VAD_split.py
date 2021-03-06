#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# File: VAD_split.py
# Date: Sun August 12 11:45:46 2018 +0800
# Author: Ricardo Cancar <ricardocancar@gmail.com>

import subprocess ## replace os 
#import matplotlib.pyplot as plt
#from matplotlib import style
import numpy as np
import scipy.io.wavfile
import datetime
import argparse
import time
from math import ceil, floor
from scipy.fftpack import dct
from pathlib import Path

#style.use('ggplot')

def get_args():
    desc = "Split audio.wav when nobody is speaking"
    epilog = """
             this file is make automatic audio samples split.

             Examples:
             python VAD_plit.py -f Path/to/audio.wav 
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-f', '--file_name',
                       help='name of audio you want to split',
                       required=True)
   
    parser.add_argument('-o', '--output',
                       help='path to output example /path/to/output/',
                       required=True)

   
    ret = parser.parse_args()
    return ret

class VAD():

    def __init__(self,file_name , NFFT = 512, low_ban = 300, hight_ban = 3000):
         self._read_wav(file_name)
         self.NFFT = NFFT
         self.low_ban = low_ban
         self.hight_ban = hight_ban
         

    def _read_wav(self,file_name):
        self.rate, self.signal = scipy.io.wavfile.read(file_name)
        return self
    
    def pre_proccessing(self, pre_emphasis = 0.97, frame_size=0.02, frame_stride=0.01):
        emphasized_signal = np.append(self.signal[0], self.signal[1:] - pre_emphasis * self.signal[:-1])
        frame_length, frame_step = frame_size * self.rate, frame_stride * self.rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal) 
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step)\
        , (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        return frames

    def power_spect(self):
        frames = self.pre_proccessing()
        mag_frames = np.absolute(np.fft.rfft(frames, self.NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        return pow_frames

    def mel_filter(self, nfilt = 40):
        pow_frames = self.power_spect()
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.NFFT + 1) * hz_points / self.rate)

        fbank = np.zeros((nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
           f_m_minus = int(bin[m - 1])   # left
           f_m = int(bin[m])             # center
           f_m_plus = int(bin[m + 1])    # right

           for k in range(f_m_minus, f_m):
              fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
           for k in range(f_m, f_m_plus):
              fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        return hz_points ,filter_banks

    def voice_frecuency(self):
        frec_wanted = []
        hz_points, filter_banks = self.mel_filter()
        for i in range(len(hz_points)-2):
           if  hz_points[i]<= self.hight_ban and hz_points[i] >=self.low_ban:
              frec_wanted.append(1)
           else:
              frec_wanted.append(0)
        
        sum_voice_energy = np.dot(filter_banks, frec_wanted)/1e+6  ## 1e+6 is use to reduce the signal amplitud 
        return(sum_voice_energy)

    def plotea(self,data):
        time = [i/100 for i in range(0, len(data))]
        plt.plot(time,data,color = 'b')
        plt.xlabel('time (s)')
        plt.ylabel('amp')
        plt.show()

    def write_wav(self,start, end, index,output):
        my_file = Path("./{}".format(output))#"/home/rcancar/Documents/TFM/codigo/predict")
        if not my_file.is_dir():
          bashCommand = "mkdir ./predict".format(output) #/home/rcancar/Documents/TFM/codigo/predict"
          process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
          out, error = process.communicate()
        st =time.time()
        for i in range(len(start)):
            signal = self.signal[int(start[i] * self.rate):int(end[i] * self.rate)]  # Keep the first 3.5 seconds
            scipy.io.wavfile.write("./{}/out{}.wav".format(output,index[i]), self.rate,signal)
        end = time.time()
        print(end - st)
            
################################################################################333
def get_cut_points(aux):
   cont = 0
   silence = False
   aux_s = [0]
   start =[]
   end = []
   index_start = []
   aux_index = [0]
   for i in range(len(aux)):
       if aux[i]  < 15:
           cont+=1
           if cont == 55:  ##frames minimun frames between pauses in speech
              silence = True
              #print(start[-1])
              if ((i-54)/100)  - aux_s[-1] > 1.5: # longitud de audio minima permitida en segundos 
                  start.append(aux_s[-1])
                  index_start.append(aux_index[-1])
                  index_start[-1] = str(datetime.timedelta(seconds=int(index_start[-1]/100)))
                  end.append(((i-54)/100))
                  #print(i-99,aux[i-99])
       if aux[i] > 15:
           cont =0
           if silence:
              silence = False
              aux_s.append( ((i-1)/100) )
              aux_index.append(i-1) # return the time start of out audio file 
   #print(index_start)      
   if len(aux_s) > len(start):
       start.append(aux_s[-1])
       index_start.append(aux_index[-1])    
       index_start[-1] = str(datetime.timedelta(seconds=int(index_start[-1]/100)))      
       end.append(len(aux)/100)
   return start, end, index_start



if __name__ == '__main__':
   args = get_args()
   a = VAD(args.file_name)
   voice_energy = a.voice_frecuency()
   #a.plotea(voice_energy) # to see the signal voice wave....
   start, end , index = get_cut_points(voice_energy)
   #print(start,end)
   a.write_wav(start, end, index, args.output)
   
