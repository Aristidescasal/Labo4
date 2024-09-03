# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:02:05 2024

@author: Compumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter as savgol

#%%

#Cosmética para que MatPlotLib use la fuente de LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size']=15

#%%

placa = np.load('C:/Users/Compumar/Documents/Facultad/Lic. Física/labo 4/ferromagnetismo/datos_nitrogeno1.npz',allow_pickle=True)

Vin, Vout = placa['tensiones']
R = placa['R']
t = placa['tiempo']
temp = placa['temp']


print(len(Vin))

temp_suave = savgol(temp,151,3)

#%%
plt.figure(1)
plt.plot(Vin, Vout)
plt.show()
plt.figure(2)
plt.plot(t, Vin)
plt.plot(t, Vout)
plt.show()
plt.figure(3)
plt.plot(t, temp)
plt.plot(t, temp_suave)
plt.show()
#%%

for jj in range(0,900000,90000):
    #plt.figure(jj)
    plt.plot(Vin[jj+0:jj+200],Vout[jj+0:jj+200],'.')
    plt.gca().axhline(0,ls='--',color='gray')
    plt.gca().axvline(0,ls='--',color='gray')
    plt.xlabel(r'$V_{in}$ $(V)$')
    plt.ylabel(r'$V_{out}$ $(V)$')
    #plt.show()
#%%
for j in range(0, 20000):    
    if Vin[j]*Vin[j+1]<0:
          print(Vin[j])
          print(Vout[j])
          print(j)
#%%

plt.plot(Vin[0:200],Vout[0:200])
#plt.plot(Vin[0:200],'.-')
#plt.plot(Vout[0:200],'.-')
plt.gca().axhline(0,ls='--',color='gray')