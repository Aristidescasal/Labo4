# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:04:27 2024

@author: Publico
"""

# NI-DAQmx Python Documentation: https://nidaqmx-python.readthedocs.io/en/latest/index.html
# NI USB-621x User Manual: https://www.ni.com/pdf/manuals/371931f.pdf
import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time
import pyvisa as visa
#%%
#Cosmética para que MatPlotLib use la fuente de LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size']=15

#%%
#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)

rm= visa.ResourceManager()

print(rm.list_resources())

#multi = visa.ResourceManager().open_resource('GPIB0::22::INSTR')

#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai1_channel = task.ai_channels.add_ai_voltage_chan("Dev2/ai1",max_val=10,min_val=-10)
    ai2_channel = task.ai_channels.add_ai_voltage_chan("Dev2/ai2",max_val=10,min_val=-10)
    ai4_channel = task.ai_channels.add_ai_voltage_chan("Dev2/ai4",max_val=10,min_val=-10)
    #v1 = 
    # print(ai1_channel.ai_term_cfg, ai2_channel.ai_term_cfg)    
    # print(ai1_channel.ai_max, ai2_channel.ai_max)
    # print(ai1_channel.ai_min, ai2_channel.ai_min)	
	
#%%
## Medicion por tiempo/samples de una sola vez
def medicion_una_vez1(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev2/ai2", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev2/ai4", terminal_config = modo,max_val=10,min_val=-10)
        
               
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        
        datos1 = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE, timeout=duracion+0.1)           
    
    datos = np.asarray(datos1)
    temp = PT100_res2temp_interp(datos[2,:]/1e-3)
    return datos,temp

#%%
def PT100_res2temp_interp(R): #en Ohm
    data = np.loadtxt('C:/Users/publico/Desktop/Grupo Miércoles/Ferromagnetismo/Pt100_resistencia_temperatura.csv',delimiter=',') 
    temperature_vals = data[:,0] # en Celsius
    resistance_vals = data[:,1] #en Ohm
    return np.interp(R, resistance_vals, temperature_vals)




#%%
duracion = 1/25 * 25 *60 *3
fs = 5000 #Frecuencia de muestreo   
[y1, y2, vr], temp = medicion_una_vez1(duracion, fs)
t1 = np.arange(len(y1))/fs
np.savez('datos_nitrogeno_sin_integradoractivo_1.npz', tiempo = t1 , tensiones = [y1, y2], R = vr, temp=temp)
I = 0.001 #mA

#%%
import numpy as np
placa = np.load('C:/Users/Compumar/Downloads/datos_nitrogeno_final_3.npz',allow_pickle=True)

Vin, Vout = placa['tensiones']
R = placa['R']
t = placa['tiempo']
temp = placa['temp']
eT = abs((0.5*temp)/100)
evin, evout = (3*placa['tensiones'])/100
from scipy.signal import savgol_filter as savgol


temp_suave = savgol(temp,151,3)


plt.figure(1)
plt.plot(t, temp,'.', label = r'Datos')
plt.plot(t, temp_suave, '-', label = r'Datos filtrados')
plt.xlabel(r'$Tiempo \ (s)$')
plt.ylabel(r'$Temperatura \ (^{\circ}C)$')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(2)
plt.plot(t[0:150], Vin[0:150], label = r'$V_{in}$')
plt.plot(t[0:150], Vout[0:150], label = r'$V_{out}$')
plt.xlabel(r'$Tiempo \ (s)$')
plt.ylabel(r'$Tensión \ (V)$')
plt.gca().axhline(0,ls='--',color='gray')
plt.gca().axvline(0.011466,ls='--',color='gray')
plt.gca().axvline(0.01365,ls='--',color='gray')
plt.fill_between([0.011466, 0.01365], np.min(Vin), np.max(Vin), color ='red', alpha = 0.2, label = r'$Desfasaje$')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(3)
plt.plot(t[0:150], Vin[0:150], label = r'$V_{in}$')
plt.plot(t[0:150], Vout[10:160], label = r'$V_{out}$')
plt.xlabel(r'$Tiempo \ (s)$')
plt.ylabel(r'$Tensión \ (V)$')
plt.gca().axhline(0,ls='--',color='gray')
plt.legend()
plt.grid(True)
plt.show()
#%%
for jj in range(0,900000, 2*90000):
    plt.figure(4)
    r=200
    plt.plot(Vin[jj+1:jj+r]-((np.max(Vin[jj+1:jj+r])+np.min(Vin[jj+1:jj+r]))/2),Vout[jj+0:jj+r-1]-((np.max(Vout[jj+0:jj+r-1])+np.min(Vout[jj+0:jj+r-1]))/2),'.-', label = r"$T \simeq {} K$".format(int(temp_suave[jj]+273.15)))
    plt.gca().axhline(0,ls='--',color='gray')
    plt.gca().axvline(0,ls='--',color='gray')
    plt.xlabel(r'$V_{in} \ (V)$')
    plt.ylabel(r'$V_{out} \ (V)$')
    plt.legend()
#%%


j_vec = []
Vm = []

for jj in range(0,899999):
    plt.figure(4)
    r=151
    #plt.plot(Vin[jj+1:jj+r]-((np.max(Vin[jj+1:jj+r])+np.min(Vin[jj+1:jj+r]))/2),Vout[jj+0:jj+r-1]-((np.max(Vout[jj+0:jj+r-1])+np.min(Vout[jj+0:jj+r-1]))/2),'.-')
    #plt.gca().axhline(0,ls='--',color='gray')
    #plt.gca().axvline(0,ls='--',color='gray')
    if Vin[jj]*Vin[jj+1]<0:
        #if Vout[jj] > 0:  
            #print(Vin[jj])
            #print(np.array(Vout[jj]))
            #print(jj)
            datos = np.array(Vout[jj])
            Vm.extend(Vm+datos)
            j_vec.append(jj)
V_m = Vout[j_vec]-((np.max(Vout[j_vec])+np.min(Vout[j_vec]))/2)                   
plt.plot(temp_suave[j_vec], V_m, '.')
plt.xlabel(r'$Temperatura \ (^{\circ} C)$')
plt.ylabel(r'$Tensión \ (V)$')   
plt.gca().axhline(0,ls='--',color='gray') 
plt.grid(True)
print((V_m>0))
#%%
n_vec = []
for nn in range(-len(V_m), len(V_m)):
        if V_m[nn] > 0:
            n_vec.append(nn)
#V_M = Vout[n_vec]-((np.max(Vout[n_vec])+np.min(Vout[n_vec]))/2)         
plt.plot(temp_suave[n_vec], Vout[n_vec]-((np.max(Vout[n_vec])+np.min(Vout[n_vec]))/2), '.')
plt.xlabel(r'$Temperatura \ (^{\circ} C)$')
plt.ylabel(r'$Tensión \ (V)$')   
plt.gca().axhline(0,ls='--',color='gray') 
plt.grid(True)
  
#%%
for jj in range(0,899999, 89999):
    plt.figure(4)
    r=151
    plt.plot(Vin[jj+1:jj+r]-((np.max(Vin[jj+1:jj+r])+np.min(Vin[jj+1:jj+r]))/2),Vout[jj+0:jj+r-1]-((np.max(Vout[jj+0:jj+r-1])+np.min(Vout[jj+0:jj+r-1]))/2),'.-')
    plt.gca().axhline(0,ls='--',color='gray')
    plt.gca().axvline(0,ls='--',color='gray')      
#%%



# Ordenar de menor a mayor
V_M = np.sort(V_m)[::-1]
T = np.sort(temp_suave[j_vec])
print("Arreglo ordenado de menor a mayor:", V_M, T)

plt.plot(T, V_M, '.')
#%%

#%%
nv = []

print(len(V_m))
for j in range(0, len(V_m)):    
    if 0.012<V_m[j]:
          nv.append(j)
plt.plot(T[nv], V_m[nv], '.')     
plt.xlabel(r'$Temperatura \ (^{\circ} C)$')
plt.ylabel(r'$Tensión \ (V)$')   
plt.gca().axhline(0,ls='--',color='gray') 
plt.grid(True)  
   
#%%
#Ajuste
V_M = np.sort(V_m[nv])[::-1]
plt.hist(V_M)
plt.hist(V_m[nv])
#%%
from scipy.optimize import curve_fit
x_datos = T[nv]
y_datos = V_m[nv]
Verr = (3* V_m[nv])/100
def modelo(x, b):
    return 0.0238 * ((b - x)**(0.3))

# Parámetros iniciales con los que vamos a iniciar el proceso de fiteo
parametros_iniciales=[np.max(T[nv])]

# Hacemos el ajuste con curve_fit
popt, pcov = curve_fit(modelo, x_datos, y_datos, p0=parametros_iniciales, sigma=Verr)

# curve_fit devuelve dos resultados. El primero (popt) son los
# parámetros óptimos hallados. El segundo (pcov) es la matriz de
# covarianza de los parámetros hallados.

x_modelo  = np.linspace(np.min(T[nv]), np.max(T[nv]), 10000)

plt.figure()
plt.errorbar( x_datos,                 y_datos, yerr = Verr,  fmt = '.', ecolor = 'cyan', label='datos')
plt.plot(x_modelo, modelo(x_modelo, *popt), 'r-', label=r'$M_r \propto (T_c - T)^{\beta}$')
plt.xlabel(r'$Temperaura \ (^{\circ} C)$')
plt.ylabel(r'$V_{out} \ (V)$')
plt.gca().axvline(-17.007,ls='--',color='gray', label = r'$T_c = (-17.007 \pm 0.002)^{\circ}C$') 
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# De la matris de covarinza podemos obtener los valores de desviacion estandar
# de los parametros hallados
pstd = np.sqrt(np.diag(pcov))
nombres_de_param=['a','b','c', 'd']
print('Parámetros hallados:')
for i,param in enumerate(popt):
    print('{:s} = {:5.3f} ± {:5.3f}'.format( nombres_de_param[i] , param , pstd[i]) )
#
#%%
r = y_datos - modelo(x_datos, *popt)
plt.plot(r)
print('Coeficiente de determinacion R2')

# Suma de los cuadrados de los residuos
ss_res = np.sum( (y_datos - modelo(x_datos, *popt))**2  )

# Suma total de cuadrados
ss_tot = np.sum( (y_datos - np.mean(y_datos) )**2  )

R2     = 1 - (ss_res / ss_tot)

print('R2 = {:10.8f}'.format(R2) )
#%%
from scipy.optimize import least_squares

def modelo(p,x):
    # p es un vector con los parámetros
    # x es el vector de datos x
    return p[0] * ((p[1] - x)**(p[2]))

param_list = ['a', 'b', 'c']

def residuos(p, x, y):
    # p es un vector con los parámetros
    # x es el vector de datos x
    # y es el vector de datos y
    y_modelo = modelo(p, x)
    plt.clf()
    plt.plot(x,y,'o',x,y_modelo,'r-')
    plt.pause(0.05)
    param_list.append(p)
    return y_modelo - y

parametros_iniciales=[0.03, np.max(T[nv]), 0.25]  # Ajusta
res = least_squares(residuos, parametros_iniciales, args=(x_datos, y_datos), verbose=1)

# Estos son los parámetros hallados:
print('parámetros hallados')
print(res.x)

# Calculamos la matriz de covarianza "pcov"
def calcular_cov(res,y_datos):
    U, S, V = np.linalg.svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * S[0]
    S = S[S > threshold]
    V = V[:S.size]
    pcov = np.dot(V.T / S**2, V)

    s_sq = 2 * res.cost / (y_datos.size - res.x.size)
    pcov = pcov * s_sq
    return pcov

pcov = calcular_cov(res,y_datos)

# De la matriz de covarinza podemos obtener los valores de desviación estándar
# de los parametros hallados
pstd = np.sqrt(np.diag(pcov))

print('Parámetros hallados (con incertezas):')
for i,param in enumerate(res.x):
    print('parametro[{:d}]: {:5.3f} ± {:5.3f}'.format(i,param,pstd[i]/2))

y_modelo = modelo(res.x, x_datos)

plt.figure()
plt.plot(x_datos, y_datos ,  'o', markersize=4, label='datos')
plt.plot(x_datos, y_modelo, 'r-',               label='modelo fiteado')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='best')
plt.tight_layout()

#%%
M = savgol(V_m[nv],151,5)
plt.plot(T[nv], V_m[nv],'.')
plt.plot(T[nv], M, '.')

#%%
## Medicion continua
def medicion_continua(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2", terminal_config = modo,max_val=10,min_val=-10)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4", terminal_config = modo,max_val=10,min_val=-10)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        data =[]
        while total<cant_puntos:
            time.sleep(0.1)
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
            data.extend(datos)
            total = total + len(datos)
            t1 = time.time()
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)))            
        return data
fs1 = 250 #Frecuencia de muestreo
duracion1 = 0.5 #segundos
vin, vout, vr = medicion_continua(duracion1, fs1)


#%%
## Modo conteo de flancos 
# Obtengo el nombre de la primera placa y el primer canal de conteo (ci)
cDaq = system.devices[0].name
ci_chan1 = system.devices[0].ci_physical_chans[0].name
print(cDaq)
print(ci_chan1 )

# Pinout: 
# NiUSB6212 
# gnd: 5 or 37
# src: 33


def daq_conteo(duracion):
    with nidaqmx.Task() as task:

        # Configuro la task para edge counting
        task.ci_channels.add_ci_count_edges_chan(counter=ci_chan1,
            name_to_assign_to_channel="",
            edge=nidaqmx.constants.Edge.RISING,
            initial_count=0)
        
        # arranco la task
        task.start()
        counts = [0]
        t0 = time.time()
        try:
            while time.time() - t0 < duracion:
                count = task.ci_channels[0].ci_count
                print(f"{time.time()-t0:.2f}s {count-counts[-1]} {count}")
                counts.append(count)
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            pass
        
        finally:
            task.stop()
            
    return counts  

duracion = 1 # segundos
y = daq_conteo(duracion)
t = np.arange(len(y))/fs
plt.plot(t, y)
plt.grid()
plt.show()