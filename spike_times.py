import matplotlib.pyplot as plt
import random as rnd
import numpy as np
import math

def integrate_fire_single(time, step, Tm, El, Vr, Vt, Rm, Ie):
    iterations = int(time/step)
    voltage = np.zeros(iterations + 1) 
    voltage[0] = Vr
    for i in range(iterations):
        voltage[i+1] = voltage[i] + step * ((El - voltage[i] + Rm * Ie)/Tm)
        if voltage[i+1] > Vt:
            voltage[i+1] = Vr
    return voltage

def integrate_fire_pair(time, step, Tm, El, Vr, Vt, Rm, Ie, Ts, Gs, Es):
    iterations = int(time/step)
    voltage = np.zeros((2, iterations + 1))
    voltage[0,0] = rnd.uniform(Vr, Vt)
    voltage[1,0] = rnd.uniform(Vr, Vt)
    activity = np.zeros(2)
    for i in range(iterations): 
        for j in range(2):
            voltage[j,i+1] = voltage[j,i] + step * ((El - voltage[j,i] + Rm * Ie + Rm * Gs * activity[1-j] * (Es - voltage[j,i]))/Tm)
            if voltage[j,i+1] > Vt:
                voltage[j,i+1] = Vr
                activity[j] += 0.5
            activity[j] = activity[j] + step * (-activity[j]/Ts)
    return voltage

def integrate_fire_synapses(time, step, Tm, El, Vr, Vt, Rm, Ts, Gs, Es, Fr):
    iterations = int(time/step)
    voltage = np.zeros(iterations + 1)
    voltage[0] = Vr
    activity = np.zeros(40)
    conductance = np.full(40, Gs)
    for i in range(iterations):
        voltage[i+1] = voltage[i] + step * ((El - voltage[i] + Rm * np.dot(conductance, activity) * (Es - voltage[i]))/Tm)
        if voltage[i+1] > Vt:
            voltage[i+1] = Vr
        for j in range(len(activity)):
            if rnd.uniform(0,1) < (step * Fr):
                activity[j] += 0.5
            activity[j] = activity[j] + step * (-activity[j]/Ts)
    return voltage

def integrate_fire_synapses_stdp(stdp, time, step, Tm, El, Vr, Vt, Rm, Ts, Gs, Es, Fr, Ap, An, Tp, Tn):
    iterations = int(time/step)
    voltage = np.zeros(iterations + 1)
    voltage[0] = Vr
    activity = np.zeros(40)
    conductance = np.full(40, Gs)
    post_spike = None
    pre_spikes = np.full(40, None)
    for i in range(iterations):
        voltage[i+1] = voltage[i] + step * ((El - voltage[i] + Rm * np.dot(conductance, activity) * (Es - voltage[i]))/Tm)
        if voltage[i+1] > Vt:
            voltage[i+1] = Vr
            post_spike = i
        for j in range(len(activity)):
            if rnd.uniform(0,1) < (step * Fr):
                activity[j] += 0.5
                pre_spikes[j] = i
            activity[j] = activity[j] + step * (-activity[j]/Ts)
            if (stdp == True) and (post_spike is not None) and (pre_spikes[j] is not None) and (post_spike == i or pre_spikes[j] == i):
                spike_delta = (post_spike - pre_spikes[j]) * step
                if spike_delta > 0:
                    conductance[j] = conductance[j] + Ap * math.exp(-abs(spike_delta) / Tp)
                else:
                    conductance[j] = conductance[j] - An * math.exp(-abs(spike_delta) / Tn)
                conductance[j] = np.clip(conductance[j], 0, Gs)
    return voltage, conductance

def integrate_fire_synapses_stdp_correlated(stdp, time, step, Tm, El, Vr, Vt, Rm, Ts, Gs, Es, Fr, Ap, An, Tp, Tn, Bc):
    iterations = int(time/step)
    voltage = np.zeros(iterations + 1)
    voltage[0] = Vr
    activity = np.zeros(40)
    conductance = np.full(40, Gs)
    post_spike = None
    pre_spikes = np.full(40, None)
    for i in range(iterations):
        voltage[i+1] = voltage[i] + step * ((El - voltage[i] + Rm * np.dot(conductance, activity) * (Es - voltage[i]))/Tm)
        if voltage[i+1] > Vt:
            voltage[i+1] = Vr
            post_spike = i
        rate = Fr + Bc * math.sin(2 * math.pi * 10 * step * i)
        for j in range(len(activity)):
            if rnd.uniform(0,1) < (step * rate):
                activity[j] += 0.5
                pre_spikes[j] = i
            activity[j] = activity[j] + step * (-activity[j]/Ts)
            if (stdp == True) and (post_spike is not None) and (pre_spikes[j] is not None) and (post_spike == i or pre_spikes[j] == i):
                spike_delta = (post_spike - pre_spikes[j]) * step
                if spike_delta > 0:
                    conductance[j] = conductance[j] + Ap * math.exp(-abs(spike_delta) / Tp)
                else:
                    conductance[j] = conductance[j] - An * math.exp(-abs(spike_delta) / Tn)
                conductance[j] = np.clip(conductance[j], 0, Gs)
    return voltage, conductance

M = 1000000
m = 0.001
n = 0.000000001

# voltage = integrate_fire_single(1, 0.25*m, 10*m, -70*m, -70*m, -40*m, 10*M, 3.1*n)
# plt.plot(voltage)
# plt.title('Integrate And Fire Neuron - Single')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.xticks([0,800,1600,2400,3200,4000],[0,200,400,600,800,1000])
# plt.yticks([-0.07,-0.065,-0.06,-0.055,-0.05,-0.045,-0.04],[-70,-65,-60,-55,-50,-45,-40])
# plt.show()

# voltage = integrate_fire_pair(1, 0.25*m, 20*m, -70*m, -80*m, -54*m, 0.15, 0.12, 10*m, 1, 0)
# plt.plot(voltage[0], label = 'Nueron 1')
# plt.plot(voltage[1], label = 'Nueron 2')
# plt.title('Integrate And Fire Neuron - Excitatory Pair')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.xticks([0,800,1600,2400,3200,4000],[0,200,400,600,800,1000])
# plt.yticks([-0.08,-0.075,-0.07,-0.065,-0.06,-0.055],[-80,-75,-70,-65,-60,-55])
# plt.legend(loc = 3)
# plt.show()

# voltage = integrate_fire_pair(1, 0.25*m, 20*m, -70*m, -80*m, -54*m, 0.15, 0.12, 10*m, 1, -80*m)
# plt.plot(voltage[0], label = 'Nueron 1')
# plt.plot(voltage[1], label = 'Nueron 2')
# plt.title('Integrate And Fire Neuron - Inhibitory Pair')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.xticks([0,800,1600,2400,3200,4000],[0,200,400,600,800,1000])
# plt.yticks([-0.08,-0.075,-0.07,-0.065,-0.06,-0.055],[-80,-75,-70,-65,-60,-55])
# plt.legend(loc = 3)
# plt.show()

# voltage = integrate_fire_synapses(1, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, 15)
# plt.plot(voltage)
# plt.title('Integrate And Fire Neuron - 40 Pre-Synaptic Neurons')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.xticks([0,800,1600,2400,3200,4000],[0,200,400,600,800,1000])
# plt.yticks([-0.065,-0.06,-0.055,-0.05],[-65,-60,-55,-50])
# plt.show()

# voltage, conductance = integrate_fire_synapses_stdp(True, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, 15, 0.2*n, 0.25*n, 20*m, 20*m)
# plt.hist(conductance, bins = 10, range = [0.0,0.000000004])
# plt.title('STDP - Steady-State Synaptic Strengths')
# plt.xlabel('Synaptic Strength (nS)')
# plt.ylabel('Synpapses')
# plt.xticks([0,0.0000000004,0.0000000008,0.0000000012,0.0000000016,0.000000002,0.0000000024,0.0000000028,0.0000000032,0.0000000036,0.000000004],[0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0])
# plt.show()

# frequency = np.count_nonzero(np.asarray(np.split(voltage[1:], 30)) == -65*m, axis = 1)/10
# plt.plot(frequency)
# plt.title('STDP - Post-Synaptic Firing Rate')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.xticks([0,5,10,15,20,25,30],[0,50,100,150,200,250,300])
# plt.show()

# mean_conductances = []
# for i in range(5):
#     voltage, conductance = integrate_fire_synapses_stdp(True, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, 15, 0.2*n, 0.25*n, 20*m, 20*m)
#     mean_conductance = np.mean(conductance)
#     mean_conductances.append(mean_conductance)
#     print(mean_conductance)
# print('Average: ', np.mean(mean_conductances))

# steady_frequencies = []
# for i in range(5):
#     voltage, conductance = integrate_fire_synapses_stdp(True, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, 15, 0.2*n, 0.25*n, 20*m, 20*m)
#     frequency = np.count_nonzero(np.asarray(np.split(voltage[1:], 30)) == -65*m, axis = 1)/10
#     steady_frequency = np.count_nonzero(np.asarray(np.split(voltage[1:], 10)[-1]) == -65*m, axis = 0)/30
#     steady_frequencies.append(steady_frequency)
#     print(frequency)
#     print(steady_frequency)
# print('Average: ', np.mean(steady_frequencies))

# steady_frequencies = []
# for stdp in (True, False):
#     print('STDP: ', stdp)
#     for Fr in range(10, 22, 2):
#         print('Fr: ', Fr)
#         for i in range(5):
#             voltage, conductance = integrate_fire_synapses_stdp(stdp, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, Fr, 0.2*n, 0.25*n, 20*m, 20*m)
#             frequency = np.count_nonzero(np.asarray(np.split(voltage[1:], 30)) == -65*m, axis = 1)/10
#             steady_frequency = np.count_nonzero(np.asarray(np.split(voltage[1:], 10)[-1]) == -65*m, axis = 0)/30
#             steady_frequencies.append(steady_frequency)
#             print(frequency)
#             print(steady_frequency)
# steady_frequencies = np.asarray(np.split(np.asarray(np.split(np.asarray(steady_frequencies), 12)), 2, axis = 0))
# print(steady_frequencies)
# steady_frequencies = np.mean(steady_frequencies, axis = 2)
# print('Average: ', steady_frequencies)
# plt.plot(steady_frequencies[0], label = 'STPD = ON')
# plt.plot(steady_frequencies[1], label = 'STPD = OFF')
# plt.title('STDP - Pre-Synaptic VS Post-Synaptic Firing Rate')
# plt.xlabel('Pre-Synaptic Firing Rate (Hz)')
# plt.ylabel('Post-Synaptic Firing Rate (Hz)')
# plt.xticks([0,1,2,3,4,5],[10,12,14,16,18,20])
# plt.legend(loc = 2)
# plt.show()

# conductances = []
# for Fr in (10, 20):
#     voltage, conductance = integrate_fire_synapses_stdp(True, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, Fr, 0.2*n, 0.25*n, 20*m, 20*m)
#     conductances.append(conductance)
# plt.hist([conductances[0],conductances[1]], bins = 10, range = [0.0,0.000000004], label = ['F = 10Hz', 'F = 20Hz'])
# plt.title('STDP - Steady-State Synaptic Strengths')
# plt.xlabel('Synaptic Strength (nS)')
# plt.ylabel('Synpapses')
# plt.xticks([0,0.0000000004,0.0000000008,0.0000000012,0.0000000016,0.000000002,0.0000000024,0.0000000028,0.0000000032,0.0000000036,0.000000004],[0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0])
# plt.legend(loc = 9)
# plt.show()

# std_conductances = []
# mean_conductances = []
# for Bc in range(0, 25, 5):
#     voltage, conductance = integrate_fire_synapses_stdp_correlated(True, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, 20, 0.2*n, 0.25*n, 20*m, 20*m, Bc)
#     std_conductances.append(np.std(conductance))
#     mean_conductances.append(np.mean(conductance))
# plt.plot(std_conductances, label = "σ")
# plt.plot(mean_conductances, label = "μ")
# plt.title('STDP - Steady-State Synaptic Strengths With Correlation')
# plt.xlabel('Correlation Factor (Hz)')
# plt.ylabel('Synaptic Strength (nS)')
# plt.xticks([0.0,1.0,2.0,3.0,4.0],[0,5,10,15,20])
# plt.yticks([0.000000001,0.0000000012,0.0000000014,0.0000000016,0.0000000018],[1.0,1.2,1.4,1.6,1.8])
# plt.legend(loc = 9)
# plt.show()

# conductances = []
# for Bc in (0, 20):
#     voltage, conductance = integrate_fire_synapses_stdp_correlated(True, 300, 0.25*m, 10*m, -65*m, -65*m, -50*m, 100*M, 2*m, 4*n, 0, 20, 0.2*n, 0.25*n, 20*m, 20*m, Bc)
#     conductances.append(conductance)
# plt.hist([conductances[0],conductances[1]], bins = 10, range = [0.0,0.000000004], label = ['B = 0Hz', 'B = 20Hz'])
# plt.title('STDP - Steady-State Synaptic Strengths With Correlation')
# plt.xlabel('Synaptic Strength (nS)')
# plt.ylabel('Synpapses')
# plt.xticks([0,0.0000000004,0.0000000008,0.0000000012,0.0000000016,0.000000002,0.0000000024,0.0000000028,0.0000000032,0.0000000036,0.000000004],[0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0])
# plt.legend(loc = 9)
# plt.show()