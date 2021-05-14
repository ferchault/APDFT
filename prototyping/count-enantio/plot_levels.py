import matplotlib.pyplot as plt
import numpy as np

#-----------------------------COT-----------------------------------------------
NNCNNCCC_actual = -739.3292463171272
CCNCCNNN_actual = -739.3284616392402

energies = [-739.3200703431687, -739.3196289410411, -739.32248609872, -739.3223047406021, -738.0412427380556, -738.0412452979488, -738.0412478578419, -739.3223098604061,-739.3221285029887, -739.3192713375028, -739.3188297713626]
text_pos = [(-4,-6),(-5,3),(-5,-6),(7,3),(-15,3),(-15,-7),(-13,3),(-17,-6),(-4,-6),(-22,3),(-22,3)]
N = len(energies)
delta_energies = []
for i in range(int((N-1)/2)):
    delta_energies.append(f'{energies[i]-energies[i+1]:+.3f}')
delta_energies.append(f'{energies[int((N-1)/2)]:.3f}')
for i in range(int((N-1)/2),N-1):
    delta_energies.append(f'{energies[i+1]-energies[i]:+.3f}')
orders = [r'$E^'+str(abs(i))+'$' for i in range(-int((N-1)*0.5),int((N+1)*0.5))]
fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2)
fig.subplots_adjust(hspace=0.07)
#----Set the lines----
ax1.scatter(range(0,N), energies, marker='')
for i in range(N):
    ax1.annotate(str(energies[i])[:8], (i, energies[i]), textcoords='offset points', xytext=text_pos[i], fontsize='xx-small')
ax1.errorbar(range(0,N), energies, xerr=0.45, fmt='--k', color='#000000')
ax2.scatter(range(0,N), energies, marker='')
for i in range(N):
    ax2.annotate(str(energies[i])[:8], (i, energies[i]), textcoords='offset points', xytext=text_pos[i], fontsize='xx-small')
ax2.errorbar(range(0,N), energies, xerr=0.45, fmt='--k', color='#000000')
#---ticks and axis labels----
fig.text(0.02, 0.5, 'Energy [Ha]', va='center', rotation='vertical')
fig.text(0.47, 0.98, r'$\Delta E^n$ [Ha]', va='center')
low_min =  min(min(energies[:int((N-3)/2)], energies[:-int((N-3)/2)]))
low_max = max(energies[:int((N-3)/2)])
low_min -= abs(low_min)*0.00001
low_max += abs(low_max)*0.000003
d = abs(low_max - low_min)
high_min = energies[int((N-1)/2)]
#print(high_min)
high_min -= abs(high_min)*0.000005
high_max = high_min + d

ax2.set_ylim([low_min, low_max])
ax1.set_ylim([high_min, high_max])

ax1.xaxis.tick_top()
ax1.set_xticks(range(0,N))
ax1.set_xticklabels(delta_energies, fontsize='xx-small', rotation=-45, va="bottom")
ax1.ticklabel_format(style='plain', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
ax2.set_xticks(range(0,N))
ax2.set_xticklabels(orders)
ax2.ticklabel_format(style='plain', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.set_yticklabels(np.arange(high_low,high_high,0.05))
#ax2.set_yticklabels(np.arange(low_low,low_high,0.05))
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
ax1.spines['bottom'].set_linestyle("dotted")
ax2.spines['top'].set_linestyle("dotted")
ax1.tick_params(left=True, bottom=False)
ax2.tick_params(left=True)
#----Draw baselines----
ax2.axhline(y = NNCNNCCC_actual, color = '#000000', linestyle = '--')
ax2.annotate('Actual value NNCNNCCC: '+str(NNCNNCCC_actual)[:11], (0,NNCNNCCC_actual), textcoords='offset points', xytext=(0,2), fontsize='xx-small')
ax2.axhline(y = CCNCCNNN_actual, color = '#000000', linestyle = '--')
ax2.annotate('Actual value CCNCCNNN: '+str(CCNCCNNN_actual)[:11], (0,CCNCCNNN_actual), textcoords='offset points', xytext=(0,2), fontsize='xx-small')
ax2.set_xlabel(r'Order $n$')
#ax2.set_ylabel('Energy [Ha]')
fig.savefig("energy_levels/energy_levels_COT.png", dpi=300)

#-----------------------------Benzene-------------------------------------------
NBBNCC_actual = -437.00076572264766
BNNBCC_actual = -437.02662368217403

energies = [-437.0076927936165, -437.020459387003, -436.8683962488271, -436.8767459183802, -434.35308694265393, -434.352916409248, -434.35274587584274, -436.87640485155026, -436.88475452086084, -437.0368176615888, -437.04958431107826]
text_pos = [(-10,4),(-10,-10),(-10,2),(-10,-6),(-24,3),(-15,-7),(-13,3),(-22,-7),(-13,4),(-29,-6),(-22,-6)]
N = len(energies)
delta_energies = []
for i in range(int((N-1)/2)):
    delta_energies.append(f'{energies[i]-energies[i+1]:+.3f}')
delta_energies.append(f'{energies[int((N-1)/2)]:.3f}')
for i in range(int((N-1)/2),N-1):
    delta_energies.append(f'{energies[i+1]-energies[i]:+.3f}')
orders = [r'$E^'+str(abs(i))+'$' for i in range(-int((N-1)*0.5),int((N+1)*0.5))]
fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2)
fig.subplots_adjust(hspace=0.07)
#----Set the lines----
ax1.scatter(range(0,N), energies, marker='')
for i in range(N):
    ax1.annotate(str(energies[i])[:8], (i, energies[i]), textcoords='offset points', xytext=text_pos[i], fontsize='xx-small')
ax1.errorbar(range(0,N), energies, xerr=0.45, fmt='--k', color='#000000')
ax2.scatter(range(0,N), energies, marker='')
for i in range(N):
    ax2.annotate(str(energies[i])[:8], (i, energies[i]), textcoords='offset points', xytext=text_pos[i], fontsize='xx-small')
ax2.errorbar(range(0,N), energies, xerr=0.45, fmt='--k', color='#000000')
#---ticks and axis labels----
fig.text(0.03, 0.5, 'Energy [Ha]', va='center', rotation='vertical')
fig.text(0.47, 0.98, r'$\Delta E^n$ [Ha]', va='center')
low_min =  min(min(energies[:int((N-3)/2)], energies[:-int((N-3)/2)]))
low_max = max(energies[:int((N-3)/2)])
low_min -= abs(low_min)*0.0002
low_max += abs(low_max)*0.0002
d = abs(low_max - low_min)
high_min = energies[int((N-1)/2)]
#print(high_min)
high_min -= abs(high_min)*0.0002
high_max = high_min + d

ax2.set_ylim([low_min, low_max])
ax1.set_ylim([high_min, high_max])

ax1.xaxis.tick_top()
ax1.set_xticks(range(0,N))
ax1.set_xticklabels(delta_energies, fontsize='xx-small', rotation=-45, va="bottom")
ax1.ticklabel_format(style='plain', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
ax2.set_xticks(range(0,N))
ax2.set_xticklabels(orders)
ax2.ticklabel_format(style='plain', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.set_yticklabels(np.arange(high_low,high_high,0.05))
#ax2.set_yticklabels(np.arange(low_low,low_high,0.05))
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
ax1.spines['bottom'].set_linestyle("dotted")
ax2.spines['top'].set_linestyle("dotted")
ax1.tick_params(left=True, bottom=False)
ax2.tick_params(left=True)
#----Draw baselines----
ax2.axhline(y = NBBNCC_actual, color = '#000000', linestyle = '--')
ax2.annotate('Actual value NBBNCC: '+str(NBBNCC_actual)[:11], (0,NBBNCC_actual), textcoords='offset points', xytext=(120,2), fontsize='xx-small')
ax2.axhline(y = BNNBCC_actual, color = '#000000', linestyle = '--')
ax2.annotate('Actual value BNNBCC: '+str(BNNBCC_actual)[:11], (0,BNNBCC_actual), textcoords='offset points', xytext=(120,2), fontsize='xx-small')
ax2.set_xlabel(r'Order $n$')
#ax2.set_ylabel('Energy [Ha]')
fig.savefig("energy_levels/energy_levels_benzene.png", dpi=300)


#-----------------------------Benzene_geomenforced------------------------------
NBBNCC_forced_actual = -435.87721373221757
BNNBCC_forced_actual = -435.9038223625185

energies = [-435.8847603778811, -435.88442255408, -435.8976108093157, -435.7447725850432, -435.75333130160647, -433.2225764838317, -433.2225790071666, -433.2225815305072, -435.7533363466079, -435.7618949855427, -435.9147359307182, -435.9280476827084, -435.92754913163907]
text_pos = [(-10,4),(-10,4),(-10,-10),(-10,2),(-10,-6),(-24,3),(-15,-7),(-13,3),(-22,-7),(-13,4),(-29,-6),(-22,-6),(-22,-6)]
N = len(energies)
delta_energies = []
for i in range(int((N-1)/2)):
    delta_energies.append(f'{energies[i]-energies[i+1]:+.3f}')
delta_energies.append(f'{energies[int((N-1)/2)]:.3f}')
for i in range(int((N-1)/2),N-1):
    delta_energies.append(f'{energies[i+1]-energies[i]:+.3f}')
orders = [r'$E^'+str(abs(i))+'$' for i in range(-int((N-1)*0.5),int((N+1)*0.5))]
fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2)
fig.subplots_adjust(hspace=0.07)
#----Set the lines----
ax1.scatter(range(0,N), energies, marker='')
for i in range(N):
    ax1.annotate(str(energies[i])[:8], (i, energies[i]), textcoords='offset points', xytext=text_pos[i], fontsize='xx-small')
ax1.errorbar(range(0,N), energies, xerr=0.45, fmt='--k', color='#000000')
ax2.scatter(range(0,N), energies, marker='')
for i in range(N):
    ax2.annotate(str(energies[i])[:8], (i, energies[i]), textcoords='offset points', xytext=text_pos[i], fontsize='xx-small')
ax2.errorbar(range(0,N), energies, xerr=0.45, fmt='--k', color='#000000')
#---ticks and axis labels----
fig.text(0.03, 0.5, 'Energy [Ha]', va='center', rotation='vertical')
fig.text(0.47, 0.98, r'$\Delta E^n$ [Ha]', va='center')
low_min =  min(min(energies[:int((N-3)/2)], energies[:-int((N-3)/2)]))
low_max = max(energies[:int((N-3)/2)])
low_min -= abs(low_min)*0.0002
low_max += abs(low_max)*0.0002
d = abs(low_max - low_min)
high_min = energies[int((N-1)/2)]
#print(high_min)
high_min -= abs(high_min)*0.0002
high_max = high_min + d

ax2.set_ylim([low_min, low_max])
ax1.set_ylim([high_min, high_max])

ax1.xaxis.tick_top()
ax1.set_xticks(range(0,N))
ax1.set_xticklabels(delta_energies, fontsize='xx-small', rotation=-45, va="bottom")
ax1.ticklabel_format(style='plain', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
ax2.set_xticks(range(0,N))
ax2.set_xticklabels(orders)
ax2.ticklabel_format(style='plain', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.set_yticklabels(np.arange(high_low,high_high,0.05))
#ax2.set_yticklabels(np.arange(low_low,low_high,0.05))
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize('xx-small')
ax1.spines['bottom'].set_linestyle("dotted")
ax2.spines['top'].set_linestyle("dotted")
ax1.tick_params(left=True, bottom=False)
ax2.tick_params(left=True)
#----Draw baselines----
ax2.axhline(y = NBBNCC_forced_actual, color = '#000000', linestyle = '--')
ax2.annotate('Actual value NBBNCC: '+str(NBBNCC_forced_actual)[:8], (0,NBBNCC_forced_actual), textcoords='offset points', xytext=(120,2), fontsize='xx-small')
ax2.axhline(y = BNNBCC_forced_actual, color = '#000000', linestyle = '--')
ax2.annotate('Actual value BNNBCC: '+str(BNNBCC_forced_actual)[:8], (0,BNNBCC_forced_actual), textcoords='offset points', xytext=(120,2), fontsize='xx-small')
ax2.set_xlabel(r'Order $n$')
#ax2.set_ylabel('Energy [Ha]')
fig.savefig("energy_levels/energy_levels_benzene_forced.png", dpi=300)
