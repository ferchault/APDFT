import matplotlib.pyplot as plt
import numpy as np

order = [1,2,3,4,5]

def2SVP = -1.8087083589896444
def2TZVP =-1.970466136518705
def2QZVP =-1.9941334002289228
ccpVDZ =-1.8088055688826548
ccpVTZ =-1.9701391491932403
ccpVQZ =-1.9968723869922795
ccpV5Z =-1.9993045906665468
ccpV6Z =-1.9997770788378368

ccpVDZ_orders = np.array([-1.4897007775938205, -1.357282924029936,-1.343093145150746,-1.3903924311533644, -1.3610918795181335])
ccpVDZ_orders = np.absolute(ccpVDZ_orders- ccpVDZ)
print(ccpVDZ_orders)


fig, ax= plt.subplots()

#Median energy differences
ax_Delta_E.plot(order, median_E_QM9, label=r'$| \Delta E |$ (QM9)', color='tab:blue', marker='+')
ax_Delta_E.plot(order, median_E_ZINC, label=r'$| \Delta E |$ (ZINC)', color='tab:red', marker='+')
ax_Delta_E.set_xlabel(r'order', fontsize=14)
ax_Delta_E.set_ylabel(r'$| \Delta E |$  [Ha]', fontsize=14)
#ax.set_ylim([0.0004,1])
ax.set_yscale('log')

ax_Delta_E.axhline(y = LDA_E, color = 'black', linestyle = 'solid', linewidth = 1, xmax = 0.32)
ax_Delta_E.text(0.38,LDA_E + 0.005,'LDA')
ax_Delta_E.axhline(y = GGA_E, color = 'black', linestyle = 'solid', linewidth = 1, xmax = 0.32)
ax_Delta_E.text(0.38,GGA_E+0.002,'PBE')
ax_Delta_E.axhline(y = hybrid_E, color = 'black', linestyle = 'solid', linewidth = 1, xmax = 0.32)
ax_Delta_E.text(0.38,hybrid_E-0.004,'TPSSh')

ax.legend(loc="upper right",framealpha=0, edgecolor='black')

fig.tight_layout()
fig.savefig("figures/basis_singleatoms.png", dpi=400)
