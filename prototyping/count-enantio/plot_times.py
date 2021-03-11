import matplotlib.pyplot as plt
import numpy as np

def unique_average(arr1, arr2):
    #Takes two arrays, calculates the averages of elements in arr2,
    #where the elements in arr1 are equal
    unique_entries, amount = np.unique(arr1, return_counts = True)
    averages = np.zeros((len(unique_entries)))
    variance = np.zeros((len(unique_entries)))
    for i in range(len(unique_entries)):
        averages[i] = np.sum([arr2[k] for k in np.where(arr1 == unique_entries[i])[0]])/amount[i]
        variance[i] = np.sum(np.square([arr2[k] for k in np.where(arr1 == unique_entries[i])[0]]))/amount[i] - averages[i]*averages[i]
    return unique_entries, averages, np.sqrt(variance)

# Data for plotting
x_benzene_geom = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0]
y_benzene_geom = [0.0655977725982666, 0.02986598014831543, 0.029832839965820312, 0.04710078239440918, 0.0663290023803711, 0.06626486778259277, 0.06628680229187012, 0.047200918197631836, 0.04717659950256348, 0.024738550186157227, 0.024796724319458008, 0.024837493896484375, 0.02215266227722168, 0.16579389572143555, 0.2028810977935791, 0.1661677360534668, 0.20386481285095215, 0.15006470680236816, 0.11501765251159668, 0.13146066665649414]

x_benzene_graph = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0]
y_benzene_graph = [0.007934331893920898, 0.008654594421386719, 0.008385658264160156, 0.008409261703491211, 0.008989334106445312, 0.00897526741027832, 0.0086212158203125, 0.008178234100341797, 0.008211851119995117, 0.008185863494873047, 0.008076190948486328, 0.008115053176879883, 0.008067846298217773, 0.009568929672241211, 0.009435176849365234, 0.009201765060424805, 0.009514331817626953, 0.009648799896240234, 0.00869894027709961, 0.00900888442993164]

x_naphthalene_geom = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
y_naphthalene_geom = [1.0235497951507568, 1.0211069583892822, 1.0270884037017822, 1.0241551399230957, 1.2916820049285889, 1.2943410873413086, 1.2952148914337158, 1.006821870803833, 1.029066562652588, 1.7547507286071777, 1.7590100765228271, 1.7639405727386475, 1.2601706981658936, 1.052241325378418, 1.5308525562286377, 1.536454677581787, 1.5334722995758057, 25.29233479499817, 25.411638259887695, 25.85024642944336, 25.92821717262268, 26.511577129364014, 25.577887058258057, 25.701408863067627, 25.723045825958252, 25.882932662963867, 25.840433597564697, 25.427305936813354, 25.87976598739624, 26.119029760360718]
x_naphthalene_graph = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
y_naphthalene_graph = [0.05203843116760254, 0.05348372459411621, 0.04729747772216797, 0.051151275634765625, 0.07581067085266113, 0.05625557899475098, 0.07397937774658203, 0.06010174751281738, 0.09139895439147949, 0.13733696937561035, 0.1364130973815918, 0.13741540908813477, 0.10596179962158203, 0.07405471801757812, 0.11819124221801758, 0.11025071144104004, 0.10877394676208496, 0.6441943645477295, 0.6590325832366943, 0.7389793395996094, 0.8975772857666016, 2.0682294368743896, 0.8055605888366699, 1.2277007102966309, 0.863776445388794, 0.8683903217315674, 1.2457799911499023, 1.20969820022583, 2.1372721195220947, 3.4326844215393066]

x_phenanthrene_geom = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
y_phenanthrene_geom =
x_phenanthrene_graph = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
y_phenanthrene_graph = [7.096792459487915, 6.938782453536987, 6.856269359588623,7.269430160522461, 8.23498821258545, 7.967238664627075, 7.9096856117248535, 7.3395280838012695, 17.085789680480957, 107.07211780548096, 108.12050366401672, 108.02251744270325, 63.102222204208374, 212.98513770103455, 1401.785100698471, 1410.2719943523407, 1411.6955749988556, 426.9249300956726, 431.1462433338165, 471.003466129303, 522.3381226062775, 2730.2871599197388, 644.0184328556061, 1409.4556698799133, 10218.455395936966]

x_benzene_geom,y_benzene_geom, yerr_benzene_geom = unique_average(x_benzene_geom, y_benzene_geom)
x_benzene_graph,y_benzene_graph, yerr_benzene_graph = unique_average(x_benzene_graph, y_benzene_graph)
x_naphthalene_geom,y_naphthalene_geom, yerr_naphthalene_geom = unique_average(x_naphthalene_geom, y_naphthalene_geom)
x_naphthalene_graph,y_naphthalene_graph, yerr_naphthalene_graph = unique_average(x_naphthalene_graph, y_naphthalene_graph)
x_phenanthrene_geom,y_phenanthrene_geom, yerr_phenanthrene_geom = unique_average(x_phenanthrene_geom, y_phenanthrene_geom)
x_phenanthrene_graph,y_phenanthrene_graph, yerr_phenanthrene_graph = unique_average(x_phenanthrene_graph, y_phenanthrene_graph)

fig, ax = plt.subplots()
ax.scatter(x_benzene_geom, y_benzene_geom, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(x_benzene_geom, y_benzene_geom, yerr= yerr_benzene_geom, fmt='none', capsize=4)
ax.scatter(x_benzene_graph, y_benzene_graph, marker='x', color='#ff7f0e', label='graph-based')
plt.errorbar(x_benzene_graph, y_benzene_graph, yerr= yerr_benzene_graph, fmt='none', capsize=4)
ax.set_xticks(range(2,9))
ax.set_xlim([1.5, 8.7])
ax.set(xlabel='Total number of transmuted atoms m', ylabel='Time / s')
#ax.grid(which='both')
plt.yscale('log')
ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("benzene.png", dpi=300)

fig, ax = plt.subplots()
ax.scatter(x_naphthalene_geom, y_naphthalene_geom, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(x_naphthalene_geom, y_naphthalene_geom, yerr= yerr_naphthalene_geom, fmt='none', capsize=4)
ax.scatter(x_naphthalene_graph, y_naphthalene_graph, marker='x', color='#ff7f0e', label='graph-based')
plt.errorbar(x_naphthalene_graph, y_naphthalene_graph, yerr= yerr_naphthalene_graph, fmt='none', capsize=4)
ax.set_xticks(range(2,9))
ax.set_xlim([1.5, 8.7])
ax.set(xlabel='Total number of transmuted atoms m', ylabel='Time / s')
#ax.grid(which='both')
plt.yscale('log')
ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("naphthalene.png", dpi=300)
