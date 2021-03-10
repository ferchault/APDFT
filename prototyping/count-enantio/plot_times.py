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
y_benzene_geom = [0.07077813148498535, 0.03452801704406738, 0.03272294998168945, 0.05200815200805664, 0.08956122398376465, 0.07868766784667969, 0.13334870338439941, 0.09191417694091797, 0.06476974487304688, 0.03295445442199707, 0.031191587448120117, 0.03102421760559082, 0.027004241943359375, 0.19867563247680664, 0.22060275077819824, 0.1844043731689453, 0.23004436492919922, 0.16398382186889648, 0.12388181686401367, 0.21945619583129883]
x_benzene_graph = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0]
y_benzene_graph = [0.005930423736572266, 0.006354093551635742, 0.008337020874023438, 0.005499839782714844, 0.006399631500244141, 0.0063359737396240234, 0.007510662078857422, 0.0062940120697021484, 0.0062901973724365234, 0.0073773860931396484, 0.007192373275756836, 0.005376577377319336, 0.005330324172973633, 0.0076885223388671875, 0.008650541305541992, 0.006918191909790039, 0.006831645965576172, 0.006878376007080078, 0.0075800418853759766, 0.009011507034301758]

x_naphthalene_geom = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
y_naphthalene_geom = [1.2885758876800537, 1.219634771347046, 1.210921049118042, 1.2065443992614746, 1.510908603668213, 1.4937160015106201, 1.5214874744415283, 1.1927168369293213, 1.2040925025939941, 1.9637696743011475, 1.9897549152374268, 1.9841580390930176, 1.4253320693969727, 1.2352361679077148, 1.728545904159546, 1.7756686210632324, 1.7253837585449219, 28.765464067459106, 28.531126976013184, 29.046489477157593, 29.20816159248352, 29.885051727294922, 28.775543451309204, 29.31024169921875, 28.851908922195435, 29.212769508361816, 29.218756437301636, 28.956728219985962, 28.885541200637817, 29.09013032913208]
x_naphthalene_graph = [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
y_naphthalene_graph = [0.08035588264465332, 0.04633355140686035, 0.0841820240020752, 0.05828499794006348, 0.06595373153686523, 0.07195568084716797, 0.06621074676513672, 0.09588003158569336, 0.13596701622009277, 0.2158803939819336, 0.2244887351989746, 0.211045503616333, 0.14056015014648438, 0.10062408447265625, 0.17010951042175293, 0.1289994716644287, 0.17113637924194336, 0.9788789749145508, 1.059906005859375, 1.1207072734832764, 1.427905559539795, 4.518952369689941, 1.5364999771118164, 2.2630534172058105, 1.4910247325897217, 1.5040547847747803, 2.544736862182617, 2.3294594287872314, 3.795930862426758, 6.005266189575195]

x_benzene_geom,y_benzene_geom, yerr_benzene_geom = unique_average(x_benzene_geom, y_benzene_geom)
x_benzene_graph,y_benzene_graph, yerr_benzene_graph = unique_average(x_benzene_graph, y_benzene_graph)
x_naphthalene_geom,y_naphthalene_geom, yerr_naphthalene_geom = unique_average(x_naphthalene_geom, y_naphthalene_geom)
x_naphthalene_graph,y_naphthalene_graph, yerr_naphthalene_graph = unique_average(x_naphthalene_graph, y_naphthalene_graph)

fig, ax = plt.subplots()
ax.scatter(x_benzene_geom, y_benzene_geom, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(x_benzene_geom, y_benzene_geom, yerr= yerr_benzene_geom, fmt='none', capsize=4)
ax.scatter(x_benzene_graph, y_benzene_graph, marker='x', color='#ff7f0e', label='graph-based')
plt.errorbar(x_benzene_graph, y_benzene_graph, yerr= yerr_benzene_graph, fmt='none', capsize=4)
ax.set_xticks(range(2,9))
ax.set_xlim([1.5, 8.7])
ax.set(xlabel='Total number of transmuted atoms m', ylabel='Time / s',
       title='Mean computation time of FindAE for benzene (N = 6)')
ax.grid(which='both')
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
ax.set(xlabel='Total number of transmuted atoms m', ylabel='Time / s',
       title='Mean computation time of FindAE for naphthalene (N = 10)')
ax.grid(which='both')
plt.yscale('log')
ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("naphthalene.png", dpi=300)
