import numpy as np 
from lin_model import * 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.text import OffsetFrom
import warnings
warnings.simplefilter("ignore", UserWarning)

cont_demand = set_demand_pattern()
tank_constant = calc_constant()
initial_height = 3.0
gt = epanet_groundtruth(initial_height)[0]

t_arr = []
for i in range(2, 100):
    for j in range(2, 100):
        n_demand_states = i
        n_tank_states = j
        round_disc_demand = round_to_disc(cont_demand, n_demand_states)
        mt_disc = disc_tank_height(round_disc_demand, n_tank_states, initial_height)
        disc_y = mt_disc[1]
        r = calc_rmse(disc_y, gt)
        n = i*j
        t_arr.append([n,r,(i,j)])

t_arr_n = sorted(t_arr, key= lambda x: x[0])
t_arr_r = sorted(t_arr, key= lambda x: x[1])[:5]

# for i in t_arr_r:
#     print(i[2])
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.spines.right']=False
plt.rcParams['axes.spines.top']=False
fig, ax = plt.subplots()
x = [n[0] for n in t_arr_n]
y = [r[1] for r in t_arr_n]
xy = np.array(list(zip(x,y)))
pf = keep_efficient(xy)
px, py = zip(*pf)

n_label = str(t_arr_r[0][2])
r_minx, r_miny = t_arr_r[0][0], t_arr_r[0][1]
r_label = "RMSE=" + str(r_miny)
ax.scatter(x, y, marker='.', c='b')
#  ax.scatter(px, py, marker='s', c='k')
#  ax.plot(r_minx, r_miny, 'rs', markersize=5)
#  ax.annotate(n_label,
            #  xy=(r_minx, r_miny),
            #  xycoords='data',
            #  xytext=(0,-7),
            #  textcoords='offset points',
            #  horizontalalignment='center',
            #  verticalalignment='top'
            #  )
ax.set(xlabel='number of states', ylabel='RMSE') 
#  at = AnchoredText(
    #  r_label, prop=dict(size=10), frameon=True, loc='upper right')
#  at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#  ax.add_artist(at)
plt.show()

