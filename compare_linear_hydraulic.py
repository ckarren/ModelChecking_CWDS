import numpy as np 
import linear_model as lm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import warnings
warnings.simplefilter("ignore", UserWarning)

cont_demand = lm.set_demand_pattern()
tank_constant = lm.calc_constant()
initial_height = 3.5
n_demand_states = 39
n_tank_states = 47

tank_states = lm.create_states(0.0,6.5,n_tank_states)
#  print(tank_states)

i = lm.disc_pump_rules(6.3, n_tank_states)
j = lm.disc_pump_rules(4.4, n_tank_states)
k = lm.disc_pump_rules(4.0, n_tank_states)
l = lm.disc_pump_rules(1.0, n_tank_states)


round_disc_demand = lm.round_to_disc(cont_demand,n_demand_states)
#  print(round_disc_demand[2])

demand_state_val = np.unique(round_disc_demand[1])
phd = lm.deltaHeight(demand_state_val, tank_states, tank_constant)

mt_linear = lm.lin_tank_height(cont_demand, initial_height)        # linear model 
mt_disc = lm.disc_tank_height(round_disc_demand, n_tank_states, initial_height)    #discretized model
gt = lm.epanet_groundtruth(initial_height)     #hydraulic sim in WNTR
mt_dict = lm.dict_tank_height(round_disc_demand, phd, n_tank_states, h_init=initial_height)        #dictionary model

#  for i in range(169):
    #  print('linear:', mt_linear[2][i], mt_linear[1][i], '| disc:', mt_disc[2][i], mt_disc[1][i], '| dict:', mt_dict[2][i], mt_dict[1][i],  '| epanet:', gt[1][i], gt[0][i])
    # print(i, 'disc:', mt_disc[1][i], mt_disc[2][i], 'dict', mt_dict[1][i], mt_dict[2][i])

x = mt_linear[0]
lin_y = mt_linear[1]
disc_y = mt_disc[1]
gt_y = gt[0]
dict_y = mt_dict[1]
gt_p = gt[1]
lin_p = mt_linear[2]
dis_p = mt_disc[2]

# for i, item in enumerate(gt_p):
#     print("groundtruth:", round(gt_y[i],2), gt_p[i], "linear:", round(lin_y[i],2), lin_p[i], "discrete:", disc_y[i], dis_p[i])
    # if item != dis_p[i]:
    #     print(i)
    # else:
    #     pass
r = lm.calc_rmse(disc_y, gt_y)
r_label = "RMSE=" + str(r)
plt.rcParams['axes.spines.right']=False
plt.rcParams['axes.spines.top']=False
plt.rcParams.update({'font.size':20})
fig, ax = plt.subplots()
#  ax.plot(x, lin_y, 'tab:orange', label='Linear model')
ax.plot(x, gt_y, 'tab:blue', linewidth=3.0, label='EPANET simulation')
ax.plot(x, disc_y, 'tab:orange', linewidth=3.0, label='Linear-discrete model')
#  ax.plot(x, dict_y, 'tab:red', label='Dictionary model')
ax.set(xlabel='Time (hr)', ylabel='TANK water level (m)')
at = AnchoredText(
    r_label, prop=dict(size=16), frameon=False, loc='lower right')
#  at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
ax.legend(frameon=False)
plt.show()

# fig.savefig()
