import wntr
import numpy as np
#  import wntr.network.controls as controls
from sklearn import linear_model 
import warnings
warnings.simplefilter("ignore", UserWarning)

# tank_height, flow, demand
regime1 = [[],[]]
regime2 = [[],[]]

minitown = wntr.network.WaterNetworkModel('minitown_map.inp')
minitown.options.time.duration = 604800
def set_demand_pattern(demand_file = 'demand.txt'):
    demand_pat = np.loadtxt(demand_file)
    return demand_pat
demand = set_demand_pattern()

sim = wntr.sim.EpanetSimulator(minitown)
results = sim.run_sim()
tank_height = results.node['pressure'].loc[:,'TANK']
p1_stat = results.link['status'].loc[:, 'PUMP1']     
p2_stat = results.link['status'].loc[:,'PUMP2']
tank_flow = results.node['demand'].loc[:,'TANK'] * 1000
p1_flow = results.link['flowrate'].loc[:,'PUMP1'] * 1000 
p2_flow = results.link['flowrate'].loc[:,'PUMP2'] * 1000
print(p1_stat)
def regress(network='minitown'):
    if network == 'minitown':
        for t, status in enumerate(p2_stat):
            if status == 1: 
                total_flow = p1_flow.loc[t*3600] + p2_flow.loc[t*3600]
                regime2[0].append([tank_height.loc[t*3600],demand[t]])
                regime2[1].append(total_flow)
            elif status == 0:
                total_flow = p1_flow.loc[t*3600]
                regime1[0].append([tank_height.loc[t*3600],demand[t]])
                regime1[1].append(total_flow)

        xmat1 = regime1[0]
        y1 = regime1[1]

        xmat2 = regime2[0]
        y2 = regime2[1]

        reg1 = linear_model.LinearRegression()
        reg2 = linear_model.LinearRegression()

        reg1.fit(xmat1, y1)
        reg2.fit(xmat2,  y2)

        eq1 = [reg1.intercept_, reg1.coef_[0], reg1.coef_[1]]
        eq1 = [round(x,3) for x in eq1]
        eq2 = [reg2.intercept_, reg2.coef_[0], reg2.coef_[1]]
        eq2 = [round(x,3) for x in eq2]

    return eq1, eq2 

