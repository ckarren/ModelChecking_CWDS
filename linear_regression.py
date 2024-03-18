import wntr
import numpy as np
#  import wntr.network.controls as controls
from sklearn import linear_model 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
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

        X1 = regime1[0]
        Y1 = regime1[1]

        X2 = regime2[0]
        Y2 = regime2[1]

        reg1 = linear_model.LinearRegression()
        reg2 = linear_model.LinearRegression()

        model1 = reg1.fit(X1, Y1)
        model2 = reg2.fit(X2,  Y2)


        eq1 = [reg1.intercept_, reg1.coef_[0], reg1.coef_[1]]
        eq1 = [round(x,3) for x in eq1]
        eq2 = [reg2.intercept_, reg2.coef_[0], reg2.coef_[1]]
        eq2 = [round(x,3) for x in eq2]

        x1 = [i[0] for i in X1]
        y1 = [i[1] for i in X1]
        z1 = Y1

        x_pred = np.linspace(0, 6.5, 50)   # range of porosity values
        y_pred = np.linspace(50, 250, 50)  # range of brittleness values
        xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
        model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

        predicted = model1.predict(model_viz)

        plt.style.use('default')

        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        #  ax3 = fig.add_subplot(133, projection='3d')

        #  axes = [ax1, ax2, ax3]
        axes = [ax1, ax2]
        #  ax1.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        #  ax1.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted,
                   #  facecolor=(0,0,0,0.31), s=10, edgecolor='#b0b0b0')
        #  ax1.set_xlabel('Tank water level (m)', fontsize=10)
        #  ax1.set_ylabel('Water demand (m^3/s)', fontsize=10)
        #  ax1.set_zlabel('Pump flowrate (m^3/s)', fontsize=10)
 

        for ax in axes:
            ax.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
            ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted,
                       facecolor=(0,0,0,0.31), s=10, edgecolor='#b0b0b0')
            ax.set_xlabel('Tank water level (m)', fontsize=10)
            ax.set_ylabel('Water demand (m^3/s)', fontsize=10)
            ax.set_zlabel('Pump flowrate (m^3/s)', fontsize=10)
            #  ax.locator_params(nbins=4, axis='x')
            #  ax.locator_params(nbins=5, axis='x')


        ax1.view_init(elev=28, azim=120)
        ax2.view_init(elev=4, azim=114)
        #  ax3.view_init(elev=60, azim=165)


        fig.tight_layout()
        plt.show()
    return eq1, eq2

regress()
