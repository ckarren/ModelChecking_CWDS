import numpy as np 
import wntr
from linear_regression import regress

def set_max_height(h=6.5):
    h=h
    return h

def round_dh_to_disc(num, num_states):
    lo = 0.0
    hi = 6.5
    n = round((hi - lo) / (num_states-1),2)
    res = round(num / n) * n
    return res

def set_demand_pattern(demand_file = 'demand.txt'):
    demand_pat = np.loadtxt(demand_file)
    return demand_pat

def calc_constant(diameter = 31.3):
    tank_const = np.pi * (diameter**2/4) / 3.6 #constant = area of top/bottom of circular tank * 1000 (convert from m^3 to liters) / 3600 (convert from seconds to hours)
    return tank_const
   
def pump_on_off(pump_one, pump_two, tank_h):
    if tank_h > 6.3:
        pump_one = 0
    elif tank_h < 4: # tested to ensure minimum RMSE b/w linear and epanet
        pump_one = 1
    if tank_h > 4.4: #re-calibrated tank rules to minimize RMSE b/w linear and epanet
        pump_two = 0
    elif tank_h < 1:
        pump_two = 1
    return pump_one, pump_two    

def set_pump_eqns(**kwargs):
    if 'from_regression' in kwargs:
        if not isinstance (kwargs['from_regression'], bool):
            raise TypeError('keyword argument "from_regression" must be True or False (bool)')
        if kwargs['from_regression']: 
            eq1, eq2 = regress()
        elif 'equation1' and 'equation2'in kwargs:
            eq1 = kwargs['equation1']
            eq2 = kwargs['equation2']
            if not isinstance(eq1, list) or len(eq1) != 3 or not isinstance(eq2,
                                                                        list) or len(eq2) != 3:
                raise TypeError("equation1 and equation2 must be lists of length 3. Each list must have the equation constant in position 0, the coefficient for the tank height in position 1, and the coefficient for the demand in position 2.")

    else:
        if 'equation1' and 'equation2' in kwargs:
            eq1 = kwargs['equation1']
            eq2 = kwargs['equation2']
            if not isinstance(eq1, list) or len(eq1) != 3 or not isinstance(eq2,
                                                                        list) or len(eq2) != 3:
                raise TypeError("equation1 and equation2 must be lists of length 3. Each list must have the equation constant in position 0, the coefficient for the tank height in position 1, and the coefficient for the demand in position 2.")
        else:
            raise TypeError("missing at least 1 required positional argument: 'from_regression' or 'equation1' and 'equation2'")
    return eq1, eq2

def pump_flow(pump1, pump2, tank_h, demand):
    # to set linear equations manually, pass set_pump_eqns() with 2 lists of
    # coefficients from linear regression output. To use linear_regression(), pass
    # 'from_regression=True'. When using calc_RMSE.py, the linear_regression()
    # considerably slows performance. Passing equaiton coefficients is suggested
    eq1, eq2 = set_pump_eqns(equation1=[113.9,-1.34,0.036],
                    equation2=[183.54,-2.20,0.067])
    if pump1 + pump2 == 0:
        Qpump = 0
    elif pump1 + pump2 == 1:
        Qpump = eq1[0] + eq1[1] * tank_h + eq1[2] * demand
        #  Qpump = 113.9 + 0.036 * demand - 1.34 * tank_h
    else:
        Qpump = eq2[0] + eq2[1] * tank_h + eq2[2] * demand
        #  Qpump = 183.54 + .067 * demand - 2.2 * tank_h
    Qpump = round(Qpump, 3)

    return Qpump

def slope_calc(demand, tank_constant):
    s = demand / tank_constant
    return s

def round_near(value, nearest):
    return round(value/nearest) * nearest

def deltaHeight(demand_list, tank_list, const):
    tank_inc = tank_list[1] - tank_list[0]
    dHeight_dict = {}   #create dictionary
    dHeight_dict[0] = {}    #create key for when 0 pumps are on 
    dHeight_dict[1] = {}  #create key for when 1 pumps are on
    dHeight_dict[2] = {}  #create key for when 2 pumps are on
    for i in demand_list:
        # print(i)
        dHeight_dict[0][i] = {}   #create key for each demand level
        dHeight_dict[1][i] = {}   #create key for each demand level
        dHeight_dict[2][i] = {}   #create key for each demand level 
        for j in tank_list:
            q = i - pump_flow(0,0,j,i)
            dh = slope_calc(q,const)
            dHeight_dict[0][i][j] = round(dh / (tank_inc)) * tank_inc #create value for each pump and demand key
            q = i - pump_flow(1,0,j,i)
            dh = slope_calc(q,const)
            dHeight_dict[1][i][j] = round(dh / (tank_inc)) * tank_inc   #create value for each pump and demand key
            q = i - pump_flow(1,1,j,i)
            dh = slope_calc(q,const)
            dHeight_dict[2][i][j] = round(dh / (tank_inc))  * tank_inc   #create value for each pump and demand key
    return dHeight_dict

def epanet_groundtruth(init_tank=3.0, file_name='minitown_map.inp', duration=168, pump1_init=0, pump2_init=1):
    """runs an EPANET hydraulic simulation and returns an array of tank values"""
    wn = wntr.network.WaterNetworkModel(file_name)
    duration = 3600 * duration
    wn.options.time.duration = duration
    pump1 = wn.get_link("PUMP1")
    pump2 = wn.get_link("PUMP2")
    tank = wn.get_node('TANK')
    pump1.initial_status = pump1_init
    pump2.initial_status = pump2_init
    tank.init_level = init_tank
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    tank = wn.get_node('TANK')
    t_height = results.node['pressure'].loc[:,'TANK']
    p1_status = results.link['status'].loc[:,'PUMP1']
    p2_status = results.link['status'].loc[:,"PUMP2"]
    th = np.array(t_height.values.tolist())
    p1 = np.array(p1_status.values.tolist())
    p2 = np.array(p2_status.values.tolist())
    # t_arr = [x for x in range(duration)]
    pump_status = [(int(x),int(y)) for ix, x in enumerate(p1) for iy, y in enumerate(p2) if ix == iy]
    
    return th, pump_status

def create_states(lb, ub, num_states):
    increment = round((ub - lb) / (num_states-1), 2)
    states = []
    i = lb
    while i <= ub:
        i = round(i, 2)
        states.append(i)
        i += increment
    return states[:num_states]

def lin_tank_height(demand, h_init=3.0, p1_init=0, p2_init=1,  t_tot=168):
    """"Calculates the height of the tank at every time step.
        p1_init::Int {0,1} = initial status of pump1; 0 = off, 1 = on
        p2_init::Int {0,1} = inital status of pump2; 0 = off, 1 = on
        h_init::Float = initial height of tank. For network_ min = 0, max = 6.5
        t_tot::Int = total time of simulation in hours
        demand::List = list of demand values at each time step
        
        returns a list of time steps, and a list of tank height values """

    pump_1 = p1_init
    pump_2 = p2_init
    height = h_init
    time_arr = []
    height_arr = []
    pump_arr = []
    t = 0
    while t <= t_tot:
        time_arr.append(t)
        height_arr.append(height)
        pump_arr.append((pump_1, pump_2))
        pump_1, pump_2 = pump_on_off(pump_1, pump_2, height)
        dt = demand[t]
        qt = dt - pump_flow(pump_1, pump_2, height, dt)
        c = calc_constant()
        height = height - slope_calc(qt,c)
        t += 1

    return time_arr, height_arr, pump_arr

def calc_rmse(pred, obs):
    r = pred - obs
    length = len(r)
    rmse = np.sqrt(np.sum(r**2)/length)

    return round(rmse, 3)

def digi_to_disc(pattern, num_states):
    lo = np.min(pattern)
    hi = np.max(pattern)
    demand_states = create_states(lo,hi,num_states)
    dem_disc_list = []
    states = np.digitize(pattern, demand_states, right=True)
    for i, item in enumerate(pattern):
        dem_disc_list.append(demand_states[states[i]])

    return states, dem_disc_list

def round_to_disc(pattern, num_states, is_tank=False):
    if is_tank:
        lo = 0.0
        hi = 6.5
    else:
        lo = np.min(pattern)
        hi = np.max(pattern)
    n = (hi - lo) / (num_states-1)
    res = [round(x / n) for x in pattern]
    st = [x - np.min(res) for x in res]
    res = [round(x * n,2) for x in res]
    st_val = np.unique(res)
    return st, res, st_val



def disc_pump_rules(level, n_tank_states, max_tank_height=6.5):
    rule = round(level / max_tank_height * n_tank_states)
    return rule


def disc_pump_on_off(pump_one, pump_two, tank_h, num_tank_states, max_tank_height=6.5):
    i = disc_pump_rules(6.3, num_tank_states, max_tank_height)
    j = disc_pump_rules(4.0, num_tank_states, max_tank_height) 
    k = disc_pump_rules(4.4, num_tank_states, max_tank_height)
    l = disc_pump_rules(1.0, num_tank_states, max_tank_height)
    tank_s = round(tank_h / max_tank_height * num_tank_states)
    if tank_s >= i: 
        pump_one = 0
    elif tank_s <= j:  #tested to ensure minimum RMSE b/w linear and epanet
        pump_one = 1
    if tank_s >= k : #re-calibrated tank rules to minimize RMSE b/w linear and epanet
        pump_two = 0
    elif tank_s <= l: 
        pump_two = 1

    return pump_one, pump_two

def disc_tank_height(demand, num_tank_states, h_init=3.0, p1_init=0, p2_init=1, t_tot=168):
    """"Calculates the height of the tank at every time step.
        demand::List = list of demand values at each time step
        h_init::Float = initial height of tank. For network_ min = 0, max = 6.5
        p1_init::Int {0,1} = initial status of pump1; 0 = off, 1 = on
        p2_init::Int {0,1} = inital status of pump2; 0 = off, 1 = on
        t_tot::Int = total time of simulation in hours
        
        returns a list of time steps, and a list of corresponding height values """

    pump_1 = p1_init
    pump_2 = p2_init
    height = h_init
    time_arr = []
    height_arr = []
    pump_arr = []
    t = 0
    while t <= t_tot:
        time_arr.append(t)
        height_arr.append(height)
        pump_arr.append((pump_1, pump_2))
        pump_1, pump_2 = disc_pump_on_off(pump_1, pump_2, height, num_tank_states)[:2]
        dt = demand[1][t]
        pf = pump_flow(pump_1, pump_2, height, dt)
        qt = dt - pf
        c = calc_constant()
        slope = slope_calc(qt,c)
        height = height - slope
        t += 1
    disc_h_res = round_to_disc(height_arr, num_tank_states, True)[1]
    
    return time_arr, disc_h_res, pump_arr

def dict_tank_height(demand, dict, num_tank_states, h_init=3.0, p1_init=0, p2_init=1, t_tot=168):
    """"Calculates the height of the tank and every time step.
        p1_init = initial status of pump1; 0 = off, 1 = on
        p2_int = inital status of pump2; 0 = off, 1 = on
        height = initial height of tank. For minitown min = 0, max = 6.5
        t_step = size of time step in hours"""
    pump_1 = p1_init
    pump_2 = p2_init
    height = round_dh_to_disc(h_init, num_tank_states)
    time_arr = []
    height_arr = []
    pump_arr = []
    t = 0
    while t <= t_tot:
        time_arr.append(t)
        height = round(height, 2)
        height_arr.append(height)
        pump_arr.append((pump_1, pump_2))
        pump_1, pump_2 = disc_pump_on_off(pump_1, pump_2, height, num_tank_states)[:2]
        dt = demand[1][t]
        p = pump_1 + pump_2
        delta_height = dict[p][dt][height]
        height -= delta_height
        t += 1 

    return time_arr, height_arr, pump_arr

def keep_efficient(pts):
    'returns Pareto efficient row subset of pts'
    # sort points by increasing sum of coordinates
    pts = pts[pts.sum(1).argsort()]
    #  print(pts)
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        # process each point in turn
        n = pts.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i+1:n] = (pts[i+1:] <= pts[i]).any(1) 
        # keep points undominated so far
        pts = pts[undominated[:n]]
        #  undominated = np.array([True]*len(pts))
    return pts

# def discrete_results(demand_pattern, num_states_low, num_states_high=False):
#     dem_list_of_list = list()
#     lo = np.min(demand_pattern)
#     hi = np.max(demand_pattern)
#     if num_states_high == False:
#         demand_states = create_states(lo,hi,num_states_low)
#         # `print`(demand_states)
#         dem_state_list = []
#         demand_res = np.digitize(demand_pattern, demand_states, right=True)
#         for i in range(len(demand_pattern)):
#             dem_state_list.append(demand_states[demand_res[i]])
#         dem_list_of_list.append(dem_state_list)
#     else:
#         for j in range(num_states_low,num_states_high):
#             demand_states = create_states(lo, hi, j)
#             # print(demand_states)
#             dem_state_list = []
#             demand_res = np.digitize(demand_pattern, demand_states, right=True)
#             for i in range(len(demand_pattern)):
#                 dem_state_list.append(demand_states[demand_res[i]])
#             dem_list_of_list.append(dem_state_list)

#     return dem_list_of_list

# def round_to_disc(demand_pattern, num_states_low, num_states_high):
#     dem_list_of_list = list()
#     lo = np.min(demand_pattern)
#     hi = np.max(demand_pattern)
#     if num_states_high == False:
#         n = (hi - lo) / (num_states_low-1)
#         res = [round(round(x/n)*n,2) for x in demand_pattern]
#     else:
#         for j in range(num_states_low,num_states_high):
#             n = (hi - lo) / (j - 1)
#             res = [round(round(x/n)*n,2) for x in demand_pattern]
#             dem_list_of_list.append(res)
    
#     return dem_list_of_list

    
    # for k in range(5,30):
    #     tank_states = create_states(0.0, 6.5, k)
    #     tank_state_list = []
    #     tank_res = disc_results(lin_timeseries, tank_states)
    #     for l in range(len(tank_res)):
    #         tank_state_list.append(tank_states[tank_res[l]])

    
# def tank_height(p1_init, p2_init, height, t_tot, demand):
#     """"Calculates the height of the tank at every time step.
#         p1_init = initial status of pump1; 0 = off, 1 = on
#         p2_init = inital status of pump2; 0 = off, 1 = on
#         height = initial height of tank. For Minitown min = 0.0, max = 6.5
#         """
#     pump_1 = p1_init
#     pump_2 = p2_init
#     time_arr = [0.0]
#     height_arr = [height]
#     t = 0
#     while t < t_tot:
#         dt = demand[t]
#         pf = pump_flow(pump_1, pump_2, height, dt)
#         qt = dt - pf
#         c = calc_constant(tank_diameter)
#         slope = slope_calc(qt,c) 
#         height = round(height - slope, 3)
#         t += 1
#         pump_1, pump_2 = pump_on_off(pump_1, pump_2,height)
#         time_arr.append(t)
#         height_arr.append(height)

#     return time_arr, height_arr
