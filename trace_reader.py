
with open('attack_trace04.txt', 'r') as f:
    lines = f.readlines()
    trace_demand = [int(x[-2:-1]) for x in lines if 'demand' in x and 'chan' not in x]
    trace_open = [int(x[-2:-1]) for x in lines if 'open' in x]
    trace_pump_chan = [x[:-1] for x in lines if 'pump' in x and 'chan' not in x]
    trace_tank = [int(x[-2:-1]) for x in lines if 'Tank' in x and 'chan' not in x]
    trace_tank_chan = [x[:-1] for x in lines if 'tank' in x and 'chan' in x]
    trace_plc1 = [x[:-1] for x in lines if 'plc1' in x]
    trace_pump = [x[:-1] for x in lines if 'Pump' in x and 'chan' not in x]

print(trace_tank)


