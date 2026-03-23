from libs.Rules import *
from libs.Geometry import *
from libs.Cells import *
from libs.Sim import *














if __name__=='__main__':
    #TODO parse args
    
    
    
    
    rules = []
    test_rule = TestRule()
    rules.append(test_rule)
    
    geometry = Geometry((1,2,3), 'xyz', 'x')
    state = State(geometry=geometry)
    ca = CellularAutomaton(rules=rules, neighborhood_mask=[])
    
    while True:
        new_state = ca.apply(state)
        #TODO some processing here, maybe save states
        state = new_state
        #TODO breaking condition?