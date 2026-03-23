from Rules import Rule






#TODO implement
class CellularAutomaton():
    
    def __init__(self,rules: list[Rule],neighborhood_mask):
        self.rules = rules
        self.neighborhood_mask = neighborhood_mask
        
        
    #TODO implement
    def apply(self, state):
        new_state = ... #zeros?
        for rule in self.rules:
            pass
            #new_state += rule.apply(state)
            
        #return new_state