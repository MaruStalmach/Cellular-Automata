from Geometry import Geometry






#TODO implement
class Cell():
    
    def __init__(self):
        #TODO: allow configurable initialization
        self._data = {'bactera': 1000, 'CO2': 0.3}
       
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data
        
        
#TODO implement       
class State():
    
    def __init__(self, geometry: Geometry):
        #TODO: actually fill this with Cells based on Geometry
        self._data = []
        self.geometry = geometry
        