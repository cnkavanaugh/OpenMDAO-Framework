from openmdao.main.api import Assembly
from openmdao.lib.components.api import MetaModel
from openmdao.lib.surrogatemodels.api import NeuralNetSurrogate
from openmdao.examples.expected_improvement.branin_component import BraninComponent

class Simulation(Assembly):
    def __init__(self):
        super(Simulation,self).__init__(self)
        
    self.add("meta_model",MetaModel())
    self.meta_model.surrogate = {"default" :NeuralNetSurrogate()}
    
    self.meta_model.model = BraninComponent()
    
    
    
    
    