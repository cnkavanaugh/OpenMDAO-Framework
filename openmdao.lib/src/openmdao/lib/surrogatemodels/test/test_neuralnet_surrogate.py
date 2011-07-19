import unittest
import random

from openmdao.main.api import Assembly, Component

from openmdao.lib.datatypes.api import Float
from openmdao.lib.drivers.api import DOEdriver
from openmdao.lib.doegenerators.api import Uniform

# from openmdao.lib.surrogatemodels.neuralnet_surrogate import NeuralNetSurrogate

class NeuralNetTests(unittest.TestCase):
    
    def setUp(self):
        random.seed(10)

        def test_function(self):
            x = Float(0.,iotype="in",low=-2.,high=2.)
            y = Float(0.,iotype="in",low=-1.,high=1.)
        
            f_xy = Float(0.,iotype="out")    
    
            def function(x,y):
                self.f_xy = (4.-2.1*(self.x**2)+(self.x**4.)/3.)*(self.x**2)+self.x*self.y+(-4.+4.*(self.y**2))*(self.y**2)

class Analysis(Assembly):
    def __init__(self):
        super(Assembly,self).__init__()
        #Driver Configuration
        self.add("test", test_function())
        self.add("driver",DOEdriver())
        self.driver.sequential = True
        self.driver.DOEgenerator = Uniform()
        self.driver.num_samples = 500
        self.driver.add_parameter("test.x")
        self.driver.add_parameter("test.y")
        self.driver.case_outputs = ["test.f_xy"]
        
if __name__ == "__main__":
    a = Analysis()
    a.run()
        
        
