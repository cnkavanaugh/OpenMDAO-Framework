import unittest

from openmdao.main.api import Assembly, Component

from openmdao.lib.drivers.api import DOEdriver
from openmdao.lib.doegenerators.api import Uniform

from openmdao.lib.surrogatemodels.neuralnet_surrogate import NeuralNetSurrogate

class test_function(Component):
    def __init__(self, x, y):

        x = Float(0.,iotype="in",low=-2.,high=2.)
        y = Float(0.,iotype="in",low=-1.,high=1.)
            
        f_xy = Float(0.,iotype="out")
    
        self.f_xy = (4.-2.1*(x**2)+(x**4.)/3.)*(x**2)+x*y+(-4.+4.*(y**2))*(y**2)

class Analysis(Assembly):
    def __init__(self):
        #Driver Configuration
        self.add("test", test_function())
        self.add("driver",DOEdriver())
        self.driver.sequential = True
        self.driver.DOEgenerator = Uniform
        self.driver.num_samples = 500
        self.driver.add_parameter("test.x")
        self.driver.add_parameter("test.y")
        self.driver.case_outputs = ["test.f_xy"]
         
        
class NeuralNetSurrogateTest(unittest.TestCase):
    
    def setUp(self):
        