from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float


class TestFuncComponent(Component): 
    x = Float(0.,iotype="in",low=-2.,high=2.)
    y = Float(0.,iotype="in",low=-1.,high=1.)
    
    f_xy = Float(0.,iotype="out")
    
    def execute(self):
        self.f_xy = (4.-2.1*(self.x**2)+(self.x**4.)/3.)*(self.x**2)+self.x*self.y+(-4.+4.*(self.y**2))*(self.y**2)