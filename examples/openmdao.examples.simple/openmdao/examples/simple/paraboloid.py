"""
    paraboloid.py - Evaluates the equation (x-3)^2 + xy + (y+4)^2 = 3
"""


from enthought.traits.api import Float
from openmdao.main.api import Component
from openmdao.lib.traits.unitsfloat import UnitsFloat

class Paraboloid(Component):
    """ Evaluates the equation (x-3)^2 + xy + (y+4)^2 = 3 """
    
    # set up interface to the framework  
    # Pylint: disable-msg=E1101
    x = UnitsFloat(0.0, iotype='in', units='m', 
                           desc='The variable y')
    y = UnitsFloat(0.0, iotype='in', units='m', 
                           desc='The variable x')

    f_xy = UnitsFloat(0.0, iotype='out', units='m*m',
                     desc='F(x,y)')        

        
    def execute(self, required_outputs=None):
        """ Solve (x-3)^2 + xy + (y+4)^2 = 3
            Optimal solution (minimum): x = 6.6667; y = -7.3333
            """
        
        x = self.x
        y = self.y
        
        self.f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        
# End paraboloid.py
