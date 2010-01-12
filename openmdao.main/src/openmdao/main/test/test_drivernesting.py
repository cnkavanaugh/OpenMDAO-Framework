# pylint: disable-msg=C0111,C0103

import unittest
import logging
from math import sqrt

from enthought.traits.api import Float, Int, Str

from openmdao.main.api import Assembly, Component, Driver, set_as_top
from openmdao.main.stringref import StringRef
from openmdao.lib.drivers.conmindriver import CONMINdriver

exec_order = []

class Summer(Driver):
    """Sums the objective over some number of iterations, feeding
    its current sum back into the specified design variable."""
    
    objective = StringRef(iostatus='in')
    design = StringRef(iostatus='out')
    max_iterations = Int(1, iostatus='in')
    sum = Float(iostatus='out')
    
    def __init__(self):
        super(Summer, self).__init__()
        self.runcount = 0
        self.itercount = 0
    
    def continue_iteration(self):
        return self.itercount < self.max_iterations
    
    def start_iteration(self):
        self.itercount = 0
        self.sum = 0.
        
    def pre_iteration(self):
        self.design.set(1)
    
    def post_iteration(self):
        self.sum += self.objective.evaluate()
        self.itercount += 1
    
    def execute(self):
        global exec_order
        exec_order.append(self.name)
        super(Summer, self).execute()
        self.runcount += 1
        
class ExprComp(Component):
    """Evaluates an expression based on the input x and assigns it to f_x"""
    
    x = Float(iostatus='in')
    f_x = Float(iostatus='out')
    expr = Str('x', iostatus='in')

    def __init__(self, expr='x'):
        super(ExprComp, self).__init__()
        self.runcount = 0
        self.expr = expr
        
    def execute(self):
        global exec_order
        exec_order.append(self.name)
        x = self.x
        self.f_x = eval(self.expr)
        self.runcount += 1
    
class ExprComp2(Component):
    """Evaluates an expression based on the inputs x & y and assigns it to f_xy"""
    
    x = Float(iostatus='in')
    y = Float(iostatus='in')
    f_xy = Float(iostatus='out')
    expr = Str('x', iostatus='in')
    
    def __init__(self, expr='x'):
        super(ExprComp2, self).__init__()
        self.runcount = 0
        self.expr = expr

    def execute(self):
        global exec_order
        exec_order.append(self.name)
        x = self.x
        y = self.y
        self.f_xy = eval(self.expr)
        self.runcount += 1
        
class NestedDriverTestCase(unittest.TestCase):

    def test_2drivers_same_iterset(self):
        #
        #  D1--->
        #  |    |
        #  |<---C1----->
        #       |      |
        #       |<-----D2
        #
        top = set_as_top(Assembly())
        top.add_container('C1', ExprComp(expr='x+1'))
        top.add_driver('D1', Summer())
        top.D1.objective = 'C1.f_x'
        top.D1.design = 'C1.x'
        top.add_driver('D2', Summer())
        top.D2.objective = 'C1.f_x'
        top.D2.design = 'C1.x'
        try:
            top.run()
        except RuntimeError, err:
            self.assertEqual(str(err), 
                "Drivers D1 and D2 iterate over"+
                " the same set of components (['C1']), so their order"+
                " cannot be determined")
        else:
            self.fail('RuntimeError expected')
            
    def test_2drivers_discon_same_iterset(self):
        #
        #  D1--->
        #  |    |
        #  |    C1--------->|
        #  |                |
        #  |<----------C2   |
        #              |    |
        #              |<---D2
        #
        top = set_as_top(Assembly())
        top.add_container('C1', ExprComp(expr='x+1'))
        top.add_container('C2', ExprComp(expr='x+1'))
        top.add_driver('D1', Summer())
        top.D1.objective = 'C2.f_x'
        top.D1.design = 'C1.x'
        top.add_driver('D2', Summer())
        top.D2.objective = 'C1.f_x'
        top.D2.design = 'C2.x'
        try:
            top.run()
        except RuntimeError, err:
            self.assertEqual(str(err), 
                "Drivers D1 and D2 iterate over"+
                " the same set of components (['C1', 'C2']), so their order"+
                " cannot be determined")
        else:
            self.fail('RuntimeError expected')
            
    def test_2drivers_overlapping_iterset(self):
        #
        #  D2---------->|
        #  |            |
        #  |   C1------>|
        #  |   |        |
        #  |   |<---D1  |
        #  |        |   |
        #  |        |<--C2-->
        #  |                |
        #  |<---------------C3
        top = set_as_top(Assembly())
        top.add_container('C1', ExprComp(expr='x+1'))
        top.add_container('C2', ExprComp(expr='x+1'))
        top.add_container('C3', ExprComp(expr='x+1'))
        top.add_driver('D1', Summer())
        top.add_driver('D2', Summer())
        
        top.connect('C1.f_x', 'C2.x')
        top.connect('C2.f_x', 'C3.x')
        top.D1.objective = 'C2.f_x'
        top.D1.design = 'C1.x'
        top.D2.design = 'C2.x'
        top.D2.objective = 'C3.f_x'
        try:
            top.run()
        except RuntimeError, err:
            self.assertEqual(str(err), 
                "Drivers D2 and D1 have overlap"+
                " (['C2']) in their iteration sets, so their order"+
                " cannot be determined")
        else:
            self.fail('RuntimeError expected')
            
    def test_2nested_drivers(self):
        #
        #  D2-->
        #  |   |
        #  |   C1------> 
        #  |           |
        #  |       D1-->
        #  |       |   |
        #  |       |<--C2-->
        #  |               |
        #  |<--------------C3
        
        global exec_order
        exec_order = []
        
        top = set_as_top(Assembly())
        top.add_container('C1', ExprComp(expr='x+1'))
        top.add_container('C2', ExprComp2(expr='x+y'))
        top.add_container('C3', ExprComp(expr='x+1'))
        top.add_driver('D1', Summer())
        top.add_driver('D2', Summer())
        
        top.connect('C1.f_x', 'C2.x')
        top.connect('C2.f_xy', 'C3.x')
        top.D1.objective = 'C2.f_xy'
        top.D1.design = 'C2.y'
        top.D1.max_iterations = 2
        top.D2.design = 'C1.x'
        top.D2.objective = 'C3.f_x'
        top.D2.max_iterations = 3
        top.run()
        self.assertEqual(top.D2.runcount, 1)
        self.assertEqual(top.D1.runcount, top.D2.max_iterations)
        self.assertEqual(top.C1.runcount, top.D2.max_iterations)
        self.assertEqual(top.C2.runcount, 
                         top.D2.max_iterations*top.D1.max_iterations)
        self.assertEqual(exec_order,
                         ['D2', 'C1', 'D1', 'C2', 'C2', 'C3', 
                                     'C1', 'D1', 'C2', 'C2', 'C3', 
                                     'C1', 'D1', 'C2', 'C2', 'C3'])
        self.assertEqual(top.D1.sum, 3.*top.D1.max_iterations)
        self.assertEqual(top.D2.sum, 4.*top.D2.max_iterations)
        
        top.C1.runcount = 0
        top.C2.runcount = 0
        top.D1.runcount = 0
        top.D2.runcount = 0
        top.D1.set('max_iterations', 5)
        top.D2.set('max_iterations', 4)
        exec_order = []
        top.run()
        self.assertEqual(top.D2.runcount, 1)
        self.assertEqual(top.D1.runcount, top.D2.max_iterations)
        self.assertEqual(top.C1.runcount, top.D2.max_iterations)
        self.assertEqual(top.C2.runcount, 
                         top.D2.max_iterations*top.D1.max_iterations)
        self.assertEqual(exec_order,
            ['D2', 'C1', 'D1', 'C2', 'C2', 'C2', 'C2', 'C2', 'C3', 
                        'C1', 'D1', 'C2', 'C2', 'C2', 'C2', 'C2', 'C3', 
                        'C1', 'D1', 'C2', 'C2', 'C2', 'C2', 'C2', 'C3', 
                        'C1', 'D1', 'C2', 'C2', 'C2', 'C2', 'C2', 'C3'])
        self.assertEqual(top.D1.sum, 3.*top.D1.max_iterations)
        self.assertEqual(top.D2.sum, 4.*top.D2.max_iterations)
        
        
    def test_2peer_drivers(self):
        #
        #  D1-->
        #  |   |
        #  |<--C1------>|
        #               |
        #          D2-->|
        #          |    |
        #          |<---C2
        
        global exec_order
        exec_order = []
        
        top = set_as_top(Assembly())
        top.add_container('C1', ExprComp2(expr='x+1'))
        top.add_container('C2', ExprComp2(expr='x+y'))
        top.add_driver('D1', Summer())
        top.add_driver('D2', Summer())
        
        top.connect('C1.f_xy', 'C2.x')
        top.D1.objective = 'C1.f_xy'
        top.D1.design = 'C1.y'
        top.D1.max_iterations = 2
        top.D2.objective = 'C2.f_xy'
        top.D2.design = 'C2.y'
        top.D2.max_iterations = 3
        top.run()
        self.assertEqual(top.D2.runcount, 1)
        self.assertEqual(top.D1.runcount, 1)
        self.assertEqual(top.C1.runcount, top.D1.max_iterations)
        self.assertEqual(top.C2.runcount, top.D2.max_iterations)
        self.assertEqual(exec_order,
                         ['D1', 'C1', 'C1', 
                          'D2', 'C2', 'C2', 'C2'])
        
        top.C1.runcount = 0
        top.C2.runcount = 0
        top.D1.runcount = 0
        top.D2.runcount = 0
        top.D1.set('max_iterations', 5)
        top.D2.set('max_iterations', 4)
        exec_order = []
        top.run()
        self.assertEqual(top.D2.runcount, 1)
        self.assertEqual(top.D1.runcount, 1)
        self.assertEqual(top.C1.runcount, top.D1.max_iterations)
        self.assertEqual(top.C2.runcount, top.D2.max_iterations)
        self.assertEqual(exec_order,
                         ['D1', 'C1', 'C1', 'C1', 'C1', 'C1', 
                          'D2', 'C2', 'C2', 'C2', 'C2'])
        
    def test_3nested_drivers(self):
        #
        #  D3-->
        #  |   |
        #  |   C1------> 
        #  |           |
        #  |       D2-->
        #  |       |   |
        #  |       |   C2------->
        #  |       |            |
        #  |       |       D1--->
        #  |       |        |   |
        #  |       |        |<--C3-->
        #  |       |                |
        #  |       |<---------------C4-->
        #  |                            |
        #  |<---------------------------C5
        
        global exec_order
        exec_order = []
        
        top = set_as_top(Assembly())
        top.add_container('C1', ExprComp(expr='x+1'))
        top.add_container('C2', ExprComp2(expr='x+y'))
        top.add_container('C3', ExprComp2(expr='x+1'))
        top.add_container('C4', ExprComp(expr='x+1'))
        top.add_container('C5', ExprComp(expr='x+1'))
        top.add_driver('D1', Summer())
        top.add_driver('D2', Summer())
        top.add_driver('D3', Summer())
        
        top.connect('C1.f_x', 'C2.x')
        top.connect('C2.f_xy', 'C3.x')
        top.connect('C3.f_xy', 'C4.x')
        top.connect('C4.f_x', 'C5.x')
        top.D1.objective = 'C3.f_xy'
        top.D1.design = 'C3.y'
        top.D1.max_iterations = 2
        top.D2.objective = 'C4.f_x'
        top.D2.design = 'C2.y'
        top.D2.max_iterations = 3
        top.D3.objective = 'C5.f_x'
        top.D3.design = 'C1.x'
        top.D3.max_iterations = 2
        top.run()
        self.assertEqual(top.D3.runcount, 1)
        self.assertEqual(top.D2.runcount, top.D3.max_iterations)
        self.assertEqual(top.D1.runcount, top.D3.max_iterations*top.D2.max_iterations)
        self.assertEqual(top.C3.runcount, 
                         top.D3.max_iterations*top.D2.max_iterations*top.D1.max_iterations)
        self.assertEqual(exec_order,
                         ['D3', 'C1', 'D2', 'C2', 'D1', 'C3', 'C3', 'C4', 
                                            'C2', 'D1', 'C3', 'C3', 'C4', 
                                            'C2', 'D1', 'C3', 'C3', 'C4', 'C5', 
                                'C1', 'D2', 'C2', 'D1', 'C3', 'C3', 'C4', 
                                            'C2', 'D1', 'C3', 'C3', 'C4', 
                                            'C2', 'D1', 'C3', 'C3', 'C4', 'C5'])
        
        """
                        ['D3', 'D2', 'C1', 'C2', 'D1', 'C3', 'C3', 'C4', 
                                           'C2', 'D1', 'C3', 'C3', 'C4', 
                                           'C2', 'D1', 'C3', 'C3', 'C4', 
                               'D2', 'C1', 'C2', 'D1', 'C3', 'C3', 'C4', 
                                           'C2', 'D1', 'C3', 'C3', 'C4', 
                                           'C2', 'D1', 'C3', 'C3', 'C4']
        """
        
        top.C3.runcount = 0
        top.D1.runcount = 0
        top.D2.runcount = 0
        top.D3.runcount = 0
        top.D1.set('max_iterations', 3)
        top.D2.set('max_iterations', 2)
        top.D3.set('max_iterations', 1)
        exec_order = []
        top.run()
        self.assertEqual(top.D3.runcount, 1)
        self.assertEqual(top.D2.runcount, top.D3.max_iterations)
        self.assertEqual(top.D1.runcount, top.D3.max_iterations*top.D2.max_iterations)
        self.assertEqual(top.C3.runcount, 
                         top.D3.max_iterations*top.D2.max_iterations*top.D1.max_iterations)
        self.assertEqual(exec_order,
                         ['D3', 'C1', 'D2', 'C2', 'D1', 'C3', 'C3', 'C3', 'C4', 
                                            'C2', 'D1', 'C3', 'C3', 'C3', 'C4', 'C5'])
        
if __name__ == "__main__":    
    unittest.main()


