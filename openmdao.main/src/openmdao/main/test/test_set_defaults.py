"""
Test for setting input variables back to their default values.
"""

import unittest

import numpy
from enthought.traits.api import Float, Array, List

from openmdao.main.api import Component, Assembly

class MyDefComp(Component):
    f_in = Float(3.14, io_direction='in')
    f_out = Float(io_direction='out')
    arr_in = Array(dtype=numpy.float, value=numpy.array([1.,2.,3.]), io_direction='in')
    list_in = List(value=['a','b','c'], io_direction='in')
    
    def execute(self):
        self.f_out = self.f_in + 1.
        
class MyNoDefComp(Component):
    f_in = Float(io_direction='in')
    f_out = Float(io_direction='out')
    arr_in = Array(dtype=numpy.float, io_direction='in')
    list_in = List(io_direction='in')
    
    def execute(self):
        self.f_out = self.f_in + 1.
        

class SetDefaultsTestCase(unittest.TestCase):

    def test_set_to_unset_default(self):
        comp = MyNoDefComp()
        self.assertEqual(0., comp.f_in)
        comp.f_in = 42.
        comp.arr_in = numpy.array([88., 32.])
        comp.list_in = [1,2,3]
        self.assertEqual(comp.get_valid('f_out'), False)
        comp.run()
        self.assertEqual(comp.get_valid('f_out'), True)
        comp.revert_to_defaults()
        # make sure reverting to defaults invalidates our outputs
        self.assertEqual(comp.get_valid('f_out'), False)
        self.assertEqual(0., comp.f_in)
        self.assertTrue(numpy.all(numpy.zeros(0,'d')==comp.arr_in))
        self.assertEqual([], comp.list_in)
    
    def test_set_to_default(self):
        comp = MyDefComp()
        self.assertEqual(3.14, comp.f_in)
        comp.f_in = 42.
        comp.arr_in = numpy.array([88., 32.])
        self.assertFalse(numpy.all(numpy.array([1.,2.,3.])==comp.arr_in))
        self.assertEqual(comp.get_valid('f_out'), False)
        comp.run()
        self.assertEqual(comp.get_valid('f_out'), True)
        comp.revert_to_defaults()
        # make sure reverting to defaults invalidates our outputs
        self.assertEqual(comp.get_valid('f_out'), False)
        self.assertEqual(3.14, comp.f_in)
        self.assertTrue(numpy.all(numpy.array([1.,2.,3.])==comp.arr_in))
        
    def test_set_recursive(self):
        asm = Assembly()
        asm.add_container('defcomp', MyDefComp())
        asm.add_container('nodefcomp', MyNoDefComp())
        self.assertEqual(0., asm.nodefcomp.f_in)
        self.assertEqual(3.14, asm.defcomp.f_in)
        asm.nodefcomp.f_in = 99
        asm.defcomp.f_in = 99
        asm.revert_to_defaults()
        self.assertEqual(0., asm.nodefcomp.f_in)
        self.assertEqual(3.14, asm.defcomp.f_in)
    
if __name__ == '__main__':
    unittest.main()

