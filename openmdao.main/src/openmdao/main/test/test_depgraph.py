# pylint: disable-msg=C0111,C0103

import unittest
import logging

from enthought.traits.api import Int, TraitError

from openmdao.main.api import Assembly, Component, set_as_top

def dump_stuff(obj, **kwargs):
    valid = kwargs.get('valid')
    io = kwargs.get('iotype')
    if io == 'in' or io == 'both':
        for name in obj.list_inputs(valid=valid):
            print '.'.join([obj.get_pathname(),name])
    if io == 'out' or io == 'both':
        for name in obj.list_outputs(valid=valid):
            print '.'.join([obj.get_pathname(),name])
    if kwargs.get('recurse'):
        for val in obj.values():
            if isinstance(val, Component):
                dump_stuff(val, **kwargs)
        
class Simple(Component):
    a = Int(iotype='in')
    b = Int(iotype='in')
    c = Int(iotype='out')
    d = Int(iotype='out')
    
    def __init__(self):
        super(Simple, self).__init__()
        self.a = 1
        self.b = 2
        self.c = 3
        self.d = -1
        self.run_count = 0

    def execute(self, required_outputs=None):
        self.run_count += 1
        self.c = self.a + self.b
        self.d = self.a - self.b

allcomps = ['sub.comp1','sub.comp2','sub.comp3','sub.comp4','sub.comp5','sub.comp6',
            'comp7','comp8']

topouts = ['sub.c2', 'sub.c4', 'sub.d1', 'sub.d3','sub.d5'
           'comp7.c', 'comp7.d','comp8.c', 'comp8.d']

topins = ['sub.a1', 'sub.a3', 'sub.b2', 'sub.b4','sub.b6'
          'comp7.a', 'comp7.b','comp8.a', 'comp8.b']

subins = ['comp1.a', 'comp1.b',
          'comp2.a', 'comp2.b',
          'comp3.a', 'comp3.b',
          'comp4.a', 'comp4.b',
          'comp5.a', 'comp5.b',
          'comp6.a', 'comp6.b',]

subouts = ['comp1.c', 'comp1.d',
           'comp2.c', 'comp2.d',
           'comp3.c', 'comp3.d',
           'comp4.c', 'comp4.d',
           'comp5.c', 'comp5.d',
           'comp6.c', 'comp6.d',]


subvars = subins+subouts

class DepGraphTestCase(unittest.TestCase):

    def setUp(self):
        print '***** setup'
        top = set_as_top(Assembly())
        self.top = top
        top.add_container('sub', Assembly())
        top.add_container('comp7', Simple())
        top.add_container('comp8', Simple())
        sub = top.sub
        sub.add_container('comp1', Simple())
        sub.add_container('comp2', Simple())
        sub.add_container('comp3', Simple())
        sub.add_container('comp4', Simple())
        sub.add_container('comp5', Simple())
        sub.add_container('comp6', Simple())

        sub.create_passthrough('comp1.a', 'a1')
        sub.create_passthrough('comp3.a', 'a3')
        sub.create_passthrough('comp2.b', 'b2')
        sub.create_passthrough('comp4.b', 'b4')
        sub.create_passthrough('comp6.b', 'b6')
        sub.create_passthrough('comp2.c', 'c2')
        sub.create_passthrough('comp4.c', 'c4')
        sub.create_passthrough('comp1.d', 'd1')
        sub.create_passthrough('comp3.d', 'd3')
        sub.create_passthrough('comp5.d', 'd5')
        
        sub.connect('comp1.c', 'comp4.a')
        sub.connect('comp5.c', 'comp1.b')
        sub.connect('comp2.d', 'comp5.b')
        sub.connect('comp3.c', 'comp5.a')
        sub.connect('comp4.d', 'comp6.a')
        
        top.connect('comp7.c', 'sub.a3')
        top.connect('sub.c4', 'comp8.a')
        top.connect('sub.d3', 'comp8.b')
        print '***** end setup'

    def test_simple(self):
        print '***** test_simple'
        top = set_as_top(Assembly())
        top.add_container('comp1', Simple())
        vars = ['a','b','c','d']
        self.assertEqual(top.comp1.run_count, 0)
        valids = [top.comp1.get_valid(v) for v in vars]
        # outputs are valid here because they are set in __init__
        self.assertEqual(valids, [True, True, False, False])
        top.run()
        self.assertEqual(top.comp1.run_count, 1)
        self.assertEqual(top.comp1.c, 3)
        self.assertEqual(top.comp1.d, -1)
        valids = [top.comp1.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, True, True, True])
        top.set('comp1.a', 5)
        valids = [top.comp1.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, True, False, False])
        top.run()
        self.assertEqual(top.comp1.run_count, 2)
        self.assertEqual(top.comp1.c, 7)
        self.assertEqual(top.comp1.d, 3)
        top.run()
        self.assertEqual(top.comp1.run_count, 2) # run_count shouldn't change
        valids = [top.comp1.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, True, True, True])
        
        # now add another comp and connect them
        top.add_container('comp2', Simple())
        top.connect('comp1.c', 'comp2.a')
        self.assertEqual(top.comp2.run_count, 0)
        self.assertEqual(top.comp2.c, 3)
        self.assertEqual(top.comp2.d, -1)
        valids = [top.comp2.get_valid(v) for v in vars]
        self.assertEqual(valids, [False, True, False, False])
        top.run()
        self.assertEqual(top.comp1.run_count, 2)
        self.assertEqual(top.comp2.run_count, 1)
        self.assertEqual(top.comp2.c, 9)
        self.assertEqual(top.comp2.d, 5)
        valids = [top.comp2.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, True, True, True])
        
        
    def test_lazy1(self):
        print '***** test_lazy1'
        self.top.run()
        run_counts = [self.top.get(x).run_count for x in allcomps]
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1], run_counts)
        outs = [(5,-3),(3,-1),(5,1),(7,3),(4,6),(5,1),(3,-1),(8,6)]
        newouts = []
        for comp in allcomps:
            newouts.append((self.top.get(comp+'.c'),self.top.get(comp+'.d')))
        self.assertEqual(outs, newouts)
        self.top.run()
        # run_count should stay at 1 for all comps
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1], 
                         [self.top.get(x).run_count for x in allcomps])
        
    def test_lazy2(self):
        print '***** test_lazy2'
        vars = ['a','b','c','d']
        self.top.run()        
        run_count = [self.top.get(x).run_count for x in allcomps]
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1], run_count)
        valids = [self.top.sub.comp6.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, True, True, True])
        self.top.sub.b6 = 3
        valids = [self.top.sub.comp6.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, False, False, False])
        self.top.run()
        # run_count should change only for comp6
        run_count = [self.top.get(x).run_count for x in allcomps]
        self.assertEqual([1, 1, 1, 1, 1, 2, 1, 1], run_count)
        outs = [(5,-3),(3,-1),(5,1),(7,3),(4,6),(6,0),(3,-1),(8,6)]
        for comp,vals in zip(allcomps,outs):
            self.assertEqual((comp,vals[0],vals[1]), 
                             (comp,self.top.get(comp+'.c'),self.top.get(comp+'.d')))
            
    def test_lazy3(self):
        print '***** test_lazy3'
        vars = ['a','b','c','d']
        self.top.run()        
        valids = [self.top.sub.comp3.get_valid(v) for v in vars]
        self.assertEqual(valids, [True, True, True, True])
        self.top.comp7.a = 3
        valids = [self.top.sub.comp3.get_valid(v) for v in vars]
        self.assertEqual(valids, [False, True, False, False])
        self.top.run()  
        # run_count should change for all sub comps but comp2
        run_count = [self.top.get(x).run_count for x in allcomps]
        self.assertEqual([2, 1, 2, 2, 2, 2, 2, 2], run_count)
        outs = [(7,-5),(3,-1),(7,3),(9,5),(6,8),(7,3),(5,1),(12,6)]
        for comp,vals in zip(allcomps,outs):
            self.assertEqual((comp,vals[0],vals[1]), 
                             (comp,self.top.get(comp+'.c'),self.top.get(comp+'.d')))
    
    def test_lazy4(self):
        print '***** test_lazy4'
        self.top.run()
        self.top.sub.set('b2', 5)
        self.top.run()  
        # run_count should change for all sub comps but comp3 and comp7 
        self.assertEqual([2, 2, 1, 2, 2, 2, 1, 2], 
                         [self.top.get(x).run_count for x in allcomps])
        outs = [(2,0),(6,-4),(5,1),(4,0),(1,9),(2,-2),(3,-1),(5,3)]
        for comp,vals in zip(allcomps,outs):
            self.assertEqual((comp,vals[0],vals[1]), 
                             (comp,self.top.get(comp+'.c'),self.top.get(comp+'.d')))
    
    def test_lazy_inside_out(self):
        print '***** test_lazy_inside_out'
        self.top.run()
        self.top.comp7.b = 4
        # now run sub.comp1 directly to make sure it will force
        # running of all components that supply its inputs
        self.top.sub.comp1.run()
        run_count = [self.top.get(x).run_count for x in allcomps]
        self.assertEqual([2, 1, 2, 1, 2, 1, 2, 1], run_count)
        outs = [(7,-5),(3,-1),(7,3),(7,3),(6,8),(5,1),(5,-3),(8,6)]
        for comp,vals in zip(allcomps,outs):
            self.assertEqual((comp,vals[0],vals[1]), 
                             (comp,self.top.get(comp+'.c'),self.top.get(comp+'.d')))
            
        # now run comp8 directly, which should force sub.comp4 to run
        self.top.comp8.run()
        run_count = [self.top.get(x).run_count for x in allcomps]
        self.assertEqual([2, 1, 2, 2, 2, 1, 2, 2], run_count)
        outs = [(7,-5),(3,-1),(7,3),(9,5),(6,8),(5,1),(5,-3),(12,6)]
        for comp,vals in zip(allcomps,outs):
            self.assertEqual((comp,vals[0],vals[1]), 
                             (comp,self.top.get(comp+'.c'),self.top.get(comp+'.d')))

    def test_set_already_connected(self):
        print '***** test_already_connected'
        try:
            self.top.sub.comp2.b = 4
        except TraitError, err:
            self.assertEqual(str(err), 
                "sub.comp2: 'b' is already connected to source 'b2' and cannot be directly set")
        else:
            self.fail('TraitError expected')
        try:
            self.top.set('sub.comp2.b', 4)
        except TraitError, err:
            self.assertEqual(str(err), 
                "sub.comp2: 'b' is connected to source 'b2' and cannot be set by source 'None'")
        else:
            self.fail('TraitError expected')            
            
        
if __name__ == "__main__":
    
    #import cProfile
    #cProfile.run('unittest.main()', 'profout')
    
    #import pstats
    #p = pstats.Stats('profout')
    #p.strip_dirs()
    #p.sort_stats('time')
    #p.print_stats()
    #print '\n\n---------------------\n\n'
    #p.print_callers()
    #print '\n\n---------------------\n\n'
    #p.print_callees()
        
    unittest.main()


