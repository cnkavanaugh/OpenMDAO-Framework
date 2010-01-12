#public symbols
__all__ = ["Driver"]




from enthought.traits.api import implements, List
from enthought.traits.trait_base import not_none
import networkx as nx
from networkx.algorithms.traversal import strongly_connected_components

from openmdao.main.interfaces import IDriver
from openmdao.main.component import Component
from openmdao.main.stringref import StringRef, StringRefArray

class Driver(Component):
    """ A Driver iterates over a collection of Components until some condition
    is met. """
    
    implements(IDriver)
    
    def __init__(self, doc=None):
        super(Driver, self).__init__(doc=doc)
        self._ref_graph = { None: None, 'in': None, 'out': None }
        self._ref_comps = { None: None, 'in': None, 'out': None }
        self.graph_regen_needed()
    
    def graph_regen_needed(self):
        """If called, reset internal graphs to that they will be
        regenerated when they are requested next.
        """
        self._iteration_comps = None
        self._simple_iteration_subgraph = None
        self._simple_iteration_set = None    
        self._driver_tree = None
        
    def _pre_execute (self):
        """Call base class _pre_execute after determining if we have any invalid
        ref variables, which will cause us to have to regenerate our ref dependency graph.
        """
        if self._call_execute:
            super(Driver, self)._pre_execute()
            return
        
        refnames = self.get_refvar_names(iostatus='in')
        
        if not all(self.get_valids(refnames)):
            self._call_execute = True
            # force regeneration of _ref_graph, _ref_comps, _iteration_comps
            self._ref_graph = { None: None, 'in': None, 'out': None } 
            self._ref_comps = { None: None, 'in': None, 'out': None }
            self.graph_regen_needed()
            
        super(Driver, self)._pre_execute()
        
        if not self._call_execute:
            # force execution of the driver if any of its StringRefs reference
            # invalid Variables
            for name in refnames:
                rv = getattr(self, name)
                if isinstance(rv, list):
                    for entry in rv:
                        if not entry.refs_valid():
                            self._call_execute = True
                            return
                else:
                    if not rv.refs_valid():
                        self._call_execute = True
                        return

    def execute(self):
        """ Iterate over a collection of Components until some condition
        is met. If you don't want to structure your driver to use pre_iteration,
        post_iteration, etc., just override this function. As a result, none
        of the <start/pre/post/continue>_iteration() functions will be called.
        """
        self.start_iteration()
        while self.continue_iteration():
            self.pre_iteration()
            self.run_iteration()
            self.post_iteration()

    def start_iteration(self):
        """Called just prior to the beginning of an iteration loop. This can 
        be overridden by inherited classes. It can be used to perform any 
        necessary pre-iteration initialization.
        """
        self._continue = True

    def continue_iteration(self):
        """Return False to stop iterating."""
        return self._continue
    
    def pre_iteration(self):
        """Called prior to each iteration."""
        pass
        
    def run_iteration(self):
        """Runs a single iteration of the workflow that this driver is associated with."""
        if self.parent:
            self.parent.workflow.run()
        else:
            self.raise_exception('Driver cannot run referenced components without a parent',
                                 RuntimeError)

    def post_iteration(self):
        """Called after each iteration."""
        self._continue = False  # by default, stop after one iteration

    def get_refvar_names(self, iostatus=None):
        """Return a list of names of all StringRef and StringRefArray traits
        in this instance.
        """
        if iostatus is None:
            checker = not_none
        else:
            checker = iostatus
        
        return [n for n,v in self._traits_meta_filter(iostatus=checker).items() 
                    if v.is_trait_type(StringRef) or 
                       v.is_trait_type(StringRefArray)]
        
    def get_referenced_comps(self, iostatus=None):
        """Return a set of names of Components that we reference based on the 
        contents of our StringRefs and StringRefArrays.  If iostatus is
        supplied, return only component names that are referenced by ref
        variables with matching iostatus.
        """
        if self._ref_comps[iostatus] is None:
            comps = set()
        else:
            return self._ref_comps[iostatus]
    
        for name in self.get_refvar_names(iostatus):
            obj = getattr(self, name)
            if isinstance(obj, list):
                for entry in obj:
                    comps.update(entry.get_referenced_compnames())
            else:
                comps.update(obj.get_referenced_compnames())
                
        self._ref_comps[iostatus] = comps
        return comps
        
    def get_ref_graph(self, iostatus=None):
        """Returns the dependency graph for this Driver based on
        StringRefs and StringRefArrays.
        """
        if self._ref_graph[iostatus] is not None:
            return self._ref_graph[iostatus]
        
        self._ref_graph[iostatus] = nx.DiGraph()
        name = self.name
        
        if iostatus == 'out' or iostatus is None:
            self._ref_graph[iostatus].add_edges_from([(name,rv) 
                                  for rv in self.get_referenced_comps(iostatus='out')])
            
        if iostatus == 'in' or iostatus is None:
            self._ref_graph[iostatus].add_edges_from([(rv, name) 
                                  for rv in self.get_referenced_comps(iostatus='in')])
        return self._ref_graph[iostatus]
