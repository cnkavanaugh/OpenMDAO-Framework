"""A pyevolve based driver for OpenMDAO"""



import random

from enthought.traits.api import Int, Float, CBool, Any, \
                                 on_trait_change, TraitError

from pyevolve import G1DList, G1DBinaryString, G2DList, GAllele, GenomeBase
from pyevolve import GSimpleGA, Selectors, Initializators, Mutators, Consts
try:
    from pyevolve import DBAdapters
except ImportError:
    # Apparently the egg doesn't record it's dependencies.
    import logging
    logging.warning('No pyevolve.DBAdaptors available.')

from openmdao.main.api import Driver, StringRef

def G1DListCrossOverRealHypersphere(genome, **args):
    """ A genome reproduction algorithm, developed by Tristan Hearn at 
    the NASA Glenn Research Center, which uses a hypersphere defined
    by a pair of parents to find the space of possible children. 
    Children are then picked at random from that space. """
    
    gMom = args['mom']
    gDad = args['dad']
    
    sister = gMom.clone()
    brother = gDad.clone()
    
    #bounds = (genome.getParam("rangemin",0), genome.getParam("rangemax",100))
    bounds = (gMom.getParam("rangemin",0), gMom.getParam("rangemax",100))
    #dim = len(genome)
    dim = len(gMom)
    numparents = 2.0
        
    # find the center of mass (average value) between the two parents
    # for each dimension
    cmass = [(gm+gd)/2.0 for gm,gd in zip(gMom,gDad)]
    
    radius = max(sum([(cm-gM)**2 for cm,gM in zip(cmass,gMom)]),
                 sum([(cm-gD)**2 for cm,gD in zip(cmass,gDad)])
             )**.5
    
    #generate a random unit vectors in the hyperspace
    seed_sister = [random.uniform(-1,1) for i in range(0,dim)]
    magnitude = sum([x**2 for x in seed_sister])**.5
    #checksum to enforce a circular distribution of random numbers
    while magnitude > 1:
        seed_sister = [random.uniform(-1,1) for i in range(0,dim)]
        magnitude = sum([x**2 for x in seed_sister])**.5        
    
    seed_brother = [random.uniform(-1,1) for i in range(0,dim)]
    magnitude = sum([x**2 for x in seed_brother])**.5
    #checksum to enforce a circular distribution of random numbers
    while magnitude > 1:
        seed_brother = [random.uniform(-1,1) for i in range(0,dim)]
        magnitude = sum([x**2 for x in seed_brother])**.5    
    
    #create a children
    sister.resetStats()
    brother.resetStats()
    
    sister.genomeList = [cm+radius*sd for cm,sd in zip(cmass,seed_sister)]
    brother.genomeList = [cm+radius*sd for cm,sd in zip(cmass,seed_brother)]
    
    #preserve the integer type of the genome if necessary
    if type(gMom.genomeList[0]) == int:
        sister.genomeList = [int(round(x)) for x in sister.genomeList]   
        brother.genomeList = [int(round(x)) for x in brother.genomeList]  
    
    return (sister,brother)


class pyevolvedriver(Driver):
    """OpenMDAO wrapper for the pyevolve genetic algorithm framework. The 
    wrapper uses two attributes to configure the optmization: 
    
    The wrapper conforms to the pyevolve API for configuring the GA. It makes use 
    of two attributes which map to pyevolve objects: genome and GA. The default 
    selection for genome is G1DList. By default the wrapper uses the GsimpleGA engine 
    from pyevolve with all its default settings. Currently, only the default configuration for the 
    genome is supported. 

    The standard pyevolve library is provided:
        G1Dlist,G1DBinaryString,G2dList,GAllele
        GsimpleGA,Initializators,Mutators,Consts,DBadapters

    TODO: Implement function-slots as sockets
    """

    # inputs
    objective = StringRef(iotype='in',
                          desc='A string containing the objective function'
                               ' expression.')
    freq_stats = Int(0, iotype='in')
    seed = Float(0., iotype='in')
    population_size = Int(Consts.CDefGAPopulationSize, iotype='in')
    
    sort_type = CBool(Consts.sortType["scaled"], iotype='in',
                      desc='use Consts.sortType["raw"] or Consts.sortType["scaled"]') # can accept
    mutation_rate = Float(Consts.CDefGAMutationRate, iotype='in')
    crossover_rate = Float(Consts.CDefGACrossoverRate, iotype='in')
    generations = Int(Consts.CDefGAGenerations, iotype='in')
    mini_max = CBool(Consts.minimaxType["minimize"], iotype='in',
                    desc='use Consts.minimaxType["minimize"] or Consts.minimaxType["maximize"]')
    elitism = CBool(True, iotype='in', desc='True of False')
    
    #outputs
    best_individual = Any(GenomeBase.GenomeBase(), iotype='out')
        
    def __init__(self, doc=None): 
        super(pyevolvedriver,self).__init__(doc)

        self.genome = GenomeBase.GenomeBase() #TODO: Mandatory Socket
        self.GA = GSimpleGA.GSimpleGA(self.genome) #TODO: Mandatory Socket, with default plugin

        # value of None means use default
        self.decoder = None #TODO: mandatory socket       
        self.selector = None #TODO: optional socket
        self.stepCallback = None #TODO: optional socket
        self.terminationCriteria = None #TODO: optional socket
        self.DBAdapter = None #TODO: optional socket

    def _set_GA_FunctionSlot(self, slot, funcList, RandomApply=False,):
        if funcList is None: 
            return
        slot.clear()
        if not isinstance(funcList, list):
            funcList = [funcList]
        for func in funcList: 
            if slot.isEmpty(): 
                slot.set(func)
            else:
                slot.add(func)
        slot.setRandomApply(RandomApply)

    @on_trait_change('objective') 
    def _refvar_changed(self, obj, name, old, new):
        expr = getattr(obj, name)
        try:
            expr.refs_valid()  # force checking for existence of vars referenced in expression
        except (AttributeError, RuntimeError), err:
            self.raise_exception("invalid value '%s' for input ref variable '%s': %s" % 
                                 (str(expr), name, err), TraitError)
            
    def evaluate(self, genome):
        self.decoder(genome)
        self.run_iteration()
        return self.objective.evaluate()

    def verify(self):
        #genome verify
        if not isinstance(self.genome, GenomeBase.GenomeBase):
            self.raise_exception("genome provided is not valid."
                " Does not inherit from pyevolve.GenomeBase.GenomeBase",
                TypeError)

        #decoder verify
        if self.decoder is None: # check if None first
            self.raise_exception("decoder specified as 'None'."
                                 " A valid decoder must be present", TypeError)
        try: # won't work if decoder is None
            self.decoder(self.genome)
        except TypeError, err:
            self.raise_exception(
                "decoder as specified does not have the right signature. Must take only 1 argument: %s"%
                err, TypeError)

    def execute(self, required_outputs=None):
        """Perform the optimization"""
        self.verify()
        #configure the evaluator function of the genome
        self.genome.evaluator.set(self.evaluate)
        
        self.GA = GSimpleGA.GSimpleGA(self.genome, self.seed)
        
        self.GA.setPopulationSize(self.population_size)
        self.GA.setSortType(self.sort_type)
        self.GA.setMutationRate(self.mutation_rate)
        self.GA.setCrossoverRate(self.crossover_rate)
        self.GA.setGenerations(self.generations)
        self.GA.setMinimax(self.mini_max)
        self.GA.setElitism(self.elitism)

        #self.GA.setDBAdapter(self.DBAdapter) #
        
        self._set_GA_FunctionSlot(self.GA.selector, self.selector)
        self._set_GA_FunctionSlot(self.GA.stepCallback, self.stepCallback)
        self._set_GA_FunctionSlot(self.GA.terminationCriteria,
                                  self.terminationCriteria)
        
        self.GA.evolve(freq_stats=self.freq_stats)
        self.best_individual = self.GA.bestIndividual()

