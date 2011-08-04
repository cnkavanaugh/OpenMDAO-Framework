from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import DOEdriver
from openmdao.lib.components.api import MetaModel
from openmdao.lib.casehandlers.api import DBCaseRecorder
from openmdao.lib.surrogatemodels.api import NeuralNetSurrogate
from openmdao.examples.expected_improvement.testfunc_component import TestFuncComponent

class Simulation(Assembly):
    def setUp(self):
        random.seed(10)
        
    def __init__(self):
        super(Simulation,self).__init__(self)
    
    #Components
    self.add("nn_meta_model",MetaModel())
    self.nn_meta_model.surrogate = {"default":NeuralNetSurrogate()}    
    self.nn_meta_model.model = TestFuncComponent()
    self.nn_meta_model.recorder = DBCaseRecorder(':memory:')
    self.nn_meta_model.force_execute = True
    
    #Driver Configuration
    self.add("driver",DOEdriver())
    self.driver.workflow.add("nn_meta_model")
    self.driver.sequential = True
    self.driver.DOEgenerator = Uniform()
    self.driver.num_samples = 500
    self.driver.add_parameter("nn_meta_model.x")
    self.driver.add_parameter("nn_meta_model.y")
    self.driver.add_event("nn_meta_model.train_next")
    self.driver.case_outputs = ["nn_meta_model.f_xy"]
    self.driver.recorder = DBCaseRecorder(os.path.join(self._tdir,'trainer.db'))
    
    def cleanup(self):
        shutil.rmtree(self._tdir, ignore_errors=True)
    
# if __name__ == "__main__":