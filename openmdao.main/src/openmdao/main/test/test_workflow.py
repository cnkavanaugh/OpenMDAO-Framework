"""
Test run/step/stop aspects of a simple workflow.
"""

import unittest

from enthought.traits.api import Bool, Int
from openmdao.main.api import Assembly, Component, set_as_top
from openmdao.main.exceptions import RunStopped

# pylint: disable-msg=E1101,E1103
# "Instance of <class> has no <attr> member"


class TestComponent(Component):
    """
    Component which tracks it's total executions
    and can request that the run be stopped.
    """

    dummy_input = Int(0, io_direction='in')
    set_stop = Bool(False, io_direction='in')
    total_executions = Int(0, io_direction='out')

    def execute(self, required_outputs=None):
        self.total_executions += 1
        if self.set_stop:
            self.parent.workflow.stop()


class Model(Assembly):
    """ Just a simple three-component workflow. """

    def __init__(self):
        super(Model, self).__init__()
        self.add_container('comp_a', TestComponent())
        self.add_container('comp_b', TestComponent())
        self.add_container('comp_c', TestComponent())

        self.connect('comp_a.total_executions', 'comp_b.dummy_input')
        self.connect('comp_b.total_executions', 'comp_c.dummy_input')

    def rerun(self):
        """ Called to force the model to run. """
        self.comp_a.set('dummy_input', 0)
        self.run()


class TestCase(unittest.TestCase):
    """ Test run/step/stop aspects of a simple workflow. """

    def setUp(self):
        """ Called before each test. """
        self.model = set_as_top(Model())

    def tearDown(self):
        """ Called after each test. """
        pass

    def test_simple(self):
        self.assertEqual(self.model.comp_a.total_executions, 0)
        self.assertEqual(self.model.comp_b.total_executions, 0)
        self.assertEqual(self.model.comp_c.total_executions, 0)

        self.model.run()

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 1)

        self.model.rerun()

        self.assertEqual(self.model.comp_a.total_executions, 2)
        self.assertEqual(self.model.comp_b.total_executions, 2)
        self.assertEqual(self.model.comp_c.total_executions, 2)

    def test_stepping(self):
        self.model.step()

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 0)
        self.assertEqual(self.model.comp_c.total_executions, 0)

        self.model.step()

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 0)

        self.model.step()

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 1)

        self.model.step() # this shouldn't run anything
        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 1)

    def test_run_stop_run(self):
        self.model.comp_b.set_stop = True
        try:
            self.model.run()
        except RunStopped, exc:
            self.assertEqual(str(exc), 'Stop requested')
        else:
            self.fail('Expected RunStopped')

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 0)

        self.model.comp_b.set_stop = False
        self.model.rerun()
        self.assertEqual(self.model.comp_a.total_executions, 2)
        self.assertEqual(self.model.comp_b.total_executions, 2)
        self.assertEqual(self.model.comp_c.total_executions, 1)

    def test_run_stop_step(self):
        self.model.comp_b.set_stop = True
        try:
            self.model.run()
        except RunStopped, exc:
            self.assertEqual(str(exc), 'Stop requested')
        else:
            self.fail('Expected RunStopped')

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 0)

        self.model.step()

        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 1)

        self.model.step() # shouldn't run anything
        self.assertEqual(self.model.comp_a.total_executions, 1)
        self.assertEqual(self.model.comp_b.total_executions, 1)
        self.assertEqual(self.model.comp_c.total_executions, 1)


if __name__ == '__main__':
    import nose
    import sys
    sys.argv.append('--cover-package=openmdao')
    sys.argv.append('--cover-erase')
    nose.runmodule()

