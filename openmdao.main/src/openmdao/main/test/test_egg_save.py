"""
Test saving and loading of simulations as eggs.
"""

import cPickle
import logging
import os.path
import pkg_resources
import shutil
import subprocess
import sys
import unittest

from enthought.traits.api import Bool, List, Str, Array, Int, \
                                 Instance, Callable, TraitError

from openmdao.main.api import Assembly, Component, Container, SAVE_PICKLE, \
                              SAVE_CPICKLE, set_as_top
from openmdao.main.filevar import FileMetadata, FileTrait

from openmdao.main.pkg_res_factory import PkgResourcesFactory

from openmdao.main.eggchecker import check_save_load
from openmdao.util.testutil import find_python

# pylint: disable-msg=E1101,E1103
# "Instance of <class> has no <attr> member"

EXTERNAL_FILES = ('xyzzy', '../sub/data2', 'hello', '../sub/data4')

SOURCE_INIT = False  # Used to verify __init__() gets called when expected.
SINK_INIT = False

# Various Pickle issues arise only when this test runs as the main module.
# This is used to detect when we're the main module or not.
MODULE_NAME = __name__

# Set local dir in case we're running in a different directory.
PY_DIR = pkg_resources.resource_filename('openmdao.main', 'test')

# Observations made by observer().
OBSERVATIONS = []


class Source(Assembly):
    """
    Produces files. A fair amount of stuff happens in Component.save_to_egg()
    in relation to handling external files and file variables.
    """

    write_files = Bool(True, iotype='in')
    text_data = Str(iotype='in')
    text_file = FileTrait(path='source.txt', iotype='out')

    def __init__(self, *args, **kwargs):
        super(Source, self).__init__(*args, **kwargs)

        global SOURCE_INIT
        SOURCE_INIT = True

        self.add_container('sub', Subcontainer())
        self.create_passthrough('sub.binary_file')

        # Some custom objects that must be restored.
        self.obj_list = [DataObj(i) for i in range(3)]

        # External file that doesn't exist at time of save.
        self.external_files.append(FileMetadata(path='does-not-exist'))

    def tree_rooted(self):
        """ Called once we have a valid hierarchy above us. """
        super(Source, self).tree_rooted()
        
        self.directory = self.get_abs_directory()  # Force absolute.
        # Absolute external file that exists at time of save.
        path = os.path.join(self.directory, EXTERNAL_FILES[0])
        out = open(path, 'w')
        out.write('Twisty narrow passages.\n')
        out.close()
        self.external_files.append(FileMetadata(path=path, input=True,
                                   constant=True))

        # Absolute external file that exists at time of save, in separate tree.
        path = os.path.join(self.directory, EXTERNAL_FILES[1])
        leaf = os.path.dirname(path)
        if not os.path.exists(leaf):
            os.makedirs(leaf)
        out = open(path, 'w')
        out.write('Some external data.\n')
        out.close()
        self.external_files.append(FileMetadata(path=path))

        # Relative external file that exists at time of save.
        self.push_dir(self.get_abs_directory())
        path = EXTERNAL_FILES[2]
        out = open(path, 'w')
        out.write('Hello world!\n')
        out.close()
        self.pop_dir()
        self.external_files.append(FileMetadata(path=path))

        # Relative external file that exists at time of save, in separate tree.
        self.push_dir(self.get_abs_directory())
        path = EXTERNAL_FILES[3]
        leaf = os.path.dirname(path)
        if not os.path.exists(leaf):
            os.makedirs(leaf)
        out = open(path, 'w')
        out.write('Some more external data.\n')
        out.close()
        self.pop_dir()
        self.external_files.append(FileMetadata(path=path))
    
    def execute(self, required_outputs=None):
        """ Write test data to files. """
        if self.write_files:
            cwd = os.getcwd()
            self.debug("opening file '%s' in %s" % 
                       (self.text_file.path, cwd))
            out = open(self.text_file.path, 'w')
            out.write(self.text_data)
            out.close()

            self.debug("opening file '%s' in %s" % 
                       (self.sub.binary_file.path, cwd))
            out = open(self.sub.binary_file.path, 'wb')
            cPickle.dump(self.sub.binary_data, out, 2)
            out.close()


class Subcontainer(Container):
    """ Just a subcontainer for Source. """

    binary_data = Array('d', value=[], iotype='in')
    binary_file = FileTrait(path=os.path.join('..', 'sub', 'source.bin'),
                            iotype='out', binary=True)
        

class DataObj(object):
    """ Just a custom class for objects to save & reload. """

    def __init__(self, data):
        self.data = data


class Sink(Component):
    """ Consumes files. """

    text_data = Str(iotype='out')
    binary_data = Array('d', value=[], iotype='out')
    text_file = FileTrait(iotype='in')
    binary_file = FileTrait(iotype='in')
    executions = Int(0, iotype='in',
                     desc='Count of Oddball instance_method() calls.')

    def __init__(self, *args, **kwargs):
        super(Sink, self).__init__(*args, **kwargs)
        global SINK_INIT
        SINK_INIT = True

    def execute(self, required_outputs=None):
        """ Read test data from files. """
        inp = self.text_file.open()
        self.text_data = inp.read()
        inp.close()

        inp = self.binary_file.open()
        self.binary_data = cPickle.load(inp)
        inp.close()


class Oddball(Assembly):
    """
    Just a component that needs a separate directory to be created.
    Also has some attributes used to exercise some Pickle issues.
    """

    # FIXME: I tried the built-in trait types of Callable, Method, and Function
    # for these two sockets and couldn't get them to work.  We may have to
    # create new TraitTypes for these...
    #function_socket = Instance(Callable, none_allowed=True,
    #                           desc='Just something to call.', required=False)
    #method_socket = Instance(Callable, none_allowed=True,
    #                         desc='Just something to call.', required=False)
    executions = Int(0, iotype='out', desc='Counts instance_method() calls.')

    def __init__(self, *args, **kwargs):
        super(Oddball, self).__init__(*args, **kwargs)
        self.add_container('oddcomp', OddballComponent())
        self.add_container('oddcont', OddballContainer())
        self.thing_to_call = self.instance_method
        self.list_to_call = [[self.instance_method, ()],
                             [Assembly.get_pathname, (self,)]]
                             # Generate IMHolder with self == None.
        self.function_socket = os.getpid
        self.method_socket = self.instance_method
        self.peer_class = Source  # Check that class in __main__ is handled.
        self.scratch_tuple = (1, 2)

    def execute(self, required_outputs=None):
        """ Call stuff. Empty sockets are clumsy. """
        if self.thing_to_call:
            self.debug('thing_to_call returned %s', self.thing_to_call())

        for thing, args in self.list_to_call:
            self.debug('list-thing returned %s', thing(*args))

        try:
            self.debug('function_socket returned %s', self.function_socket())
        except RuntimeError, exc:
            if not str(exc).find('empty'):
                raise exc
        try:
            self.debug('method_socket returned %s', self.method_socket())
        except RuntimeError, exc:
            if not str(exc).find('empty'):
                raise exc

    def instance_method(self):
        """ Called by execute(). """
        self.executions += 1
        return self.executions

    @staticmethod
    def static_method():
        """ This won't pickle. """
        return None


class OddballComponent(Component):
    """ Just a subcomponent for Oddball to test nested entry points. """

    def __init__(self):
        super(OddballComponent, self).__init__()
        # Some custom objects that must be restored.
        self.obj_list = [DataObj(i) for i in range(3)]


class OddballContainer(Container):
    """ Just a subcontainer for Oddball to test nested entry points. """

    def __init__(self):
        super(OddballContainer, self).__init__()
        # Some custom objects that must be restored.
        self.obj_list = [DataObj(i) for i in range(3)]


def observer(state, string, file_fraction, byte_fraction):
    """ Observe progress. """
    if state != 'analyze':  # 'analyze' is sporadic due to re-use of analyses.
        OBSERVATIONS.append((state, string, file_fraction, byte_fraction))
    return True


class Model(Assembly):
    """ Transfer files from producer to consumer. """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.add_container('Source', Source(directory='Source'))
        self.add_container('Oddball', Oddball(directory='Oddball'))
        self.add_container('Sink', Sink(directory='Sink'))

        self.connect('Source.text_file', 'Sink.text_file')
        self.connect('Source.binary_file', 'Sink.binary_file')

        self.connect('Oddball.executions', 'Sink.executions')

        self.Source.text_data = 'oiuyoiuyoiuy'
        self.Source.sub.binary_data = [3.14159, 2.781828, 42]


class TestCase(unittest.TestCase):
    """ Test saving and loading of simulations as eggs. """

    def setUp(self):
        """ Called before each test in this class. """
        self.model = set_as_top(Model(directory='Egg'))
        self.model.name = 'Egg_TestModel'
        self.child_objs = [self.model.Source, self.model.Sink,
                           self.model.Oddball, self.model.Oddball.oddcomp,
                           self.model.Oddball.oddcont]
        self.egg_name = None

    def tearDown(self):
        """ Called after each test in this class. """
        self.model.pre_delete()  # Paranoia.  Only needed by NPSS I think.
        self.model = None
        if self.egg_name and os.path.exists(self.egg_name):
            os.remove(self.egg_name)
        if os.path.exists('Egg'):
            shutil.rmtree('Egg')

    def save_load(self, fmt, use_setuptools=False):
        """
        Save to egg using `fmt` and reload.
        Writes egg natively or via setuptools.
        """
        global SOURCE_INIT, SINK_INIT

        # Verify initial state.
        self.assertEqual(SOURCE_INIT, True)
        self.assertEqual(SINK_INIT, True)
        self.assertNotEqual(self.model.Sink.text_data,
                            self.model.Source.text_data)
        self.assertNotEqual(self.model.Sink.binary_data,
                            self.model.Source.sub.binary_data)

        for path in EXTERNAL_FILES:
            path = os.path.join(self.model.Source.get_abs_directory(), path)
            if not os.path.exists(path):
                self.fail("pre-save path '%s' does not exist" % path)

        for i in range(3):
            self.assertEqual(self.model.Source.obj_list[i].data, i)

        self.assertEqual(self.model.Sink.executions, 0)

        # Save to egg.
        global OBSERVATIONS
        OBSERVATIONS = []
        egg_info = self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                          fmt=fmt, child_objs=self.child_objs,
                                          use_setuptools=use_setuptools,
                                          observer=observer)
        self.egg_name = egg_info[0]

        # Check observations.
        if use_setuptools:
            expected = [
                ('add', 'write-via-setuptools'),
                ('complete', 'Egg_TestModel-1.2.3-py%d.%d.egg' % sys.version_info[:2]),
            ]
        else:
            expected = [
                ('add', 'EGG-INFO/PKG-INFO'),
                ('add', 'EGG-INFO/dependency_links.txt'),
                ('add', 'EGG-INFO/entry_points.txt'),
                ('add', 'EGG-INFO/not-zip-safe'),
                ('add', 'EGG-INFO/requires.txt'),
                ('add', 'EGG-INFO/openmdao_orphans.txt'),
                ('add', 'EGG-INFO/top_level.txt'),
                ('add', 'EGG-INFO/SOURCES.txt'),
                ('add', 'Egg_TestModel/Egg_TestModel.pickle'),
                ('add', 'Egg_TestModel/Egg_TestModel_loader.py'),
                ('add', 'Egg_TestModel/Oddball.pickle'),
                ('add', 'Egg_TestModel/Oddball_loader.py'),
                ('add', 'Egg_TestModel/Oddball_oddcomp.pickle'),
                ('add', 'Egg_TestModel/Oddball_oddcomp_loader.py'),
                ('add', 'Egg_TestModel/Oddball_oddcont.pickle'),
                ('add', 'Egg_TestModel/Oddball_oddcont_loader.py'),
                ('add', 'Egg_TestModel/Sink.pickle'),
                ('add', 'Egg_TestModel/Sink_loader.py'),
                ('add', 'Egg_TestModel/Source.pickle'),
                ('add', 'Egg_TestModel/Source/hello'),
                ('add', 'Egg_TestModel/Source/xyzzy'),
                ('add', 'Egg_TestModel/Source_loader.py'),
                ('add', 'Egg_TestModel/__init__.py'),
                ('add', 'Egg_TestModel/buildout.cfg'),
                ('add', 'Egg_TestModel/sub/data2'),
                ('add', 'Egg_TestModel/sub/data4'),
            ]

            # Add our file if we're not considered part of an egg.
            if sys.modules[self.__module__].__file__.find('.egg') < 0:
                expected.append(('add', 'Egg_TestModel/test_egg_save.py'))
            expected.append(('complete', 'Egg_TestModel-1.2.3-py%d.%d.egg' % sys.version_info[:2]))

        self.assertEqual(len(OBSERVATIONS), len(expected))
        for i, observation in enumerate(OBSERVATIONS):
            state, string, file_fraction, byte_fraction = observation
            self.assertEqual(state, expected[i][0])
            if expected[i][1].endswith('.egg'): # Unique versions mess this up.
                self.assertEqual(string.startswith(self.model.name), True)
                self.assertEqual(string.endswith('.egg'), True)
            else:
                self.assertEqual(string.replace('\\','/'), expected[i][1])
            self.assertEqual(file_fraction, float(i)/float(len(expected)-1))

        # Run and verify correct operation.
        self.model.run()
        self.assertEqual(self.model.Sink.text_data,
                         self.model.Source.text_data)
        self.assertEqual(True,
            all(self.model.Sink.binary_data==self.model.Source.sub.binary_data))
        self.assertEqual(self.model.Sink.binary_file.binary, True)

        self.assertEqual(self.model.Sink.executions, 3)

        # Restore in test directory.
        orig_dir = os.getcwd()
        test_dir = 'EggTest'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        os.chdir(test_dir)
        try:
            # Clear flags to detect if loading calls __init__.
            SOURCE_INIT = False
            SINK_INIT = False

            # Load from saved initial state in egg.
            self.model.pre_delete()
            egg_path = os.path.join('..', self.egg_name)
            OBSERVATIONS = []
            self.model = Component.load_from_eggfile(egg_path, install=False,
                                                     observer=observer)
            self.model.directory = os.path.join(os.getcwd(), self.model.name)

            # Check observations.
            expected = [
                ('extract', 'EGG-INFO/PKG-INFO'),
                ('extract', 'EGG-INFO/dependency_links.txt'),
                ('extract', 'EGG-INFO/entry_points.txt'),
                ('extract', 'EGG-INFO/not-zip-safe'),
                ('extract', 'EGG-INFO/requires.txt'),
                ('extract', 'EGG-INFO/openmdao_orphans.txt'),
                ('extract', 'EGG-INFO/top_level.txt'),
                ('extract', 'EGG-INFO/SOURCES.txt'),
                ('extract', 'Egg_TestModel/Egg_TestModel.pickle'),
                ('extract', 'Egg_TestModel/Egg_TestModel_loader.py'),
                ('extract', 'Egg_TestModel/Oddball.pickle'),
                ('extract', 'Egg_TestModel/Oddball_loader.py'),
                ('extract', 'Egg_TestModel/Oddball_oddcomp.pickle'),
                ('extract', 'Egg_TestModel/Oddball_oddcomp_loader.py'),
                ('extract', 'Egg_TestModel/Oddball_oddcont.pickle'),
                ('extract', 'Egg_TestModel/Oddball_oddcont_loader.py'),
                ('extract', 'Egg_TestModel/Sink.pickle'),
                ('extract', 'Egg_TestModel/Sink_loader.py'),
                ('extract', 'Egg_TestModel/Source.pickle'),
                ('extract', 'Egg_TestModel/Source/hello'),
                ('extract', 'Egg_TestModel/Source/xyzzy'),
                ('extract', 'Egg_TestModel/Source_loader.py'),
                ('extract', 'Egg_TestModel/__init__.py'),
                ('extract', 'Egg_TestModel/buildout.cfg'),
                ('extract', 'Egg_TestModel/sub/data2'),
                ('extract', 'Egg_TestModel/sub/data4'),
            ]

            # Add our file if we're not considered part of an egg.
            if sys.modules[self.__module__].__file__.find('.egg') < 0:
                expected.append(('extract', 'Egg_TestModel/test_egg_save.py'))
            expected.append(('complete', None))

            self.assertEqual(len(OBSERVATIONS), len(expected))
            if use_setuptools:  # No control on order, so sort on name.
                expected.sort(key=lambda item: item[1])
                OBSERVATIONS.sort(key=lambda item: item[1])
            for i, observation in enumerate(OBSERVATIONS):
                state, string, file_fraction, byte_fraction = observation
                self.assertEqual(state,  expected[i][0])
                self.assertEqual(string, expected[i][1])
                if not use_setuptools:  # Sort messes-up this comparison.
                    self.assertEqual(file_fraction,
                                     float(i)/float(len(expected)-1))

            # Verify initial state.
            self.assertEqual(SOURCE_INIT, False)
            self.assertEqual(SINK_INIT, False)
            self.assertNotEqual(self.model.Sink.text_data,
                                self.model.Source.text_data)
            self.assertNotEqual(self.model.Sink.binary_data,
                                self.model.Source.sub.binary_data)

            for path in EXTERNAL_FILES:
                path = os.path.join(self.model.Source.get_abs_directory(), path)
                if not os.path.exists(path):
                    self.fail("after loading, path '%s' does not exist" % path)

            for i in range(3):
                self.assertEqual(self.model.Source.obj_list[i].data, i)

            self.assertEqual(self.model.Oddball.executions, 0)

            # Run and verify correct operation.
            self.model.run()
            self.assertEqual(self.model.Sink.text_data,
                             self.model.Source.text_data)
            self.assertEqual(all(self.model.Sink.binary_data==
                             self.model.Source.sub.binary_data), True)
            self.assertEqual(
                self.model.Sink.binary_file.binary, True)

            self.assertEqual(self.model.Oddball.executions, 3)

        finally:
            os.chdir(orig_dir)
            shutil.rmtree(test_dir)

    def test_save_load_cpickle(self):
        logging.debug('')
        logging.debug('test_save_load_cpickle')
        self.save_load(SAVE_CPICKLE)

    def test_save_load_pickle(self):
        logging.debug('')
        logging.debug('test_save_load_pickle')
        self.save_load(SAVE_PICKLE)

# Fails to load. It appears you can't have more than one level of
# back-pointers when loading YAML. (A component works, but an assembly doesn't)
#    def test_save_load_yaml(self):
#        logging.debug('')
#        logging.debug('test_save_load_yaml')
#        self.save_load(SAVE_LIBYAML)

    def test_save_with_setuptools(self):
        logging.debug('')
        logging.debug('test_save_with_setuptools')
        self.save_load(SAVE_CPICKLE, use_setuptools=True)

    def test_save_bad_name(self):
        logging.debug('')
        logging.debug('test_save_bad_name')
        try:
            self.model.save_to_egg('#%^&', '0', py_dir=PY_DIR)
        except ValueError, exc:
            msg = 'Egg_TestModel: Egg name must be alphanumeric'
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected ValueError')

    def test_save_bad_version(self):
        logging.debug('')
        logging.debug('test_save_bad_version')
        try:
            self.model.save_to_egg(self.model.name, '#%^&', py_dir=PY_DIR)
        except ValueError, exc:
            msg = 'Egg_TestModel: Egg version must be alphanumeric'
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected ValueError')

    def test_save_bad_directory(self):
        logging.debug('')
        logging.debug('test_save_bad_directory')

        # Set subcomponent directory outside model root.
        self.model.Oddball.directory = os.getcwd()
        try:
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR)
        except ValueError, exc:
            msg = "Egg_TestModel: Can't save, Egg_TestModel.Oddball.oddcomp" \
                  " directory"
            self.assertEqual(str(exc)[:len(msg)], msg)
        else:
            self.fail('Expected ValueError')

    def test_save_bad_destination(self):
        logging.debug('')
        logging.debug('test_save_bad_destination')
        try:
            # Attempt to save to directory we aren't allowed to write to.
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                   dst_dir='/')
        except IOError, exc:
            #msg = "Egg_TestModel: Can't save to '/', no write permission"
            #self.assertEqual(str(exc), msg)
            self.assertTrue('no write permission' in str(exc) or 
                            'Permission denied' in str(exc))
        else:
            self.fail('Expected IOError')

    def test_save_bad_external(self):
        logging.debug('')
        logging.debug('test_save_bad_external')

        # Set external file path outside model root.
        path = os.path.join(os.getcwd(), 'bad-external')
        out = open(path, 'w')
        out.close()
        metadata = self.model.Source.external_files[0]
        metadata.path = path
        try:
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR)
        except Exception, exc:
            msg = "Egg_TestModel: Can't save, Egg_TestModel.Source file"
            self.assertEqual(str(exc)[:len(msg)], msg)
        else:
            self.fail('Expected Exception')
        finally:
            os.remove(path)

    def test_save_noforce(self):
        logging.debug('')
        logging.debug('test_save_noforce')

        # Set external file path outside model root.
        path = os.path.join(os.getcwd(), 'unforced-external')
        out = open(path, 'w')
        out.close()
        metadata = self.model.Source.external_files[0]
        metadata.path = path
        try:
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                   require_relpaths=False)
        finally:
            os.remove(path)

    def test_save_bad_filevar(self):
        logging.debug('')
        logging.debug('test_save_bad_filevar')

        # Set file trait path outside model root.
        self.model.Source.text_file.path = '/illegal'
        try:
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR)
        except ValueError, exc:
            msg = "Egg_TestModel: Can't save, Egg_TestModel.Source.text_file" \
                  " path"
            self.assertEqual(str(exc)[:len(msg)], msg)
        else:
            self.fail('Expected ValueError')

    def test_save_bad_format(self):
        logging.debug('')
        logging.debug('test_save_bad_format')
        try:
            # Attempt to save in unknown format.
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                   fmt='unknown')
        except RuntimeError, exc:
            self.assertEqual(str(exc),
                             "Egg_TestModel: Unknown format 'unknown'.")
        else:
            self.fail('Expected RuntimeError')

    def test_save_bad_function(self):
        logging.debug('')
        logging.debug('test_save_bad_function')

        # Set reference to unpickleable function.
        self.model.Oddball.function_socket = observer
        try:
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR)
        except RuntimeError, exc:
            msg = "Egg_TestModel: Can't save: reference to function defined" \
                  " in main module"
            self.assertEqual(str(exc)[:len(msg)], msg)
        else:
            if MODULE_NAME == '__main__':
                self.fail('Expected RuntimeError')

    def test_save_bad_method(self):
        logging.debug('')
        logging.debug('test_save_bad_method')

        # Set reference to unpickleable static method.
        self.model.Oddball.method_socket = self.model.Oddball.static_method
        try:
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR)
        except RuntimeError, exc:
            msg = "Egg_TestModel: Can't save, 1 object cannot be pickled."
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected RuntimeError')

    def test_save_bad_pickle(self):
        logging.debug('')
        logging.debug('test_save_bad_pickle')

        # Code objects don't pickle.
        self.model.code = compile('3 + 4', '<string>', 'eval')

        # Problem was deletion of existing buildout.cfg.
        if os.path.exists('buildout.cfg'):
            buildout_size = os.path.getsize('buildout.cfg')
            remove_buildout = False
        else:
            out = open('buildout.cfg', 'w')
            out.close()
            buildout_size = 0
            remove_buildout = True

        try:
            try:
                # This will fail due to code object.
                self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR)
            except cPickle.PicklingError, exc:
                msg = "Egg_TestModel: Can't save to" \
                      " 'Egg_TestModel/Egg_TestModel.pickle': Can't pickle" \
                      " <type 'code'>: attribute lookup __builtin__.code failed"
                self.assertEqual(str(exc).replace('\\','/'), msg)
            else:
                self.fail('Expected cPickle.PicklingError')

            # Verify buildout.cfg hasn't been affected.
            self.assertTrue(os.path.exists('buildout.cfg'))
            self.assertEqual(os.path.getsize('buildout.cfg'), buildout_size)
        finally:
            if remove_buildout:
                os.remove('buildout.cfg')
                
    def test_save_bad_child(self):
        logging.debug('')
        logging.debug('test_save_bad_child')

        # Create orphan component.
        orphan = Component()
        try:
            # Try to include orphan as an entry point in egg.
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                   child_objs=[orphan])
        except RuntimeError, exc:
            self.assertEqual(str(exc), 'Entry point object has no parent!')
        else:
            self.fail('Expected RuntimeError')

        # Create non-orphan component that is not part of model.
        badboy = orphan.add_container('badboy', Component())
        try:
            # Try to include non-member component as an entry point in egg.
            self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                   child_objs=[badboy])
        except RuntimeError, exc:
            msg = 'Egg_TestModel: badboy is not a child of' \
                  ' Egg_TestModel.'
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected RuntimeError')

    def test_save_load_container(self):
        logging.debug('')
        logging.debug('test_save_load_container')

        # Save to egg.
        egg_info = self.model.Source.sub.save_to_egg(self.model.name, '0',
                                                     py_dir=PY_DIR)
        self.egg_name = egg_info[0]

        # Restore in test directory.
        orig_dir = os.getcwd()
        test_dir = 'EggTest'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        os.chdir(test_dir)
        try:
            egg_path = os.path.join('..', self.egg_name)
            sub = Container.load_from_eggfile(egg_path, install=False)
            self.assertTrue(all(sub.binary_data == self.model.Source.sub.binary_data))
        finally:
            os.chdir(orig_dir)
            shutil.rmtree(test_dir)

    def test_load_badfile(self):
        logging.debug('')
        logging.debug('test_load_badfile')
        try:
            # Try to load from non-egg.
            Component.load_from_eggfile('.')
        except ValueError, exc:
            self.assertEqual(str(exc), "'.' is not an egg/zipfile.")
        else:
            self.fail('Expected ValueError')

    def test_load_nofile(self):
        logging.debug('')
        logging.debug('test_load_nofile')
        try:
            # Try to load from nonexistant egg file.
            Component.load_from_eggfile('no-such-egg')
        except ValueError, exc:
            self.assertEqual(str(exc), "'no-such-egg' not found.")
        else:
            self.fail('Expected ValueError')

    def test_load_nopackage(self):
        logging.debug('')
        logging.debug('test_load_nopackage')
        try:
            # Try to load from nonexistant egg package.
            Component.load_from_eggpkg('no-such-egg')
        except pkg_resources.DistributionNotFound, exc:
            self.assertEqual(str(exc), 'no-such-egg')
        else:
            self.fail('Expected pkg_resources.DistributionNotFound')

    def test_check_save_load(self):
        logging.debug('')
        logging.debug('test_check_save_load')

        # Exercise check_save_load().
        retcode = check_save_load(self.model, py_dir=PY_DIR)
        self.assertEqual(retcode, 0)

    def test_install_load(self):
        # Creates egg.
        # Installs in special directory.
        # Tries to load and run from installed egg in various ways.
        logging.debug('')
        logging.debug('test_install_load')

        # Find correct python.
        python = find_python()
        logging.debug('    Using python: %s' % python)

        # Write to egg.
        egg_info = self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                          child_objs=self.child_objs)
        self.egg_name = egg_info[0]

        # Create directory for installation.
        install_dir = os.path.join(os.getcwd(), 'install_dir')
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir)
        os.mkdir(install_dir)
        try:
            # Create special installer script.
            # This basically does an easy_install.
            installer = os.path.join(install_dir, 'installer.py')
            out = open(installer, 'w')
            out.write("""\
# EASY-INSTALL-ENTRY-SCRIPT: 'setuptools>=0.6c8','console_scripts','easy_install'
__requires__ = 'setuptools>=0.6c8'
import sys
from pkg_resources import load_entry_point

sys.exit(
   load_entry_point('setuptools>=0.6c8', 'console_scripts', 'easy_install')()
)
""")
            out.close()

            # Install via subprocess with PYTHONPATH set (for easy_install).
            logging.debug('Installing via subprocess...')
            env = os.environ.copy()
            path = env.get('PYTHONPATH', '')
            if path:
                path += os.pathsep
            path += install_dir
            env['PYTHONPATH'] = path
            cmdline = '%s %s -d %s %s' % \
                      (python, installer, install_dir, self.egg_name)
            stdout = open(os.path.join(install_dir, 'installer.out'), 'w')
            retcode = subprocess.call(cmdline, env=env, shell=True,
                                      stdout=stdout, stderr=subprocess.STDOUT)
            stdout.close()
            stdout = open(os.path.join(install_dir, 'installer.out'), 'r')
            for line in stdout:
                logging.debug('    %s', line.rstrip())
            stdout.close()
            self.assertEqual(retcode, 0)

            # Load full model and run.
            package_name = self.model.name
            entry_name = ''
            retcode = self.load_n_run(python, install_dir,
                                      package_name, entry_name)
            self.assertEqual(retcode, 0)

            # Load just the Oddball component and run.
            entry_name = self.model.Oddball.get_pathname()
            retcode = self.load_n_run(python, install_dir,
                                      package_name, entry_name)
            self.assertEqual(retcode, 0)

            # Try a non-existent package.
            try:
                Component.load_from_eggpkg('no-such-pkg', 'no-such-entry')
            except pkg_resources.DistributionNotFound, exc:
                self.assertEqual(str(exc), 'no-such-pkg')
            else:
                self.fail('Expected DistributionNotFound')
            
            # Try a non-existent entry point.
            egg_path = os.path.join(install_dir, self.egg_name)
            sys.path.append(egg_path)
            orig_ws = pkg_resources.working_set
            pkg_resources.working_set = pkg_resources.WorkingSet()
            try:
                Component.load_from_eggpkg(package_name, 'no-such-entry')
            except RuntimeError, exc:
                msg = "No 'openmdao.component' 'no-such-entry' entry point."
                self.assertEqual(str(exc), msg)
            else:
                self.fail('Expected RuntimeError')
            finally:
                sys.path.pop()
                pkg_resources.working_set = orig_ws

        finally:
            shutil.rmtree(install_dir)

    def load_n_run(self, python, install_dir, package_name, entry_name):
        """ Load component from installed egg and run it. """
        # Create and move to test directory.
        orig_dir = os.getcwd()
        test_dir = 'EggTest'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        os.chdir(test_dir)
        try:
            # Create load-n-run script.
            out = open('load-n-run.py', 'w')
            out.write("""\
import sys
sys.path.append('%(egg)s')
from openmdao.main.api import Component
comp = Component.load_from_eggpkg('%(package)s', '%(entry)s')
comp.run()
    
""" % {'egg':os.path.join(install_dir, self.egg_name).replace('\\', '/'),
       'package':package_name, 'entry':entry_name})
            out.close()

            # Load & run in subprocess.
            logging.debug("Load and run '%s' in subprocess...", entry_name)
            logging.debug('    %s', os.path.join(install_dir, self.egg_name))
            cmdline = '%s load-n-run.py' % python
            stdout = open('load-n-run.out', 'w')
            retcode = subprocess.call(cmdline, shell=True, stdout=stdout,
                                      stderr=subprocess.STDOUT)
            stdout.close()
            stdout = open('load-n-run.out', 'r')
            for line in stdout:
                logging.debug('    %s'% line.rstrip())
            stdout.close()
            return retcode

        finally:
            os.chdir(orig_dir)
            shutil.rmtree(test_dir)

    def test_pkg_resources_factory(self):
        # NOTE: this test fails if run standalone:
        #       ImportError: No module named test_egg_save
        # Probably need Egg_TestModel.test_egg_save, or adjusted sys.path.
        logging.debug('')
        logging.debug('test_pkg_resources_factory')

        # Write to egg.
        egg_info = self.model.save_to_egg(self.model.name, '0', py_dir=PY_DIR,
                                          child_objs=self.child_objs)
        self.egg_name = egg_info[0]

        # Create factory.
        factory = PkgResourcesFactory(['openmdao.component',
                                       'openmdao.container'],
                                      [os.getcwd()])
        logging.debug('    loaders:')
        for key, value in factory._loaders.items():
            logging.debug('        %s:', key)
            for val in value:
                logging.debug('                name: %s', val.name)
                logging.debug('               group: %s', val.group)
                logging.debug('                dist: %s', val.dist)
                logging.debug('            entry_pt: %s', val.entry_pt)
                logging.debug('                ctor: %s', val.ctor)
                logging.debug('')

        # Create and move to test directory.
        orig_dir = os.getcwd()
        test_dir = 'EggTest'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        os.chdir(test_dir)
        try:
            # Check multiple model instances.
            self.create_and_check_model(factory, 'test_model_1',
                                        'Hello world!\n')
            self.create_and_check_model(factory, 'test_model_2',
                                        'Hello world!\n')

            # Check that reloading doesn't clobber existing files.
            file_data = 'New and interesting stuff\n'
            path = os.path.join('test_model_2', 'Source', EXTERNAL_FILES[2])
            out = open(path, 'w')
            out.write(file_data)
            out.close()
            logging.debug('updated %s', path)
            self.create_and_check_model(factory, 'test_model_2', file_data)

            # Check observations.
            global OBSERVATIONS
            OBSERVATIONS = []
            model = factory.create('Egg_TestModel', name='observed',
                                   observer=observer)
            if model is None:
                self.fail("Create of 'observed' failed.")
            expected = [
                ('copy', 'Source/xyzzy'),
                ('copy', 'sub/data2'),
                ('copy', 'Source/hello'),
                ('copy', 'sub/data4'),
                ('complete', 'observed'),
            ]
            self.assertEqual(len(OBSERVATIONS), len(expected))
            for i, observation in enumerate(OBSERVATIONS):
                state, string, file_fraction, byte_fraction = observation
                self.assertEqual(state,  expected[i][0])
                self.assertEqual(string, expected[i][1])
                self.assertEqual(file_fraction, float(i)/float(len(expected)-1))

            # Create a component.
            comp = factory.create('Egg_TestModel.Oddball', name='test_comp',
                                  observer=observer)
            if comp is None:
                self.fail('Create of test_comp failed.')
            self.assertEqual(comp.get_pathname(), 'test_comp')
            self.assertEqual(comp.executions, 0)
            comp.run()
            self.assertEqual(comp.executions, 3)

            # Create a (sub)component.
            sub = factory.create('Egg_TestModel.Oddball.oddcomp',
                                 name='test_sub')
            if sub is None:
                self.fail('Create of test_sub failed.')
            self.assertEqual(sub.get_pathname(), 'test_sub')

            # Create a (sub)container.
            sub = factory.create('Egg_TestModel.Oddball.oddcont',
                                 name='test_sub')
            if sub is None:
                self.fail('Create of test_sub failed.')
            self.assertEqual(sub.get_pathname(), 'test_sub')

            # Try a non-existent entry point.
            obj = factory.create('no-such-entry', name='xyzzy')
            self.assertEqual(obj, None)

        finally:
            os.chdir(orig_dir)
            shutil.rmtree(test_dir)

    def create_and_check_model(self, factory, name, file_data):
        """ Create a complete model instance and check it's operation. """
        model = factory.create('Egg_TestModel', name=name)
        logging.debug('model.directory = %s' % model.directory)
        if model is None:
            self.fail("Create of '%s' failed." % name)
        self.assertEqual(model.get_abs_directory(),
                         os.path.join(os.getcwd(), name))
        self.assertEqual(model.Oddball.get_pathname(), name+'.Oddball')

        # Verify initial state.
        self.assertNotEqual(model.Sink.text_data,
                            model.Source.text_data)
        self.assertNotEqual(model.Sink.binary_data,
                            model.Source.sub.binary_data)

        orig_dir = os.getcwd()
        os.chdir(model.Source.get_abs_directory())
        try:
            for path in EXTERNAL_FILES:
                if not os.path.exists(path):
                    self.fail("path '%s' does not exist" % path)

            inp = open(EXTERNAL_FILES[2])
            data = inp.read()
            inp.close()
            self.assertEqual(data.strip(), file_data.strip())
        finally:
            os.chdir(orig_dir)

        for i in range(3):
            self.assertEqual(model.Source.obj_list[i].data, i)

        self.assertEqual(model.Oddball.executions, 0)

        # Run and verify correct operation.
        model.run()
        self.assertEqual(model.Sink.text_data,
                         model.Source.text_data)
        self.assertEqual(all(model.Sink.binary_data==model.Source.sub.binary_data),
                         True)
        
        self.assertEqual(model.Sink.binary_file.binary, True)

        self.assertEqual(model.Oddball.executions, 3)


if __name__ == '__main__':
    import nose
    sys.argv.append('--cover-package=openmdao')
    sys.argv.append('--cover-erase')
    nose.runmodule()

