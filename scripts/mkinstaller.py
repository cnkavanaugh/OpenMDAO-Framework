"""
Generates either a go-openmdao.py script for installation
of an openmdao release or a go-openmdao-dev.py script for creating a 
virtualenv with 'develop' versions of all of the openmdao packages. Both
scripts bootstrap a virtualenv environment.

Use the --dev option to generate a go-openmdao-dev.py script.
"""

import sys, os
import virtualenv
import pprint
import StringIO
from pkg_resources import working_set, Requirement

#
#      EDIT THE FOLLOWING TWO LISTS TO CONTROL THE PACKAGES THAT WILL BE
#      REQUIRED FOR THE OPENMDAO RELEASE AND DEVELOPMENT ENVIRONMENTS
#
# List of packages that must be pre-installed before openmdao can be installed
# The presence of these will be checked at the beginning and the install will abort
# if they're not found
openmdao_prereqs = ['numpy', 'scipy']

# list of tuples of the form: (openmdao package, parent directory relative to 
# the top of the repository, package type).  The directory only has meaning for a dev install and
# is ignored in the user install. '' should be used instead of '.' to indicate 
# that the parent dir is the top of the repository. The package type should be 'sdist' or 'bdist_egg'
# for binary packages
openmdao_packages = [('openmdao.util', '', 'sdist'), 
                     ('openmdao.units', '', 'sdist'), 
                     ('openmdao.main', '', 'sdist'), 
                     ('openmdao.lib', '', 'sdist'), 
                     ('openmdao.test', '', 'sdist'),
                     ('openmdao.examples.simple', 'examples', 'sdist'),
                     ('openmdao.examples.bar3simulation', 'examples', 'bdist_egg'),
                     ('openmdao.examples.enginedesign', 'examples', 'bdist_egg'),
                     ('openmdao.examples.mdao', 'examples', 'sdist'),
                     ('openmdao.examples.expected_improvement', 'examples', 'sdist'),
                    ]

def get_adjust_options(options, version):
    """Return a string containing the definition of the adjust_options function
    that will be included in the generated virtualenv bootstrapping script.
    """
    if options.dev:
        code = """
    for arg in args:
        if not arg.startswith('-'):
            print 'ERROR: no args allowed that start without a dash (-)'
            sys.exit(-1)
    args.append(join(os.path.dirname(__file__), 'devenv'))  # force the virtualenv to be in <top>/devenv
"""
    else:
        code = """
    # name of virtualenv defaults to openmdao-<version>
    if len(args) == 0:
        args.append('openmdao-%%s' %% '%s')
""" % version
    
    return """
def adjust_options(options, args):
    major_version = sys.version_info[:2]
    if major_version != (2,6):
        print 'ERROR: python major version must be 2.6. yours is %%s' %% str(major_version)
        sys.exit(-1)
%s

""" % code

def main(options):
    
    if options.dev:
        openmdao_packages.append(('openmdao.devtools', '', 'sdist'))
        sout = StringIO.StringIO()
        pprint.pprint(openmdao_packages, sout)
        pkgstr = sout.getvalue()
        make_dev_eggs = """
        # now install dev eggs for all of the openmdao packages
        topdir = os.path.abspath(os.path.dirname(__file__))
        startdir = os.getcwd()
        absbin = os.path.abspath(bin_dir)
        openmdao_packages = %s
        try:
            for pkg, pdir, _ in openmdao_packages:
                os.chdir(join(topdir, pdir, pkg))
                cmdline = [join(absbin, 'python'), 'setup.py', 
                           'develop', '-N'] + cmds
                subprocess.check_call(cmdline)
        finally:
            os.chdir(startdir)
        """ % pkgstr
        wing = """
    # copy the wing project file into the virtualenv
    proj_template = join(topdir,'config','wing_proj_template.wpr')
    
    shutil.copy(proj_template, 
                join(abshome,'etc','wingproj.wpr'))
                
        """
    else:
        make_dev_eggs = ''
        wing = ''

    if options.test:
        url = 'http://torpedo.grc.nasa.gov:31004/dists'
    else:
        url = 'http://openmdao.org/dists'

    script_str = """

openmdao_prereqs = %(openmdao_prereqs)s

def extend_parser(parser):
    parser.add_option("--reqs", action="append", type="string", dest='reqs', 
                      help="specify file with additional requirements", default=[])

%(adjust_options)s

def _single_install(cmds, req, bin_dir):
    global logger
    cmdline = [join(bin_dir, 'easy_install'),'-NZ'] + cmds + [req]
        # pip seems more robust than easy_install, but won't install binary distribs :(
        #cmdline = [join(bin_dir, 'pip'), 'install'] + cmds + [req]
    logger.debug("running command: %%s" %% ' '.join(cmdline))
    subprocess.check_call(cmdline)

def after_install(options, home_dir):
    global logger, openmdao_prereqs
    
    reqs = %(reqs)s
    url = '%(url)s'
    etc = join(home_dir, 'etc')
    if sys.platform == 'win32':
        lib_dir = join(home_dir, 'Lib')
        bin_dir = join(home_dir, 'Scripts')
    else:
        lib_dir = join(home_dir, 'lib', py_version)
        bin_dir = join(home_dir, 'bin')

    if not os.path.exists(etc):
        os.makedirs(etc)
        
    failed_imports = []
    for pkg in openmdao_prereqs:
        try:
            __import__(pkg)
        except ImportError:
            failed_imports.append(pkg)
    if failed_imports:
        logger.error("ERROR: the following prerequisites could not be imported: %%s." %% failed_imports)
        logger.error("These must be installed in the system level python before installing OpenMDAO.")
        sys.exit(-1)
    
    cmds = ['-f', url]
    try:
        for req in reqs:
            _single_install(cmds, req, bin_dir)
        
%(make_dev_eggs)s

        # add packages from any specified requirements files
        if options.reqs:
            if sys.platform == 'win32':
                reqscript = 'add_reqs-script.py'
            else:
                reqscript = 'add_reqs'
            subprocess.check_call([join(bin_dir, 'python'),
                                   join(bin_dir, reqscript), '-f', url] + options.reqs,
                                  env=os.environ)

    except Exception as err:
        logger.error("ERROR: build failed: %%s" %% str(err))
        sys.exit(-1)
        
    if sys.platform != 'win32':
        import fnmatch
        def _find_files(pat, startdir):
            for path, dirlist, filelist in os.walk(startdir):
                for name in fnmatch.filter(filelist, pat):
                    yield os.path.join(path, name)

       # in order to find all of our shared libraries, modify the activate
       # script to put their directories in LD_LIBRARY_PATH
        pkgdir = os.path.join(lib_dir, 'site-packages')
        sofiles = [os.path.abspath(x) for x in _find_files('*.so',pkgdir)]
                      
        final = set()
        for f in sofiles:
            pyf = os.path.splitext(f)[0]+'.py'
            if not os.path.exists(pyf):
                final.add(os.path.dirname(f))
                
        subdict = { 'libpath': 'LD_LIBRARY_PATH',
                    'add_on': os.pathsep.join(final)
                    }
                    
        if len(final) > 0:
            activate_template = '\\n'.join([
            'export VIRTUAL_ENV',
            '',
            'if [ -z "$%%(libpath)s" ] ; then',
            '   %%(libpath)s=""',
            'fi',
            '',
            '%%(libpath)s=$%%(libpath)s:%%(add_on)s',
            'export %%(libpath)s',
            ])
            f = open(os.path.join(absbin, 'activate'), 'r')
            content = f.read()
            f.close()
            f = open(os.path.join(absbin, 'activate'), 'w')
            f.write(content.replace('export VIRTUAL_ENV', activate_template %% subdict, 1))
            f.close()

    abshome = os.path.abspath(home_dir)
    
%(wing)s

    print '\\n\\nThe OpenMDAO virtual environment has been installed in %%s.' %% abshome
    print 'From %%s, type:\\n' %% abshome
    if sys.platform == 'win32':
        print r'Scripts\\activate'
    else:
        print '. bin/activate'
    print "\\nto activate your environment and start using OpenMDAO."
    """
    
    reqs = set()
    
    version = '?.?.?'
    dists = working_set.resolve([Requirement.parse(r[0]) for r in openmdao_packages])
    excludes = set(['setuptools', 'distribute']+openmdao_prereqs)
    for dist in dists:
        if dist.project_name == 'openmdao.main':
            version = dist.version
        if dist.project_name not in excludes:
            if options.dev: # in a dev build, exclude openmdao stuff because we'll make them 'develop' eggs
                if not dist.project_name.startswith('openmdao.'):
                    reqs.add('%s' % dist.as_requirement())
            else:
                reqs.add('%s' % dist.as_requirement())

    reqs = openmdao_prereqs + list(reqs)
    
    optdict = { 
        'reqs': reqs, 
        'version': version, 
        'url': url ,
        'make_dev_eggs': make_dev_eggs,
        'wing': wing,
        'adjust_options': get_adjust_options(options, version),
        'openmdao_prereqs': openmdao_prereqs,
    }
    
    dest = os.path.abspath(options.dest)
    if options.dev:
        scriptname = os.path.join(dest,'go-openmdao-dev.py')
    else:
        scriptname = os.path.join(dest,'go-openmdao-%s.py' % version)
    with open(scriptname, 'wb') as f:
        f.write(virtualenv.create_bootstrap_script(script_str % optdict))
    os.chmod(scriptname, 0755)
    

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--dev", action="store_true", dest='dev', 
                      help="if present, a development script will be generated instead of a release script")
    parser.add_option("--dest", action="store", type="string", dest='dest', 
                      help="specify destination directory", default='.')
    parser.add_option("-t", "--test", action="store_true", dest="test",
                      help="if present, generated installer will point to /OpenMDAO/release_test/dists")
    
    (options, args) = parser.parse_args()
    
    if len(args) > 0:
        print 'unrecognized args: %s' % args
        parser.print_help()
        sys.exit(-1)

    main(options)
