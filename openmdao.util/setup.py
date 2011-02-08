
# pylint: disable-msg=F0401

import os,sys
from setuptools import setup, find_packages

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(here,
                                                 'src',
                                                 'openmdao',
                                                 'util')))

import releaseinfo
version = releaseinfo.__version__

setup(name='openmdao.util',
      version=version,
      description="various utility routines",
      long_description="""\
""",
      classifiers=[],
      keywords='',
      author='',
      author_email='',
      url='http://openmdao.org/docs/srcdocs/packages/openmdao.util.html',
      license='NASA Open Source Agreement 1.3',
      namespace_packages=["openmdao"],
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      package_data={
          'openmdao.util.test': ['src/doubler.py']
      },
      test_suite='nose.collector',
      zip_safe=False,
      install_requires=[
          'setuptools',
      ],
      entry_points = """
      [console_scripts]
      mod2dist=openmdao.util.mod2dist:mod2dist
      xyplot=openmdao.util.casedb:cmdlineXYplot
      add_reqs=openmdao.util.addreqs:add_reqs
      """
    )
