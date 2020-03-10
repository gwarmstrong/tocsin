from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

import subprocess
import os
import numpy as np
import sys

fastp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'fastp-bind/')


PREFIX = os.environ.get('PREFIX', "")

CLEAN = True
FORCE_COMPILE = True

# https://stackoverflow.com/a/33308902/379593
if sys.platform == 'darwin':
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.12'

def compile_fastp():
    """Clean and compile the SSU binary"""
    # clean the target
    if CLEAN:
        ret = subprocess.call(['make', 'clean'], cwd=fastp_dir)
        if ret != 0:
            raise Exception('Error compiling ssu!')

    # ret = subprocess.call(['make'], cwd=fastp_dir)
    # if ret != 0:
    #     raise Exception('Error compiling ssu!')

    ret = subprocess.call(['make', 'api'], cwd=fastp_dir)
    if ret != 0:
        raise Exception('Error compiling ssu!')


class build_ext(build_ext_orig):
    """Pre-installation for any time an Extension is built"""

    def run(self):
        self.run_compile_fastp()
        super().run()

    def run_compile_fastp(self):
        self.execute(compile_fastp, [], 'Compiling fastp')
        # if PREFIX:
        #     self.copy_file(os.path.join(fastp_dir, 'libfastp.so'),
        #                    os.path.join(PREFIX, 'lib/'))


if sys.platform == "darwin":
    LINK_ARGS = [] #['-Wl,-L./fastp-bind/libfastp.so']
else:
    LINK_ARGS = []


USE_CYTHON = os.environ.get('USE_CYTHON', True)
ext = '.pyx' if USE_CYTHON else '.cpp'

# extensions = [Extension("unifrac._api",
#                         sources=["unifrac/_api" + ext,
#                                  "sucpp/api.cpp"],
#                         language="c++",
#                         include_dirs=[np.get_include()] + ['sucpp/'],
#                         libraries=['fastp'])]

extensions = [Extension("tocsin.reader",
                        sources=["tocsin/reader.pyx",
                                 "fastp-bind/src/api.cpp"
                                 ],
                        language="c++",
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-std=c++11"] + LINK_ARGS,
                        # library_dirs=[
                        #               "./fastp-bind/",
                        #               ],
                        # libraries=['fastp'],
                        )]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, force=FORCE_COMPILE,
                           language="c++",
                           )

setup(
    name='tocsin',
    version='pre-alpha-0.1',
    packages=find_packages(),
    install_requires=[
        'cython >= 0.26',
        'tensorflow>=2.0',
        # 'tensorflow-probability',
        'scikit-learn>=0.22',
        'numpy',
        'tqdm',
        # 'google-nucleus', # does not get installed on OSX
    ],
    extras_require={
        "benchmark": ["asv"]
    },
    ext_modules=extensions,
    long_description=open('README.md').read(),
    cmdclass={'build_ext': build_ext},
)
