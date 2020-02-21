from setuptools import setup, find_packages

setup(
    name='tocsin',
    version='pre-alpha-0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0',
        # 'tensorflow-probability',
        'scikit-learn>=0.22',
        'numpy',
        'tqdm',
        # 'google-nucleus', # does not get installed on OSX
    ],
    long_description=open('README.md').read(),
)
