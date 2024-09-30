from setuptools import setup, find_packages

setup(
    name='cgqemcmc',
    version='0.2.1',  
    packages=['cgqemcmc'],  
    include_package_data=True,
    #install_requires=['numpy>=1.14.5'],# Ignores install rezquirements for now
    author='S. Ferguson',
    author_email='s1846096@ed.ac.uk',
    description='Code that coarse-grains the QeMCMC',

)
