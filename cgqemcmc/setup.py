from setuptools import setup, find_packages

setup(
    name='cgqemcmc',
    version='0.1.0',  # Update this for your version
    packages=find_packages(),
    include_package_data=True,
    #install_requires=['numpy>=1.14.5'],#Have ignored this for now
    # metadata to display on PyPI
    author='S. Ferguson',
    author_email='s1846096@ed.ac.uk',
    description='Code that coarse-grains the QeMCMC',

)