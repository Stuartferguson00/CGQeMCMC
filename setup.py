from setuptools import setup, find_packages

setup(
    name='cgqemcmc',
    use_scm_version=True,  # Use setuptools-scm for versioning
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    include_package_data=True,
    # install_requires=['numpy>=1.14.5'],  # Have ignored this for now
    # metadata to display on PyPI
    author='S. Ferguson',
    author_email='s1846096@ed.ac.uk',
    description='Code that coarse-grains the QeMCMC',
)
