from setuptools import setup, find_packages

setup(
    name='nnsearch',
    version='0.1',
    author='Primoz Kariz',
    author_email='vegycslol@gmail.com',
    url="https://github.com/pkariz/nnsearch",
    packages = find_packages(),
    package_data = {"nnsearch.datasets": ["sample/*.npy"]}, #include datasets
    install_requires = [
        "setuptools",
        "scikit-learn",
        "matplotlib",
        "NearPy"],
    license = 'GNU GPLv3',
    description = "Exact and approximate nearest neighbors search"
)
