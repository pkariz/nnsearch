from setuptools import setup, find_packages

setup(
    name='nnsearch',
    version='0.1',
    author='Primoz Kariz',
    author_email='vegycslol@gmail.com',
    url="https://github.com/pkariz/nnsearch",
    packages = find_packages(),
    package_data = {"": ["*.npy", "*.txt"]},
    setup_requires=["numpy"]
    install_requires = ["numpy", "matplotlib", "annoy", "NearPy", "scikit-learn"],
    license = 'GNU GPLv3',
    description = "Exact and approximate nearest neighbors search"
)
