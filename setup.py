from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires
    
setup(
    name='real2sim',
    version='0.0.1',
    author='Xuanlin Li',
    packages=find_packages(include=['real2sim*']),
    install_requires=read_requirements(),
    python_requires=">=3.9",
)