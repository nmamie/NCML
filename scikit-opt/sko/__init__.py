__version__ = '0.7'

from . import DE, GA, PSO, SA, ACA, APSO, IFSA, AFSA, IA, tools


def start():
    print('''
    scikit-opt import successfully,
    version: {version}
    Author: Guo Fei,
    Improvements: Noah Mamie,
    Email: guofei9987@foxmail.com
    repo: https://github.com/guofei9987/scikit-opt,
    documents: https://scikit-opt.github.io/
    '''.format(version=__version__))
