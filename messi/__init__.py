import sys, os

__all__ = ['__main__', 'hme', 'mrots', 'utils', 'readyData', 'data_processing']
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

for i in __all__:
    __import__(i)