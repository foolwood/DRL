# Copyright (C) Alibaba Group Holding Limited. 

import numbers
import argparse


__all__ = ['Config']


class Config(object):

    def __init__(self, **kwargs):
        self.__name__ = kwargs.pop('__name__', 'Config')
        self.git_hash = ''
        self.update(**kwargs)
    
    def update(self, **kwargs):
        if len(kwargs) == 0:
            return
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.update_dependencies(**kwargs)
    
    def parse(self):
        parser = argparse.ArgumentParser(description=self.__name__)
        for k, v in self.__dict__.items():
            if k != '__name__':
                parser.add_argument('--' + k, type=type(v), default=v)
        args = parser.parse_args()
        for k, v in self.__dict__.items():
            if k != '__name__':
                setattr(self, k, getattr(args, k, getattr(self, k)))
        self.update_dependencies(**vars(args))
    
    def update_dependencies(self, **kwargs):
        # overwrite this function if some configs are dependent on others
        pass
    
    def __repr__(self):
        s = self.__name__ + ':\n'
        for k, v in self.__dict__.items():
            if k == '__name__':
                continue
            if not isinstance(v, (numbers.Number, str, type(None))) and len(str(v)) > 200:
                v = type(v)
            s += '\t{}: {}\n'.format(k, v)
        return s
