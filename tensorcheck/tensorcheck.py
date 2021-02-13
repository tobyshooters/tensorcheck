import numpy as np
import torch

from inspect import getcallargs
from functools import wraps

class AnnotationException(Exception): pass
class TypeException(Exception):       pass
class DataTypeException(Exception):   pass
class ShapeException(Exception):      pass
class LowerBoundException(Exception): pass
class UpperBoundException(Exception): pass

def assert_types(types, named_args):

    generic_shapes = {}

    for name, annotation in types.items():
        if name not in named_args:
            raise AnnotationException(f'{name} is not a parameter of the function')
        arg = named_args[name]

        if type(arg) not in [np.ndarray, torch.Tensor]:
            raise TypeException(f'Annotation must be np.ndarray or torch.Tensor, not {type(arg)}')

        for check, wish in annotation.items():

            if check == "dtype":
                # Reference: stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
                if isinstance(arg, np.ndarray) and not isinstance(arg.flat[0], wish):
                    raise DataTypeException(f'/{name}/ dtype {arg.flat[0].dtype} is not {wish}')

                elif isinstance(arg, torch.Tensor) and not arg.dtype == wish:
                    raise DataTypeException(f'/{name}/ dtype {arg.dtype} is not {wish}')

            elif check == "shape":
                concrete_wish = []
                for wish_dim, wish_size in enumerate(wish):
                    if isinstance(wish_size, str):
                        if not wish_size in generic_shapes:
                            generic_shapes[wish_size] = arg.shape[wish_dim]
                        concrete_wish.append(generic_shapes[wish_size])
                    else:
                        concrete_wish.append(wish_size)

                for wish_dim, wish_size in enumerate(concrete_wish):
                    if arg.shape[wish_dim] != wish_size:
                        errs = f'/{name}/ dim {wish_dim} of {arg.shape} is not {wish[wish_dim]}={concrete_wish[wish_dim]}'
                        raise ShapeException(errs)

            elif check == "range":
                rmin, rmax = wish

                if not arg.min() >= float(rmin):
                    errs = f'/{name}/ min value {arg.min()} is less than {rmin}'
                    raise LowerBoundException(errs)

                if not arg.max() <= float(rmax):
                    errs = f'/{name}/ max value {arg.max()} is greater than {rmax}'
                    raise UpperBoundException(errs)


def tensorcheck(types):
    """
    The type checking decorator for Numpy arrays.
    Reference: https://realpython.com/primer-on-python-decorators
    """
    def decorator(func):
        @wraps(func)
        def func_with_asserts(*args, **kwargs):
            named_args = getcallargs(func, *args, **kwargs)
            assert_types(types, named_args)
            return func(**named_args)
        return func_with_asserts
    return decorator
