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


class TypeAsserter:

    def __init__(self, annotations):
        self.argument_annotations = {k:v for k,v in annotations.items() if k != "return"}
        self.return_annotation = annotations.get("return", None)
        # Keeps map from generic shape variables to first seen sizes
        self.generic_shapes = {}


    def assert_arguments(self, named_args):
        for name, annotation in self.argument_annotations.items():
            if name not in named_args:
                raise AnnotationException(f'{name} is not a parameter of the function')
            self.assert_type(name, named_args[name], annotation)


    def assert_return(self, output):
        if self.return_annotation is not None:
            self.assert_type("return", output, self.return_annotation)


    def assert_type(self, name, arg, annotation):
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
                if not isinstance(wish, list):
                    raise AnnotationException(f'{wish} is not a valid shape annotation.')

                # Build up shape from cache, while updating unseen variables
                concrete_wish = []
                for wish_dim, wish_size in enumerate(wish):
                    if isinstance(wish_size, str):
                        if not wish_size in self.generic_shapes:
                            self.generic_shapes[wish_size] = arg.shape[wish_dim]
                        concrete_wish.append(self.generic_shapes[wish_size])
                    elif isinstance(wish_size, (int, float)):
                        concrete_wish.append(wish_size)
                    else:
                        raise AnnotationException(f'{wish_size} in shape annotation is not an int, float, or string.')

                if len(arg.shape) != len(concrete_wish):
                    err = f'/{name}/ {arg.shape} is not of same length as desired shape {concrete_wish}'
                    raise ShapeException(err)

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



def tensorcheck(annotations):
    """
    The type checking decorator for Numpy arrays.
    Reference: https://realpython.com/primer-on-python-decorators
    """
    def decorator(func):
        @wraps(func)
        def func_with_asserts(*args, **kwargs):
            asserter = TypeAsserter(annotations)
            named_args = getcallargs(func, *args, **kwargs)
            asserter.assert_arguments(named_args)
            output = func(**named_args)
            asserter.assert_return(output)
            return output
        return func_with_asserts
    return decorator
