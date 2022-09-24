#!/usr/bin/env manim

"""Gauss map manimation generator"""

from typing import Tuple, Union
import sys

from manim import (ThreeDScene, Surface, ThreeDAxes, VGroup,  # type: ignore
                   Vector, FadeIn, FadeOut, ApplyMethod, DEGREES, ORIGIN)
from sympy.core.sympify import SympifyError
from sympy.abc import u, v
import sympy as sym
import numpy as np
import numpy.typing as npt

Range = Tuple[float, float]

Matrices = Tuple[sym.Matrix, sym.Matrix, sym.Matrix]

# Expression representing x, y, and z coordinates
ParamExpr = Union[Tuple[sym.Expr, sym.Expr, sym.Expr], sym.Matrix]

# Vectorized function representing x, y, and z coordinates
ParamVectorize = Tuple[np.vectorize, np.vectorize, np.vectorize]


class OriginalSurface(Surface):
    """A manim surface of the parameterization for the manimation."""
    def __init__(self, X: ParamVectorize, u_range: Range, v_range: Range,
                 r_max: int = 8, **kwargs) -> None:
        """Generates the parameterized surface for the manimation.

            Args:
                X: A tuple of vectorized functions that describes the
                    parameterized surface.
                u_range: A tuple containing the starting and ending u
                    coordinates.
                v_range: A tuple containing the staring and ending v
                    coordinates.
        """
        self.__r_max = r_max
        self.__X = X

        super().__init__(self.func, u_range=u_range, v_range=v_range, **kwargs)

    def func(self, u_i: float, v_i: float) -> npt.NDArray[np.float64]:
        """Evaluates the function at u and v to generate the surface."""
        X = _evaluate(self.__X, u_i, v_i)

        # limit graphs to sphere with radius r_max
        # a spherical boundary looks better than a cubic one
        r = np.linalg.norm(X)
        if r > self.__r_max:
            X = np.divide(np.multiply(self.__r_max, X),  r)
        return X


class GaussMapSurface(Surface):
    """A manim surface of the Gauss map for the manimation."""
    def __init__(self, X: ParamVectorize, N: ParamVectorize, u_range: Range,
                 v_range: Range, **kwargs) -> None:
        """Generates the Gauss map surface for the manimation.

            Args:
                X: A tuple of vectorized functions that describes the orignal
                    surface.
                N: A tuple of vectorized functions that describes the normal
                    vectors of the original surface.
                u_range: A tuple containing the starting and ending u
                    coordinates.
                v_range: A tuple containing the starting and ending v
                    coordinates.
        """
        self.__N = N

        super().__init__(self.func, u_range=u_range, v_range=v_range, **kwargs)

    def func(self, u_i: float, v_i: float) -> npt.NDArray[np.float64]:
        """Evaluates the normal vectors at u and v to generate the surface."""
        N = _evaluate(self.__N, u_i, v_i)
        return _normalize(N)


class VectorField(VGroup):
    """The normal vectors as a group of manim vectors for the manimation."""
    def __init__(self, X: ParamVectorize, N: ParamVectorize, u_range: Range,
                 v_range: Range, r_max: int = 8, amount: int = 20,
                 **kwargs) -> None:
        """Generates the normal vector field for the manimation.

            Args:

                X: A tuple of vectorized functions that describes the original
                    surface.
                N: A tuple of vectorized functions that describes the normal
                    vectors of the original surface.
                u_range: A tuple containing the starting and ending u
                    coordinates.
                v_range: A tuple containing the starting and ending v
                    coordinates.
                r_max: The maximum radius from the center where vectors are
                    generated.
                amount: The number of vectors to generate. They will be
                    dispersed evenly throughout the surface.
        """
        super().__init__(**kwargs)
        self.__r_max = r_max

        U = np.linspace(u_range[0], u_range[1], amount)
        V = np.linspace(v_range[0], v_range[1], amount)

        for u_i in U:
            for v_i in V:
                X_i = _evaluate(X, u_i, v_i)
                N_i = _evaluate(N, u_i, v_i)

                # ignore vectors that start outside a sphere of radius r_max
                r = np.linalg.norm(X_i)
                if r < self.__r_max:
                    self.add(self.__create_vector(X_i, N_i))

    @staticmethod
    def __create_vector(pos: npt.NDArray[np.float64],
                        N: npt.NDArray[np.float64]) -> Vector:
        """Evaluates the function at u and v to generate the vector field."""
        N = _normalize(N)

        # move vector away from the surface a bit so it does not clip through
        pos = pos + 0.1 * N

        # shade_in_3d prevents vectors and their tips from being visible when
        # they are behind 3d objects.
        vector = Vector(shade_in_3d=True, direction=N).shift(pos)
        vector.get_tip().set_shade_in_3d()
        return vector


class GaussMap(ThreeDScene):
    """An animated scene that demonstrates the Gauss map of a surface."""
    def construct(self) -> None:
        """Creates the objects in the scene and plays the animation."""
        axes = ThreeDAxes()
        X_expr, u_range, v_range = _get_func()

        N_expr, X_u, X_v = _gauss(X_expr)

        print(f'x = {X_expr}')
        print(f'x_u = {(X_u[0], X_u[1], X_u[2])}')
        print(f'x_v = {(X_v[0], X_v[1], X_v[2])}')
        print(f'N = {(N_expr[0], N_expr[1], N_expr[2])}')

        X = _sym_to_np(X_expr)
        N = _sym_to_np(N_expr)

        original_surface = OriginalSurface(X, u_range, v_range)
        gauss_map = GaussMapSurface(X, N, u_range, v_range)
        vector_field = VectorField(X, N, u_range, v_range)

        self.begin_ambient_camera_rotation(rate=0.1)
        self.set_camera_orientation(phi=45*DEGREES, theta=30*DEGREES)
        self.add(original_surface)
        self.wait(5)
        self.play(FadeIn(vector_field))
        self.wait(5)
        self.play(FadeOut(original_surface))
        self.remove(original_surface)
        animations = []
        for vector in vector_field:
            vec_mid = np.array(vector.get_vector()) / 2
            animations.append(ApplyMethod(vector.move_to, ORIGIN + vec_mid,
                                          run_time=0.2))
        self.play(*animations, run_time=3.0)
        self.wait(2)
        self.play(FadeIn(gauss_map))
        self.play(FadeOut(vector_field))
        self.remove(vector_field)
        self.wait(5)


def _evaluate(V: ParamVectorize, u_i: float,
              v_i: float) -> npt.NDArray[np.float64]:
    """Evaluates the vectorized function at the given coordinates.

        Args:
            V: A tuple of vectorized functions to be evaluated.
            u_i: The u coordinate to evaluate at.
            v_i: The v coordinate to evaluate at.

        Returns:
            A numpy array of the evaluated values in x, y, and z coordinates.
    """
    x, y, z = V
    return np.array([x(u_i, v_i), y(u_i, v_i), z(u_i, v_i)])


def _sym_to_np(X: ParamExpr) -> ParamVectorize:
    """Converts sympy expressions to numpy expressions.

    Args:
        X: A sequence of 3 sympy expressions to convert.

    Returns:
        A numpy array of numpy vectorized equations.
    """
    # iter needed for sympy matrix type checking
    x, y, z = iter(X)
    x_np = sym.lambdify([u, v], x, 'numpy')
    y_np = sym.lambdify([u, v], y, 'numpy')
    z_np = sym.lambdify([u, v], z, 'numpy')
    x_vec = np.vectorize(x_np)
    y_vec = np.vectorize(y_np)
    z_vec = np.vectorize(z_np)
    return x_vec, y_vec, z_vec


def _normalize(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norm = np.linalg.norm(X)
    if norm == 0:
        norm = np.float64(1.0)
    return np.divide(X, norm)


def _gauss(X: ParamExpr) -> Matrices:
    """Calculates the normal vector and partial derivatives of X.

    Args:
        X: A parametrized equation in terms of u and v where the
            elements describe the x, y, and z coordinates in that order.

    Returns:
        A tuple (N, X_u, X_v) where N is the parameterized equation of the
        normal vector in terms of u and v, X_u is the parameterized
        equation of the paritial derivative of X with respect to u in terms
        of u and v, and X_v is the parameterized equations of the partial
        derivative of X with respect to v in terms of u and v.
    """
    # iter needed for sympy Matrix type checking
    x, y, z = iter(X)
    x_u = sym.diff(x, u)
    x_v = sym.diff(x, v)
    y_u = sym.diff(y, u)
    y_v = sym.diff(y, v)
    z_u = sym.diff(z, u)
    z_v = sym.diff(z, v)
    X_u = sym.Matrix([x_u, y_u, z_u])
    X_v = sym.Matrix([x_v, y_v, z_v])
    N = X_u.cross(X_v)

    return N, X_u, X_v


def _validate_bound(bound: str, bound_name: str, is_min: bool) -> float:
    """Validates input and evaluates the expression for the boundaries.

        **Note** under the hood sympy uses the eval
        function. This function should only be used locally.

        Args:
            bound: A string to be evaluated as a sympy
                expression. The evaluated result is expected to be a
                real number which is then converted to a float value.
                Only numbers, operators, and the constants pi and exp
                are allowed in the string.
            bound_name: the name of the boundary to be used in
                error output i.e. u_min, u_max.
            is_min: A boolean used to determine if a boundary is
                a min or a max. Used to truncate large values. True if
                the value is a min and False if the value is a max. Min
                values are truncated to -100 and max values are
                truncated to 100.

        Returns:
                The boundary expressed as a float value.

        Raises:
            ValueError: An unallowed character was found in the input
                string.
            NameError: An unallowed function or expression was found in
                the input string.
            SympifyError: The expression could not be parsed by sympy.
    """
    allowed_chars = [*'pi', *'exp', *'0123456789', *'.+-*/^()']
    allowed_names = {'pi': sym.pi, 'exp': sym.exp}

    if not all([char in allowed_chars for char in bound]):
        raise ValueError(f'Unallowed character found in {bound_name}')

    code = compile(bound, '<string>', 'eval')
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError('Only pi and exp() are allowed in '
                            f'{bound_name} got {name}')

    bound_expr = sym.sympify(bound, {'__builtins__': {}}, allowed_names,
                             evaluate=True)
    if not bound_expr.is_real:
        raise ValueError(f'{bound_name} must be a real value got '
                         f'{bound_expr}')

    if is_min:
        bound_float = max(float(bound_expr), -100.0)
    else:
        bound_float = min(float(bound_expr), 100.0)

    if bound_float == 100.0 or bound_float == -100.0:
        print(f'{bound_name} too large, truncated to {bound_float}')

    return bound_float


def _validate_function(function: str, function_name: str) -> sym.Expr:
    """Validates input and evaluates the expression for the functions.

        **Note** under the hood sympy uses the eval
        function. This function should only be used locally.

        Args:
            function: A string to be evaluated as a sympy
                expression. The evaluated result is expected to be a
                parameterized function in terms of u and v.
                Only numbers, operators, and trigonometric, hyperbolic,
                and exponential functions are allowed in the string.
            function_name: The name of the function to be used in error
                output i.e. X.

        Returns:
                The function expressed as a sympy expression.

        Raises:
            ValueError: An unallowed character was found in the input
                string.
            NameError: An unallowed function or expression was found in
                the input string.
            SympifyError: The expression could not be parsed by sympy.
    """
    allowed_chars = [*'cos', *'sin', *'tan', *'csc', *'sec', *'cot', 'h',
                     *'exp', *'log', *'pi', 'u', 'v', *'0123456789',
                     *'.+-*/^()']
    allowed_names = {'cos': sym.cos, 'sin': sym.sin, 'tan': sym.tan, 'csc':
                     sym.csc, 'sec': sym.sec, 'cot': sym.cot, 'cosh': sym.cosh,
                     'sinh': sym.sinh, 'tanh': sym.tanh, 'csch': sym.csch,
                     'sech': sym.sech, 'coth': sym.coth, 'exp': sym.exp, 'log':
                     sym.log, 'pi': sym.pi, 'u': u, 'v': v}
    if not all([char in allowed_chars for char in function]):
        raise ValueError(f'Unallowed character found in {function_name}')

    code = compile(function, '<string>', 'eval')
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f'{name} not allowed. Only trigonometric, '
                            'hyperbolic, exp, and log functions, '
                            'variables u and v, and pi as a constant '
                            f'are allowed in {function_name} got '
                            f'{name}')

    function_expr = sym.sympify(function, {'__builtins__': {}},
                                allowed_names)

    return function_expr


def _get_func() -> Tuple[ParamExpr, Range, Range]:
    """Prompts the user to input a parameterized equation.

        **Note** under the hood sympy uses the eval
        function. This function should only be used locally.

        Returns:
            A tuple ((x, y, z), u_range, v_range) where (x, y, z) is a
            tuple of sympy expressions describing the x, y, and z
            coordinates and u_range and v_range are tuples containing the
            starting and ending u and v coordinates respectively.
    """

    print('Gauss map manimation generator: ')
    x_input = input('Enter x parameterization in terms of u and v: ')
    y_input = input('Enter y parameterization in terms of u and v: ')
    z_input = input('Enter z parameterization in terms of u and v: ')
    u_min_input = input('Enter minimum u value: ')
    u_max_input = input('Enter maximum u value: ')
    v_min_input = input('Enter minimum v value: ')
    v_max_input = input('Enter maximum v value: ')

    try:
        u_min = _validate_bound(u_min_input, 'u_min', True)
        u_max = _validate_bound(u_max_input, 'u_max', False)
        v_min = _validate_bound(v_min_input, 'v_min', True)
        v_max = _validate_bound(v_max_input, 'v_max', False)

        x = _validate_function(x_input, 'x')
        y = _validate_function(y_input, 'y')
        z = _validate_function(z_input, 'z')

    except (TypeError, ValueError, AttributeError, SympifyError) as e:
        sys.exit(f'Unable to parse parameterization {[x, y, z]}: {e}')

    u_range, v_range = ((u_min, u_max), (v_min, v_max))

    def strtobool(val: str) -> bool:
        """Converts an input string into a boolean. """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        else:
            raise ValueError(f'invalid truth value {repr(val)}')

    print('Generate a Gauss map manimation for the following'
          'parameterization?')
    print(f'X: [{x}, {y}, {z}]')
    print(f'u: {u_range}, v: {v_range}')
    if not strtobool(input('[y/n]: ')):
        sys.exit('Exiting')
    return (x, y, z), u_range, v_range
