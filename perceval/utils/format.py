# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sympy as sp
import numpy as np


def simple_float(alpha, precision=1e-6, nsimplify=True, fracmax=63, multiplier=1, mult10=None):
    r"""
        try to get simple formula for remarkable numbers
        simple fraction, simple fraction, simple pi fraction,
        or fraction + fraction*remarkable root sqrt(2)/sqrt(5)
        return sympy object, str object
    """
    sign = 1
    if alpha < 0:
        sign = -1
        alpha = -alpha
    # look for n/p*pi or n/p
    if nsimplify:
        for r in range(1, fracmax):
            for multiplier2 in [sp.S(1), sp.pi, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5), sp.sqrt(6)]:
                v = np.float64(alpha/multiplier2*r)
                round_v = float(np.float64(v).round())
                if abs(float(v)-round_v) < precision:
                    simple = sign*sp.Rational(round_v, r)*multiplier*multiplier2
                    return simple, str(simple)
    if mult10 is None:
        mult10 = 0
        while alpha and alpha < 1:
            mult10 += 1
            alpha = alpha * 10
    else:
        change_order = mult10
        while change_order > 0:
            alpha = alpha * 10
            change_order -= 1
    if mult10 <= 3:
        while mult10:
            alpha = alpha / 10
            mult10 -= 1

    alpha = float(np.float64(alpha/precision).round()) * precision
    simple_str = str(sp.S(alpha))
    if simple_str.find(".") != -1:
        for i in range(len(simple_str)-1, 0, -1):
            if simple_str[i] == "0":
                simple_str = simple_str[:i]
                continue
            if simple_str[i] == ".":
                simple_str = simple_str[:i]
            break
    if sign < 1:
        simple_str = "-" + simple_str
    if mult10:
        simple_str += "e-%d" % mult10
    if multiplier != 1:
        simple_str += "*"+str(multiplier)
    return sp.S(sign*alpha*10**(-mult10)), simple_str


def simple_complex(c, precision=1e-6, nsimplify=True, fracmax=63):
    r = c.real
    z = c.imag

    mult10 = None

    if r != 0 and z != 0:
        mult10 = 0
        _r = r
        _z = z
        while abs(_r) < 1 and abs(_z) < 1:
            mult10 += 1
            _r = _r * 10
            _z = _z * 10

    (spr, cr) = simple_float(r, precision, nsimplify, fracmax, mult10=mult10)
    (spz, cz) = simple_float(z, precision, nsimplify, fracmax, multiplier=sp.I, mult10=mult10)
    if cz == "0":
        return spr, cr
    if cr == "0":
        return spz*sp.I, cz
    if cz[0] != "-":
        return spr+spz*sp.I, cr+"+"+cz
    else:
        return spr+spz*sp.I, cr+cz


def format_parameters(params: dict, precision: float = 1e-6, nsimplify: bool = True, separator: str = '\n') -> str:
    """
    Prepares a string output from a dictionary of parameters.
    params: dictionary where keys are the parameter names and values are the corresponding parameter value. Values can
            either be a string or a float.
    precision: Rounds a float value to the given precision
    nsimplify: Try to simplify numerical display in case of float value
    separator: String separator for the final join
    """
    output = []
    for key, value in params.items():
        if not isinstance(value, str):
            _, value = simple_float(value, precision, nsimplify)
        output.append(f'{key}={value}')
    return separator.join(output)
