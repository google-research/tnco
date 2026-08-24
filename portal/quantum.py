# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum computing embedded in Python.

    from quantum import *

    m = Observable()
    with Qubits(3) as a:
        a[0].H()              # Hadamard gate
        a[1] ^= a[0]          # CNOT: control a[0], target a[1]
        a[2] ^= a[0] & a[1]   # Toffoli: XOR of the controls' AND
        a[0] ^= 1             # XOR with a classical one = X gate
        m << a                # measure the whole register (collapse)

    print(m)                        # bit string, e.g. "110"
    print(a.distribution_as_list)   # post-collapse: a delta

Gate powers (as in cirq): applying a gate returns a handle, and ** t
retroactively replaces the gate just applied by that power:

    a[0].X() ** 0.5           # partial negation, sqrt(X)
    a[1].CNOT(a[0]) ** 0.5    # sqrt(CNOT)
    a[0].CZ(a[1]) ** 0.5      # CZ^0.5 == CP(pi/2)

Bit convention: a[0] is the most significant (leftmost) bit; index k in
distribution_as_list is the register's bit string read as a binary
number.

Measurement is honest: it samples by the Born rule and collapses the state.
distribution_as_list is the Born distribution of the state as it stands.
"""

import cmath
from fractions import Fraction
import math
import random

__all__ = [
    "Qubits", "Observable", "QubitRef", "QubitSlice", "Circuit", "QuantumError"
]

_SQRT2 = 1.0 / math.sqrt(2.0)

_GATES = {
    "H": ((_SQRT2, _SQRT2), (_SQRT2, -_SQRT2)),
    "X": ((0, 1), (1, 0)),
    "Y": ((0, -1j), (1j, 0)),
    "Z": ((1, 0), (0, -1)),
    "S": ((1, 0), (0, 1j)),
    "Sdg": ((1, 0), (0, -1j)),
    "T": ((1, 0), (0, cmath.exp(1j * math.pi / 4))),
    "Tdg": ((1, 0), (0, cmath.exp(-1j * math.pi / 4))),
}


class QuantumError(Exception):
    pass


def _inv2(m):
    """Inverse of a 2x2 unitary: the Hermitian conjugate."""
    (a, b), (c, d) = m
    return ((complex(a).conjugate(), complex(c).conjugate()),
            (complex(b).conjugate(), complex(d).conjugate()))


def _mat_pow(m, t):
    """Principal power of a 2x2 unitary matrix (spectral decomposition)."""
    (a, b), (c, d) = [[complex(x) for x in row] for row in m]
    tr, det = a + d, a * d - b * c
    disc = cmath.sqrt(tr * tr - 4 * det)
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    p1 = cmath.exp(t * cmath.log(l1))
    if abs(l1 - l2) < 1e-12:
        # a unitary with a repeated eigenvalue is scalar
        return ((p1, 0), (0, p1))
    p2 = cmath.exp(t * cmath.log(l2))
    e = l1 - l2
    f1, f2 = p1 / e, -p2 / e
    return ((f1 * (a - l2) + f2 * (a - l1), (f1 + f2) * b),
            ((f1 + f2) * c, f1 * (d - l2) + f2 * (d - l1)))


def _pow_label(label, t):
    return "%s^%g" % (label, t)


def _zyz(m):
    """Splits a 2x2 unitary as e^{i gamma} * u3(theta, phi, lam).

    u3(t,p,l) = [[cos(t/2), -e^{il} sin(t/2)],
                 [e^{ip} sin(t/2), e^{i(p+l)} cos(t/2)]]
    """
    (a, b), (c, d) = [[complex(x) for x in row] for row in m]
    theta = 2 * math.atan2(abs(c), abs(a))
    if abs(c) < 1e-12:  # diagonal
        gamma = cmath.phase(a)
        return 0.0, 0.0, cmath.phase(d) - gamma, gamma
    if abs(a) < 1e-12:  # antidiagonal
        gamma = cmath.phase(-b)
        return math.pi, cmath.phase(c) - gamma, 0.0, gamma
    gamma = cmath.phase(a)
    phi = cmath.phase(c) - gamma
    lam = cmath.phase(-b) - gamma
    return theta, phi, lam, gamma


def _qasm_angle(phi):
    """An angle for QASM: a clean fraction of pi when possible."""
    return _angle_str(phi, "pi", "%d*pi", "%.10g")


def _angle_str(phi, pi_sym, mult_fmt, fallback):
    fr = Fraction(phi / math.pi).limit_denominator(64)
    if fr != 0 and abs(phi - float(fr) * math.pi) < 1e-9:
        num, den = fr.numerator, fr.denominator
        if num == 1:
            s = pi_sym
        elif num == -1:
            s = "-" + pi_sym
        else:
            s = mult_fmt % num
        return s if den == 1 else "%s/%d" % (s, den)
    return fallback % phi


def _fmt_angle(phi):
    return _angle_str(phi, "π", "%dπ", "%.3g")


class GateHandle:
    """Handle of an applied gate: ** t retroactively takes its power."""

    def __init__(self, reg, ref):
        self.reg = reg
        self.ref = ref
        self.log_index = len(reg._log) - 1

    def __pow__(self, t):
        return self.reg._retro_pow(self.log_index, t, self.ref)

    def __getattr__(self, name):
        return getattr(self.ref, name)


class Controls:
    """A conjunction of control qubits: a[i] & a[j] (Toffoli and deeper)."""

    def __init__(self, refs):
        self.refs = tuple(refs)

    def __and__(self, other):
        if isinstance(other, QubitRef):
            return Controls(self.refs + (other,))
        if isinstance(other, Controls):
            return Controls(self.refs + other.refs)
        return NotImplemented


class QubitRef:
    """A reference to one qubit; gate methods return a GateHandle."""

    def __init__(self, reg, index):
        self.reg = reg
        self.index = index

    def _g(self, name):
        self.reg._apply1(self.index, _GATES[name], name)
        return GateHandle(self.reg, self)

    def H(self):
        return self._g("H")

    def X(self):
        return self._g("X")

    def Y(self):
        return self._g("Y")

    def Z(self):
        return self._g("Z")

    def S(self):
        return self._g("S")

    def Sdg(self):
        return self._g("Sdg")

    def T(self):
        return self._g("T")

    def Tdg(self):
        return self._g("Tdg")

    def RX(self, theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        self.reg._apply1(self.index, ((c, -1j * s), (-1j * s, c)),
                         "RX(%s)" % _fmt_angle(theta))
        return GateHandle(self.reg, self)

    def R(self, phi):
        """A plain rotation by phi: ((cos, -sin), (sin, cos)).

        R(phi)=RY(2*phi).
        """
        c, s = math.cos(phi), math.sin(phi)
        self.reg._apply1(self.index, ((c, -s), (s, c)),
                         "R(%s)" % _fmt_angle(phi))
        return GateHandle(self.reg, self)

    def RY(self, theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        self.reg._apply1(self.index, ((c, -s), (s, c)),
                         "RY(%s)" % _fmt_angle(theta))
        return GateHandle(self.reg, self)

    def RZ(self, theta):
        e0, e1 = cmath.exp(-1j * theta / 2), cmath.exp(1j * theta / 2)
        self.reg._apply1(self.index, ((e0, 0), (0, e1)),
                         "RZ(%s)" % _fmt_angle(theta))
        return GateHandle(self.reg, self)

    def P(self, phi):
        self.reg._apply1(self.index, ((1, 0), (0, cmath.exp(1j * phi))),
                         "P(%s)" % _fmt_angle(phi))
        return GateHandle(self.reg, self)

    def CZ(self, other):
        self.reg._apply_cz(self.index, other.index)
        return GateHandle(self.reg, self)

    def CP(self, other, phi):
        """Controlled phase shift: diag(1,1,1,e^{i phi})."""
        self.reg._apply_cphase(self.index, other.index, phi)
        return GateHandle(self.reg, self)

    def __and__(self, other):
        if isinstance(other, QubitRef):
            return Controls((self, other))
        return NotImplemented

    def __ixor__(self, other):
        if isinstance(other, QubitRef):
            self.reg._apply_mcx((other.index,), self.index)
        elif isinstance(other, Controls):
            self.reg._apply_mcx(tuple(r.index for r in other.refs), self.index)
        elif isinstance(other, (bool, int)):
            if other & 1:
                self._g("X")
        else:
            return NotImplemented
        return self

    def CNOT(self, other):
        """Like ^=, but returns a handle: a[1].CNOT(a[0]) ** 0.5.

        Args:
            other: the control qubit, or a conjunction of controls
                (a[0] & a[1]) for a Toffoli and deeper.
        """
        if isinstance(other, QubitRef):
            self.reg._apply_mcx((other.index,), self.index)
        elif isinstance(other, Controls):
            self.reg._apply_mcx(tuple(r.index for r in other.refs), self.index)
        else:
            raise QuantumError(
                "a control is a qubit or a conjunction of qubits")
        return GateHandle(self.reg, self)

    def measure(self):
        return self.reg._measure_one(self.index)


class Qubits:
    """A register of n qubits, initialized to |0...0>."""

    def __init__(self, n):
        if n < 1:
            raise QuantumError("at least one qubit is needed")
        self.n = n
        self.state = [0j] * (1 << n)
        self.state[0] = 1 + 0j
        self._log = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return QubitSlice(self, tuple(range(self.n))[i])
        if -self.n <= i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise QuantumError("no such qubit: %r" % (i,))
        return QubitRef(self, i)

    def __setitem__(self, i, value):
        # Supports a[i] ^= ...: __ixor__ already applied the gate, no-op here.
        if -self.n <= i < 0:
            i += self.n
        if (isinstance(value, QubitRef) and value.reg is self and
                value.index == i):
            return
        raise QuantumError("a qubit cannot be assigned a value directly")

    def _bit(self, i):
        return 1 << (self.n - 1 - i)

    def _apply1_core(self, i, m):
        bit = self._bit(i)
        st = self.state
        for k in range(len(st)):
            if not k & bit:
                k1 = k | bit
                a0, a1 = st[k], st[k1]
                st[k] = m[0][0] * a0 + m[0][1] * a1
                st[k1] = m[1][0] * a0 + m[1][1] * a1

    def _apply1(self, i, m, label="U"):
        self._log.append(("g", label, i, m))
        self._apply1_core(i, m)

    def _apply_mc1(self, controls, target, m, label):
        """A multi-controlled single-qubit gate (for powers of CNOT)."""
        self._log.append(("mc1", tuple(controls), target, m, label))
        self._apply_mc1_core(controls, target, m)

    def _retro_pow(self, log_index, t, ref):
        """Replaces the last applied gate by its power t."""
        if log_index != len(self._log) - 1:
            raise QuantumError("only the last gate can be raised to a power")
        op = self._log.pop()
        kind = op[0]
        if kind == "g":
            _, label, i, m = op
            self._apply1_core(i, _inv2(m))  # undo
            self._apply1(i, _mat_pow(m, t), _pow_label(label, t))
        elif kind == "cx":
            _, controls, target = op
            # CX is its own inverse
            self._apply_mc1_core(controls, target, _GATES["X"])
            self._apply_mc1(controls, target, _mat_pow(_GATES["X"], t),
                            _pow_label("X", t))
        elif kind == "mc1":
            _, controls, target, m, label = op
            self._apply_mc1_core(controls, target, _inv2(m))
            self._apply_mc1(controls, target, _mat_pow(m, t),
                            _pow_label(label, t))
        elif kind == "cp":
            _, i, j, phi = op
            self._apply_cphase_core(i, j, -phi)  # undo
            self._apply_cphase(i, j, phi * t)
        else:
            raise QuantumError("this gate has no powers")
        return GateHandle(self, ref)

    def _apply_mc1_core(self, controls, target, m):
        tbit = self._bit(target)
        cmask = 0
        for c in controls:
            cmask |= self._bit(c)
        st = self.state
        for k in range(len(st)):
            if (k & cmask) == cmask and not k & tbit:
                k1 = k | tbit
                a0, a1 = st[k], st[k1]
                st[k] = m[0][0] * a0 + m[0][1] * a1
                st[k1] = m[1][0] * a0 + m[1][1] * a1

    def _apply_mcx(self, controls, target):
        if target in controls:
            raise QuantumError("a qubit cannot control itself")
        self._log.append(("cx", tuple(controls), target))
        tbit = self._bit(target)
        cmask = 0
        for c in controls:
            cmask |= self._bit(c)
        st = self.state
        for k in range(len(st)):
            if (k & cmask) == cmask and not k & tbit:
                k1 = k | tbit
                st[k], st[k1] = st[k1], st[k]

    def _apply_cz(self, i, j):
        self._apply_cphase(i, j, math.pi)

    def _apply_cphase_core(self, i, j, phi):
        mask = self._bit(i) | self._bit(j)
        ph = cmath.exp(1j * phi)
        st = self.state
        for k in range(len(st)):
            if (k & mask) == mask:
                st[k] *= ph

    def _apply_cphase(self, i, j, phi):
        if i == j:
            raise QuantumError("a qubit cannot control itself")
        self._log.append(("cp", i, j, phi))
        self._apply_cphase_core(i, j, phi)

    def _distribution(self):
        return [abs(x)**2 for x in self.state]

    @property
    def amplitudes(self):
        return [
            complex(round(z.real, 12), round(z.imag, 12)) for z in self.state
        ]

    @property
    def as_circuit(self):
        return Circuit(self.n, list(self._log))

    @property
    def distribution_as_list(self):
        return [round(p, 12) for p in self._distribution()]

    @property
    def distribution(self):
        fmt = "{:0%db}" % self.n
        return {
            fmt.format(k): p
            for k, p in enumerate(self.distribution_as_list)
            if p > 0
        }

    def _extract(self, k, bits):
        v = 0
        for b in bits:
            v = (v << 1) | (1 if k & b else 0)
        return v

    def _marginal(self, indices):
        bits = [self._bit(i) for i in indices]
        dist = [0.0] * (1 << len(indices))
        for k, z in enumerate(self.state):
            if z:
                dist[self._extract(k, bits)] += abs(z)**2
        return dist

    def _measure_subset(self, indices):
        self._log.append(("M", tuple(indices)))
        dist = self._marginal(indices)
        r = random.random()
        acc = 0.0
        outcome = len(dist) - 1
        for v, p in enumerate(dist):
            acc += p
            if r < acc:
                outcome = v
                break
        bits = [self._bit(i) for i in indices]
        norm = math.sqrt(dist[outcome])
        st = self.state
        for k in range(len(st)):
            if self._extract(k, bits) == outcome:
                st[k] /= norm
            else:
                st[k] = 0j
        return format(outcome, "0%db" % len(indices))

    def _measure_all(self):
        return self._measure_subset(tuple(range(self.n)))

    def _measure_one(self, i):
        return int(self._measure_subset((i,)))

    def _apply_permutation(self, indices, f, controls=(), name=None):
        if set(indices) & set(controls):
            raise QuantumError("the control overlaps the permuted register")
        if name is None:
            name = getattr(f, "__name__", "f")
            if name == "<lambda>":
                name = "f"
        mlen = len(indices)
        table = [f(v) for v in range(1 << mlen)]
        if sorted(table) != list(range(1 << mlen)):
            raise QuantumError(
                "the function is not a permutation of the register")
        self._log.append(
            ("perm", tuple(indices), tuple(controls), name, tuple(table)))
        bits = [self._bit(i) for i in indices]
        cmask = 0
        for c in controls:
            cmask |= self._bit(c)
        st = self.state
        new = [0j] * len(st)
        for k, z in enumerate(st):
            if not z:
                continue
            if (k & cmask) == cmask:
                w = table[self._extract(k, bits)]
                k2 = k
                for p, b in enumerate(bits):
                    if (w >> (mlen - 1 - p)) & 1:
                        k2 |= b
                    else:
                        k2 &= ~b
                new[k2] = z
            else:
                new[k] = z
        self.state = new


class QubitSlice:
    """A subregister a[i:j] — behaves like a small register."""

    def __init__(self, reg, indices):
        self.reg = reg
        self.indices = tuple(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return QubitSlice(self.reg, self.indices[i])
        return QubitRef(self.reg, self.indices[i])

    def __setitem__(self, i, value):
        if (isinstance(value, QubitRef) and value.reg is self.reg and
                value.index == self.indices[i]):
            return
        raise QuantumError("a qubit cannot be assigned a value directly")

    def permute(self, f, when=None, name=None):
        """A classical reversible function f on the subregister: |v> -> |f(v)>.

        when is a control qubit (or a conjunction a[i] & a[j]): the
        permutation is applied only on the branches where the control
        equals 1. name is the label used by as_circuit.
        """
        if when is None:
            controls = ()
        elif isinstance(when, QubitRef):
            controls = (when.index,)
        elif isinstance(when, Controls):
            controls = tuple(r.index for r in when.refs)
        else:
            raise QuantumError(
                "when must be a qubit or a conjunction of qubits")
        self.reg._apply_permutation(self.indices, f, controls, name)
        return self

    @property
    def distribution_as_list(self):
        return [round(p, 12) for p in self.reg._marginal(self.indices)]

    @property
    def distribution(self):
        fmt = "{:0%db}" % len(self.indices)
        return {
            fmt.format(v): p
            for v, p in enumerate(self.distribution_as_list)
            if p > 0
        }


class Circuit:
    """The circuit: the register's op log.

    print(a.as_circuit) draws it.
    """

    def __init__(self, n, ops):
        self.n = n
        self.ops = ops

    def __len__(self):
        return len(self.ops)

    def _column(self, op):
        cells = {}
        kind = op[0]
        if kind == "g":
            cells[op[2]] = op[1]
        elif kind == "cx":
            for c in op[1]:
                cells[c] = "●"
            cells[op[2]] = "X"
        elif kind == "cp":
            _, i, j, phi = op
            if abs(phi - math.pi) < 1e-12:
                cells[i] = cells[j] = "●"
            else:
                cells[i] = "P(%s)" % _fmt_angle(phi)
                cells[j] = "●"
        elif kind == "mc1":
            _, controls, target, m, label = op
            for c in controls:
                cells[c] = "●"
            cells[target] = label
        elif kind == "perm":
            indices, controls, name = op[1], op[2], op[3]
            for c in controls:
                cells[c] = "●"
            for i in indices:
                cells[i] = "[%s]" % name
        elif kind == "M":
            for i in op[1]:
                cells[i] = "M"
        return cells

    def __str__(self):
        cols = []
        for op in self.ops:
            cells = self._column(op)
            lo, hi = min(cells), max(cells)
            width = max(len(s) for s in cells.values())
            col = []
            for r in range(self.n):
                if r in cells:
                    col.append(cells[r].center(width, "─"))
                elif lo < r < hi:
                    col.append("│".center(width, "─"))
                else:
                    col.append("─" * width)
            cols.append(col)
        w = len(str(self.n - 1))
        rows = []
        for r in range(self.n):
            row = "─".join(col[r] for col in cols)
            rows.append("q%-*d: ─%s─" % (w, r, row))
        return "\n".join(rows)

    def __repr__(self):
        return str(self)

    def to_cirq(self):
        """Compiles the circuit into a cirq.Circuit (needs cirq installed)."""
        import cirq
        import numpy as np
        named = {
            "H": cirq.H,
            "X": cirq.X,
            "Y": cirq.Y,
            "Z": cirq.Z,
            "S": cirq.S,
            "T": cirq.T,
            "Sdg": cirq.S**-1,
            "Tdg": cirq.T**-1
        }
        q = cirq.LineQubit.range(self.n)
        circuit = cirq.Circuit()
        mcount = 0
        for op in self.ops:
            kind = op[0]
            if kind == "g":
                label, i, m = op[1], op[2], op[3]
                if label in named:
                    gate = named[label](q[i])
                else:
                    gate = cirq.MatrixGate(np.array(m), name=label)(q[i])
            elif kind == "cx":
                controls, t = op[1], op[2]
                gate = cirq.X(q[t]).controlled_by(*(q[c] for c in controls))
            elif kind == "cp":
                _, i, j, phi = op
                gate = cirq.CZ(q[i], q[j])**(phi / math.pi)
            elif kind == "mc1":
                _, controls, target, m, label = op
                gate = cirq.MatrixGate(np.array(m), name=label)(q[target])
                gate = gate.controlled_by(*(q[c] for c in controls))
            elif kind == "perm":
                indices, controls, name, table = op[1], op[2], op[3], op[4]
                size = len(table)
                mat = np.zeros((size, size))
                for v, w in enumerate(table):
                    mat[w][v] = 1
                gate = cirq.MatrixGate(mat, name=name)(*(q[i] for i in indices))
                if controls:
                    gate = gate.controlled_by(*(q[c] for c in controls))
            elif kind == "M":
                gate = cirq.measure(*(q[i] for i in op[1]), key="m%d" % mcount)
                mcount += 1
            else:
                raise QuantumError("unknown operation in the log: %r" % kind)
            circuit.append(gate)
        return circuit

    def to_qasm(self):
        """Compiles the circuit into OpenQASM 2.0 (a string).

        Named gates map to qelib1; the rest becomes u3 via an exact ZYZ
        decomposition. ``permute``, powers with several controls and
        CNOTs with more than two controls do not export.
        """
        named = {
            "H": "h",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "S": "s",
            "Sdg": "sdg",
            "T": "t",
            "Tdg": "tdg"
        }
        n_bits = sum(len(op[1]) for op in self.ops if op[0] == "M")
        lines = [
            "OPENQASM 2.0;", 'include "qelib1.inc";', "",
            "qreg q[%d];" % self.n
        ]
        if n_bits:
            lines.append("creg c[%d];" % n_bits)
        lines.append("")
        slot = 0
        for op in self.ops:
            kind = op[0]
            if kind == "g":
                label, i, m = op[1], op[2], op[3]
                if label in named:
                    lines.append("%s q[%d];" % (named[label], i))
                else:
                    theta, phi, lam, gamma = _zyz(m)
                    # the global phase is unobservable and dropped
                    lines.append("u3(%s,%s,%s) q[%d];  // %s" %
                                 (_qasm_angle(theta), _qasm_angle(phi),
                                  _qasm_angle(lam), i, label))
            elif kind == "cx":
                controls, t = op[1], op[2]
                if len(controls) == 1:
                    lines.append("cx q[%d],q[%d];" % (controls[0], t))
                elif len(controls) == 2:
                    lines.append("ccx q[%d],q[%d],q[%d];" %
                                 (controls[0], controls[1], t))
                else:
                    raise QuantumError(
                        "QASM export: more than 2 controls is not supported")
            elif kind == "cp":
                _, i, j, phi = op
                if abs(phi - math.pi) < 1e-12:
                    lines.append("cz q[%d],q[%d];" % (i, j))
                else:
                    lines.append("cu1(%s) q[%d],q[%d];" %
                                 (_qasm_angle(phi), i, j))
            elif kind == "mc1":
                _, controls, target, m, label = op
                if len(controls) != 1:
                    raise QuantumError(
                        "QASM export: multi-controlled %s is not supported" %
                        label)
                theta, phi, lam, gamma = _zyz(m)
                c = controls[0]
                if abs(gamma) > 1e-9:
                    # controlled global phase is a real phase on the control
                    lines.append("u1(%s) q[%d];  // phase of %s" %
                                 (_qasm_angle(gamma), c, label))
                lines.append("cu3(%s,%s,%s) q[%d],q[%d];  // %s" %
                             (_qasm_angle(theta), _qasm_angle(phi),
                              _qasm_angle(lam), c, target, label))
            elif kind == "perm":
                raise QuantumError(
                    "QASM export: permute has no QASM counterpart")
            elif kind == "M":
                for i in op[1]:
                    lines.append("measure q[%d] -> c[%d];" % (i, slot))
                    slot += 1
            else:
                raise QuantumError("unknown operation in the log: %r" % kind)
        return "\n".join(lines) + "\n"


class Observable:
    """The classical measurement result: m << a or m << a[i]."""

    def __init__(self):
        self.bits = ""

    def __lshift__(self, source):
        if isinstance(source, GateHandle):
            source = source.ref
        if isinstance(source, Qubits):
            self.bits += source._measure_all()
        elif isinstance(source, QubitSlice):
            self.bits += source.reg._measure_subset(source.indices)
        elif isinstance(source, QubitRef):
            self.bits += str(source.measure())
        elif isinstance(source, bool) or source in (0, 1):
            # a classical bit
            self.bits += str(int(source))
        else:
            raise QuantumError("only Qubits, a slice, a qubit or a classical "
                               "bit can be measured")
        return self

    @property
    def value(self):
        if not self.bits:
            raise QuantumError("nothing has been measured yet")
        return int(self.bits, 2)

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __str__(self):
        return self.bits if self.bits else "?"

    def __repr__(self):
        return "Observable(%s)" % self

    def __eq__(self, other):
        if isinstance(other, str):
            return self.bits == other
        if isinstance(other, int):
            return bool(self.bits) and self.value == other
        if isinstance(other, Observable):
            return self.bits == other.bits
        return NotImplemented

    def __hash__(self):
        return hash(self.bits)
