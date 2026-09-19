#!/usr/bin/env python3
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
"""Quantum TNCO Portal: program -> circuit -> network -> trees -> sandbox.

Run:

    python3 quantum_portal_server.py

Then open http://localhost:8991/
"""

import contextlib
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback

import quantum
from quantum import Observable
from quantum import Qubits

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8991
MAX_QUBITS = 12

# ------------------------------------------------------------ QASM support


def detect_lang(code):
    """Returns 'qasm' if the text looks like OpenQASM, else 'python'."""
    if re.search(r'^\s*OPENQASM\b|^\s*qreg\s', code, re.M):
        return 'qasm'
    return 'python'


_ANGLE_RE = re.compile(r'[0-9pi+\-*/(). eE]*')

# gate name -> (n_qubits, python template); {q} are qubit refs, {a} angles
_QASM_GATES = {
    'h': (1, '{q0}.H()'),
    'x': (1, '{q0}.X()'),
    'y': (1, '{q0}.Y()'),
    'z': (1, '{q0}.Z()'),
    's': (1, '{q0}.S()'),
    'sdg': (1, '{q0}.Sdg()'),
    't': (1, '{q0}.T()'),
    'tdg': (1, '{q0}.Tdg()'),
    'rx': (1, '{q0}.RX({a0})'),
    'ry': (1, '{q0}.RY({a0})'),
    'rz': (1, '{q0}.RZ({a0})'),
    'p': (1, '{q0}.P({a0})'),
    'u1': (1, '{q0}.P({a0})'),
    'cx': (2, '{q1} ^= {q0}'),
    'cz': (2, '{q0}.CZ({q1})'),
    'cp': (2, '{q0}.CP({q1}, {a0})'),
    'cu1': (2, '{q0}.CP({q1}, {a0})'),
    'ccx': (3, '{q2} ^= {q0} & {q1}'),
}


def qasm_to_python(code):
    """Translates an OpenQASM 2.0 subset into a quantum.py program.

    Args:
        code: OpenQASM 2.0 source.

    Returns:
        The equivalent quantum.py program, as Python source.
    """
    text = re.sub(r'//[^\n]*', '', code)
    stmts = [s.strip() for s in re.sub(r'\s+', ' ', text).split(';')]

    qregs, body = {}, []  # qregs: name -> (offset, size)
    cregs = {}  # name -> [size, bits filled so far]
    total = 0

    def qref(token, broadcastable=False):
        token = token.strip()
        m = re.fullmatch(r'(\w+)\s*\[\s*(\d+)\s*\]', token)
        if m:
            name, idx = m.group(1), int(m.group(2))
            if name not in qregs:
                raise quantum.QuantumError('QASM: unknown qreg %r' % name)
            if idx >= qregs[name][1]:
                raise quantum.QuantumError('QASM: %s[%d] is out of range' %
                                           (name, idx))
            return ['%s[%d]' % (name, idx)]
        if token in qregs and broadcastable:
            return ['%s[%d]' % (token, i) for i in range(qregs[token][1])]
        raise quantum.QuantumError('QASM: cannot parse qubit %r' % token)

    for s in stmts:
        if not s or s.startswith(('OPENQASM', 'include', 'barrier', 'opaque')):
            continue
        if s.startswith('if'):
            raise quantum.QuantumError(
                'QASM: classical control (if) is not supported')
        m = re.fullmatch(r'qreg (\w+) ?\[ ?(\d+) ?\]', s)
        if m:
            qregs[m.group(1)] = (total, int(m.group(2)))
            total += int(m.group(2))
            continue
        m = re.fullmatch(r'creg (\w+) ?\[ ?(\d+) ?\]', s)
        if m:
            cregs[m.group(1)] = [int(m.group(2)), 0]
            continue
        # bits land in the creg in program order; an explicit index
        # must agree with that, and the creg must not overflow
        m = re.fullmatch(r'measure (\S+) ?-> ?(\w+)(?: ?\[ ?(\d+) ?\])?', s)
        if m:
            creg, idx = m.group(2), m.group(3)
            if creg not in cregs:
                raise quantum.QuantumError('QASM: unknown creg %r' % creg)
            size, filled = cregs[creg]
            if idx is not None and int(idx) != filled:
                raise quantum.QuantumError(
                    'QASM: bits fill a creg in program order; '
                    'expected %s[%d] here' % (creg, filled))
            qs = qref(m.group(1), broadcastable=True)
            if filled + len(qs) > size:
                raise quantum.QuantumError('QASM: creg %s[%d] overflows' %
                                           (creg, size))
            cregs[creg][1] = filled + len(qs)
            for q in qs:
                body.append('%s << %s' % (creg, q))
            continue
        m = re.fullmatch(r'(\w+) ?(?:\( ?([^)]*) ?\))? (.+)', s)
        if m:
            name, args, qubits = m.group(1), m.group(2), m.group(3)
            args = [a.strip() for a in args.split(',')] if args else []
            for a in args:
                if not _ANGLE_RE.fullmatch(a):
                    raise quantum.QuantumError('QASM: bad angle %r' % a)
            qtoks = [t.strip() for t in qubits.split(',')]
            if name == 'returns':
                if len(args) != 1 or args[0] not in ('0', '1'):
                    raise quantum.QuantumError(
                        'QASM: returns expects one bit, 0 or 1')
                for q in qref(qtoks[0], broadcastable=True):
                    body.append('%s.returns(%s)' % (q, args[0]))
                continue
            if name == 'swap':
                if len(qtoks) != 2:
                    raise quantum.QuantumError('QASM: swap expects 2 qubits')
                a, b = qref(qtoks[0])[0], qref(qtoks[1])[0]
                body += [
                    '%s ^= %s' % (b, a),
                    '%s ^= %s' % (a, b),
                    '%s ^= %s' % (b, a)
                ]
                continue
            if name in ('u2', 'u3', 'u'):
                if name == 'u2':
                    args = ['pi/2'] + args
                if len(args) != 3 or len(qtoks) != 1:
                    raise quantum.QuantumError('QASM: cannot parse %r' % s)
                th, ph, lam = args
                for q in qref(qtoks[0], broadcastable=True):
                    body += [
                        '%s.RZ(%s)' % (q, lam),
                        '%s.RY(%s)' % (q, th),
                        '%s.RZ(%s)' % (q, ph)
                    ]
                continue
            if name == 'cu3':
                # the qelib1 definition of cu3, in DSL gates
                if len(args) != 3 or len(qtoks) != 2:
                    raise quantum.QuantumError('QASM: cannot parse %r' % s)
                th, ph, lam = args
                c, tq = qref(qtoks[0])[0], qref(qtoks[1])[0]
                body += [
                    '%s.P((%s+%s)/2)' % (c, lam, ph),
                    '%s.P((%s-%s)/2)' % (tq, lam, ph),
                    '%s ^= %s' % (tq, c),
                    '%s.RZ(-(%s+%s)/2)' % (tq, ph, lam),
                    '%s.RY(-(%s)/2)' % (tq, th),
                    '%s ^= %s' % (tq, c),
                    '%s.RY((%s)/2)' % (tq, th),
                    '%s.RZ(%s)' % (tq, ph)
                ]
                continue
            if name not in _QASM_GATES:
                raise quantum.QuantumError('QASM: gate %r is not supported' %
                                           name)
            arity, tpl = _QASM_GATES[name]
            if arity == 1:
                if len(qtoks) != 1:
                    raise quantum.QuantumError(
                        'QASM: %s expects one qubit operand' % name)
                for q in qref(qtoks[0], broadcastable=True):
                    body.append(tpl.format(q0=q, a0=args[0] if args else ''))
            else:
                if len(qtoks) != arity:
                    raise quantum.QuantumError('QASM: %s expects %d qubits' %
                                               (name, arity))
                refs = [qref(t)[0] for t in qtoks]
                fields = {'q%d' % i: r for i, r in enumerate(refs)}
                if args:
                    fields['a0'] = args[0]
                body.append(tpl.format(**fields))
            continue
        raise quantum.QuantumError('QASM: cannot parse %r' % s)

    if not qregs:
        raise quantum.QuantumError('QASM: no qreg declared')

    lines = ['# compiled from OpenQASM 2.0']
    for c in cregs:
        lines.append('%s = Observable()' % c)
    lines.append('with Qubits(%d) as _reg:' % total)
    for name, (off, size) in qregs.items():
        lines.append('    %s = _reg[%d:%d]' % (name, off, off + size))
    for b in body:
        lines.append('    ' + b)
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- DSL exec


def run_program(code, init_state=None, seed=None):
    """Runs a program in either language.

    QASM is compiled to a quantum.py program first: one executor for both.

    Args:
        code: the program, quantum.py or OpenQASM 2.0 (auto-detected).
        init_state: optional initial amplitudes, 2^n reals (normalized
            here); replaces |0...0> when its length matches the register.
        seed: optional seed for the measurement randomness.

    Returns:
        (register, {observable name: bits}, captured stdout).
    """
    if detect_lang(code) == 'qasm':
        code = qasm_to_python(code)
    if seed is not None:
        random.seed(seed)

    registers = []

    class PortalQubits(Qubits):

        def __init__(self, n):
            if n > MAX_QUBITS:
                raise quantum.QuantumError(
                    'the portal is limited to %d qubits' % MAX_QUBITS)
            super().__init__(n)
            if init_state is not None and len(init_state) == (1 << n):
                norm = math.sqrt(sum(x * x for x in init_state))
                if norm > 1e-9:
                    self.state = [complex(x / norm, 0) for x in init_state]
            registers.append(self)

    ns = {'__builtins__': __builtins__}
    for name in quantum.__all__:
        ns[name] = getattr(quantum, name)
    ns['Qubits'] = PortalQubits
    ns['math'] = math
    ns['pi'] = math.pi

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, '<program>', 'exec'), ns)

    if not registers:
        raise quantum.QuantumError('the program has no Qubits(n) register')
    reg = max(registers, key=lambda r: len(r._log))
    obs = {
        k: v.bits
        for k, v in ns.items()
        if isinstance(v, Observable) and not k.startswith('_')
    }
    return reg, obs, out.getvalue()[-4000:]


# ------------------------------------------------------ circuit serializer


def circuit_json(reg):
    ops = []
    for op in reg._log:
        kind = op[0]
        if kind == 'g':
            ops.append({'k': 'g', 'label': op[1], 'q': op[2]})
        elif kind == 'cx':
            ops.append({'k': 'cx', 'controls': list(op[1]), 'target': op[2]})
        elif kind == 'cp':
            ops.append({'k': 'cp', 'i': op[1], 'j': op[2], 'phi': op[3]})
        elif kind == 'mc1':
            ops.append({
                'k': 'mc1',
                'controls': list(op[1]),
                'target': op[2],
                'label': op[4]
            })
        elif kind == 'perm':
            ops.append({
                'k': 'perm',
                'q': list(op[1]),
                'controls': list(op[2]),
                'name': op[3]
            })
        elif kind == 'u2':
            ops.append({'k': 'u2', 'i': op[1], 'j': op[2], 'label': op[6]})
        elif kind == 'ret':
            ops.append({'k': 'ret', 'q': op[1], 'b': op[2]})
        elif kind == 'M':
            ops.append({'k': 'M', 'q': list(op[1])})
    return {'n': reg.n, 'ops': ops}


# ------------------------------------------------------------ tensor network


def _op_qubits_label(op):
    kind = op[0]
    if kind == 'g':
        return op[1], (op[2],)
    if kind == 'cx':
        return ('C' * len(op[1])) + 'X', tuple(op[1]) + (op[2],)
    if kind == 'mc1':
        return ('C' * len(op[1])) + op[4], tuple(op[1]) + (op[2],)
    if kind == 'cp':
        label = 'CZ' if abs(op[3] - math.pi) < 1e-9 else 'CP'
        return label, (op[1], op[2])
    if kind == 'perm':
        return op[3], tuple(op[1]) + tuple(op[2])
    if kind == 'u2':
        return op[6], (op[1], op[2])
    return '?', ()


class _TN:
    """The portal's tensor network: indices are line numbers, dims are 2."""

    def __init__(self, idx_names, where):
        order, ts = [], {}
        for k, name in enumerate(idx_names):
            for member in where[name]:
                if member == '*':
                    continue
                if member not in ts:
                    order.append(member)
                    ts[member] = []
                ts[member].append(k)
        self.ts_inds = tuple(tuple(ts[m]) for m in order)
        self.ts_tags = tuple({'name': m} for m in order)
        self.n_tensors = len(order)
        self.n_inds = len(idx_names)


def build_tn(reg):
    """Builds the tensor network for the program's observations.

    With a partial measurement this is the two-layer <psi|...|psi> sandwich:
    unmeasured qubits are glued between the layers, measured legs stay open,
    so the network computes the observation distribution. Otherwise it is a
    single layer with every output open. A returned qubit
    (q.returns(b)) ends in a cap |b> instead: its leg is closed on every
    layer it has.

    Args:
        reg: the executed register; its log defines the circuit.

    Returns:
        (tn, labels, idx_names, raw_names, net_ops): the network, display
        labels and raw names per tensor, readable index names per line,
        and {label, q} per gate.
    """
    ops = [op for op in reg._log if op[0] not in ('M', 'ret')]
    rets = {op[1]: op[2] for op in reg._log if op[0] == 'ret'}
    if not ops:
        raise quantum.QuantumError(
            'the program has no gates — nothing to build a network from')
    measured = sorted({q for op in reg._log if op[0] == 'M' for q in op[1]})
    # all qubits measured: the layers would disconnect; one layer is enough
    if len(measured) == reg.n:
        measured = []

    idx_names, where = [], {}

    def new_idx(name):
        idx_names.append(name)
        where[name] = []
        return name

    def build_layer(layer):
        tag = '' if layer == 'k' else '′'
        cur = {}
        for q in range(reg.n):
            name = ('|0⟩q%d' % q) if layer == 'k' else ('⟨0|q%d' % q)
            wire = new_idx('q%d%s·0' % (q, tag))
            where[wire].append(name)
            cur[q] = wire
        for t, op in enumerate(ops):
            label, qs = _op_qubits_label(op)
            name = '%s%s#%d' % (label, '†' if layer == 'b' else '', t)
            for q in qs:
                where[cur[q]].append(name)
                cur[q] = new_idx('q%d%s·%d' % (q, tag, t + 1))
                where[cur[q]].append(name)
        return cur

    cur_k = build_layer('k')
    if measured:
        cur_b = build_layer('b')
        for q in range(reg.n):
            if q in rets:
                # the future observer closes each bracket: the ket layer
                # ends in a bra, the bra layer ends in a ket
                where[cur_k[q]].append('⟨%d|q%d·ret' % (rets[q], q))
                where[cur_b[q]].append('|%d⟩q%d·ret' % (rets[q], q))
            elif q in measured:
                where[cur_k[q]].append('*')
                where[cur_b[q]].append('*')
            else:
                # glue: the bra leg is the ket leg
                where[cur_k[q]].extend(where.pop(cur_b[q]))
                idx_names.remove(cur_b[q])
    else:
        for q in range(reg.n):
            if q in rets:
                where[cur_k[q]].append('⟨%d|q%d·ret' % (rets[q], q))
            else:
                where[cur_k[q]].append('*')

    tn = _TN(idx_names, where)

    raw_names = [t['name'] for t in tn.ts_tags]
    labels = [n.split('#')[0].split('·')[0] for n in raw_names]
    net_ops = []
    for label, qs in map(_op_qubits_label, ops):
        net_ops.append({'label': label, 'q': sorted(qs)})

    return tn, labels, idx_names, raw_names, net_ops


def compile_tn(reg):
    tn, labels, idx_names, raw_names, net_ops = build_tn(reg)
    return {
        'n_tensors': tn.n_tensors,
        'n_inds': tn.n_inds,
        'strip': {
            'n': reg.n,
            'gates': net_ops
        },
        # both contraction trees are built by the page's annealer
        'network': {
            'ts': [list(x) for x in tn.ts_inds],
            'n_inds': tn.n_inds,
            'labels': labels,
            'raw': raw_names,
            'idx_names': idx_names
        }
    }


# ------------------------------------------------------------------ server


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'quantum_portal.html'), 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(length) or b'{}')
            code = req.get('code', '')
            if self.path == '/compile':
                reg, obs, stdout = run_program(code, seed=req.get('seed'))
                out = {
                    'circuit': circuit_json(reg),
                    'observables': obs,
                    'stdout': stdout,
                    'weight': reg.weight
                }
                out.update(compile_tn(reg))
                self._json(out)
            elif self.path == '/translate':
                # same circuit, the other language
                lang = detect_lang(code)
                if lang == 'python':
                    reg, _, _ = run_program(code, seed=req.get('seed'))
                    self._json({
                        'lang': 'qasm',
                        'text': reg.as_circuit.to_qasm()
                    })
                else:
                    self._json({'lang': 'python', 'text': qasm_to_python(code)})
            elif self.path == '/observe':
                # histogram of the observables over many runs
                base = req.get('seed')
                if base is None:
                    base = random.randrange(2**31)
                counts, names = {}, None
                t0, n_runs = time.time(), 0
                while n_runs < 500 and time.time() - t0 < 1.5:
                    _, obs, _ = run_program(code,
                                            init_state=req.get('init_state'),
                                            seed=base + n_runs)
                    if not obs:
                        raise quantum.QuantumError(
                            'the program has no observations (Observable)')
                    if names is None:
                        names = list(obs.keys())
                    key = ' '.join(obs[k] or '—' for k in names)
                    counts[key] = counts.get(key, 0) + 1
                    n_runs += 1
                self._json({'names': names, 'counts': counts, 'n': n_runs})
            elif self.path == '/simulate':
                reg, obs, stdout = run_program(code,
                                               init_state=req.get('init_state'),
                                               seed=req.get('seed'))
                self._json({
                    'n': reg.n,
                    'weight': reg.weight,
                    'distribution': reg.distribution_as_list,
                    'amplitudes': [[z.real, z.imag] for z in reg.amplitudes],
                    'observables': obs,
                    'stdout': stdout
                })
            else:
                self._json({'error': 'not found'}, 404)
        except Exception:
            self._json(
                {
                    'error': traceback.format_exc(limit=3).splitlines()[-1],
                    'trace': traceback.format_exc(limit=6)
                }, 400)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--port',
                    type=int,
                    default=PORT,
                    help='port to listen on (default: %d)' % PORT)
    PORT = ap.parse_args().port
    try:
        httpd = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    except OSError as e:
        sys.exit('cannot listen on port %d: %s\n'
                 'Another portal is probably already running there; '
                 'pass --port to pick a free one.' % (PORT, e))
    print('Quantum TNCO Portal: http://localhost:%d/' % PORT)
    httpd.serve_forever()
