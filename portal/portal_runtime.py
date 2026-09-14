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
"""The standalone page's back end: the server's routes without the server.

Runs inside Pyodide. `api(path, body)` answers exactly like the HTTP
server's POST handler, JSON in and JSON out; the contraction trees are
not built here — the page builds both with its own annealer.
"""

import json
import random
import time
import traceback

import quantum_portal_server as S


def api(path, body):
    try:
        req = json.loads(body or '{}')
        code = req.get('code', '')
        if path == '/compile':
            reg, obs, stdout = S.run_program(code, seed=req.get('seed'))
            out = {
                'circuit': S.circuit_json(reg),
                'observables': obs,
                'stdout': stdout,
                'weight': reg.weight
            }
            out.update(S.compile_tn(reg))
            return json.dumps(out)
        if path == '/translate':
            if S.detect_lang(code) == 'python':
                reg, _, _ = S.run_program(code, seed=req.get('seed'))
                return json.dumps({
                    'lang': 'qasm',
                    'text': reg.as_circuit.to_qasm()
                })
            return json.dumps({
                'lang': 'python',
                'text': S.qasm_to_python(code)
            })
        if path == '/observe':
            base = req.get('seed')
            if base is None:
                base = random.randrange(2**31)
            counts, names = {}, None
            t0, n_runs = time.time(), 0
            while n_runs < 500 and time.time() - t0 < 1.5:
                _, obs, _ = S.run_program(code,
                                          init_state=req.get('init_state'),
                                          seed=base + n_runs)
                if not obs:
                    raise S.quantum.QuantumError(
                        'the program has no observations (Observable)')
                if names is None:
                    names = list(obs.keys())
                key = ' '.join(obs[k] or '—' for k in names)
                counts[key] = counts.get(key, 0) + 1
                n_runs += 1
            return json.dumps({'names': names, 'counts': counts, 'n': n_runs})
        if path == '/simulate':
            reg, obs, stdout = S.run_program(code,
                                             init_state=req.get('init_state'),
                                             seed=req.get('seed'))
            return json.dumps({
                'n': reg.n,
                'weight': reg.weight,
                'distribution': reg.distribution_as_list,
                'amplitudes': [[z.real, z.imag] for z in reg.amplitudes],
                'observables': obs,
                'stdout': stdout
            })
        return json.dumps({'error': 'not found'})
    except Exception:
        return json.dumps({
            'error': traceback.format_exc(limit=3).splitlines()[-1],
            'trace': traceback.format_exc(limit=6)
        })
