# Quantum TNCO Portal

A small web page that shows what `TNCO` does. Write a quantum program — the
portal draws its circuit, lays out the tensor network of the program's
observations as a live force-directed graph, compares the natural contraction
plan with the optimized one, animates the annealer live, replays any tree as
a step-by-step contraction of the network, and lets you draw the input state
by hand.

Programs can be written in `quantum.py` (a small Python DSL, see the presets)
or in OpenQASM 2.0; the editor auto-detects the language, and the *compiled
view* toggle shows the same circuit translated into the other one. When only
some qubits are measured, the network is the two-layer ⟨ψ|…|ψ⟩ sandwich that
computes the observation probabilities. Up to 12 qubits. The DSL includes
the Sycamore gate set (`fSim`, `iSWAP`, `W` and gate powers) and
post-selection (`q.returns(b)`: the qubit must end in `b`; the branch weight
is the whole-outcome probability), and the presets include a miniature
Sycamore cycle and OTOC echo circuits.

The optimized tree and the live view come from an exact JavaScript port of
`TNCO`'s simulated annealing, specialized to these networks (every dimension
is 2): the same moves, the same acceptance probability.

## Run

No dependencies beyond Python 3:

```bash
python3 quantum_portal_server.py
```

Then open <http://localhost:8991/> (`--port` moves it).

## Standalone

`python3 make_standalone.py` builds `../docs/portal.html` — the same page
with the back end moved inside (Python runs in the browser via Pyodide), so
it opens with a double click and needs the network only the first time, to
fetch Pyodide from its CDN. The same file is what GitHub Pages serves.
