# Quantum TNCO Portal

A small web page that shows what `TNCO` does. Write a quantum program — the
portal draws its circuit, builds the tensor network for the program's
observations, compares the natural contraction plan with the optimized one,
animates the annealer live, and lets you draw the input state by hand.

Programs can be written in `quantum.py` (a small Python DSL, see the presets)
or in OpenQASM 2.0; the editor auto-detects the language, and the *compiled
view* toggle shows the same circuit translated into the other one. When only
some qubits are measured, the network is the two-layer ⟨ψ|…|ψ⟩ sandwich that
computes the observation probabilities. Up to 12 qubits.

The optimized tree and the live view come from an exact JavaScript port of
`TNCO`'s simulated annealing, specialized to these networks (every dimension
is 2): the same moves, the same acceptance probability.

## Run

No dependencies beyond Python 3:

```bash
python3 quantum_portal_server.py
```

Then open <http://localhost:8991/> (`--port` moves it).
