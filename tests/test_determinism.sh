#!/bin/bash
set -e

# Create random circuit
echo -ne 'Building circuit ... '
python -c 'import cirq; cirq.to_json(cirq.testing.random_circuit(8, 16, 1), "circuit.cirq")'
echo 'Done!'

# Fix seed
SEED=${RANDOM}

# Run tnco twice with the same seed
echo -ne 'Optimizing circuit ... '
PYTHONHASHSEED=${RANDOM} tnco optimize circuit.cirq '(0, 1e5)' --n-steps=10 \
                                          --n-runs=2 \
                                          --verbose=0 \
                                          --seed=${SEED} \
                                          --output-filename=res_${SEED}_1.json
echo 'Done!'

echo -ne 'Optimizing circuit ... '
PYTHONHASHSEED=${RANDOM} tnco optimize circuit.cirq '(0, 1e5)' --n-steps=10 \
                                          --n-runs=2 \
                                          --verbose=0 \
                                          --seed=${SEED} \
                                          --output-filename=res_${SEED}_2.json
echo 'Done!'

# Check difference
diff <(cat res_${SEED}_1.json | jq . | grep -v runtime) \
     <(cat res_${SEED}_2.json | jq . | grep -v runtime)
