# Two-qubit entanglement detection from coherence

Code and reference data for *Coherence Representations and Machine Learning for Two-Qubit Entanglement Detection*, Guillermo Vital Godinez, Maestría en Matemáticas, Universidad Autónoma del Estado de Hidalgo. Appendices A–C describe the state generation, counterexample search and evaluation protocol.

Two questions motivate the code. How accurately can a neural network classify the states in the experiment from four coherence values? Can those same values distinguish every separable state from every entangled one? The saved predictions let you examine the first question on 105,000 held-out states. An explicit pair of Bell-diagonal states with opposite PPT labels answers the second.

## Get the files

Clone this repository, or download and extract the ZIP from GitHub's **Code** menu. Run the commands below from the directory containing this README. The recorded environment uses Python 3.11.3. The commands use macOS/Linux shell syntax.

```bash
git clone https://github.com/gvitalgo/Entanglement-detection.git
cd Entanglement-detection
```

## Verify the counterexample (Appendix B)

This path only needs NumPy, SciPy and Matplotlib. It finds the canonical pair and searches the 28 candidates in the thesis grid.

```bash
python3.11 -m venv .venv-counterexample
.venv-counterexample/bin/python -m pip install -r requirements-counterexample.txt
.venv-counterexample/bin/python scripts/search_coherence_counterexamples.py \
  --output-dir artifacts/coherence_counterexamples --plot
```

The minimum partial-transpose eigenvalues of the canonical pair are −0.01 and +0.01. The four numerical coordinates agree to within `1e-12`. The [walkthrough](docs/coherence_counterexample.md) explains the exact entropy equation, its sign certificate, the phase convention and the optional independent QuTiP check.

## Recover the reported tables and figure (Appendices A.4 and C)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/build_revision_evidence.py --output-dir artifacts/revision_evidence
```

The last command verifies the test indices, labels and aggregate accuracies against the stored predictions. It writes the per-family, seed and Bell-path crossing tables, a JSON file with the family error counts, and the score plot. The [reference outputs](thesis/generated/revision_evidence/) are included for comparison.

## Train the networks again (Appendix A)

```bash
.venv/bin/python scripts/run_thesis_experiment.py \
  --config configs/thesis_finalization.toml --output-dir artifacts/fresh
.venv/bin/python scripts/build_revision_evidence.py \
  --run-dir artifacts/fresh --output-dir artifacts/fresh_evidence
```

Use a new or empty directory for the training run. The configuration verifies the SHA-256 of the archived 350,000 feature rows, reserves 30% for testing and fits the scaler on the outer development set before Keras takes its validation subset. The main run uses seed 42. The two additional model seeds keep the data and split fixed. Retraining can produce small numerical differences across environments.

To generate new states and compute their features independently, run the following command. The saved dataset contains four natural-log coherence coordinates, the PPT label and the family identifier for each row. The original full density matrices were not retained.

```bash
.venv/bin/python -m asif_reproduction generate \
  --full-dataset --seed 42 --phi 0 --output-dir artifacts/regenerated
```

The [run directory](thesis/generated/ml_finalization_seed42/) contains the weights, scaler, split indices, predictions, metrics and run manifest. The public metadata uses relative project paths and omits the original machine's display inventory. The archived feature data have SHA-256 `3d563b789ab0293fbfeed6888b57ddd943d2c6bcac41a4d27f856540d1675cb3`. Proposition 5.1 of the thesis proves the four-bit bound used in the recorded witness baseline.
