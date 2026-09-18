# VECTOR: Vector-Guided Euclidean Chemical Trajectory & Organic Recombination Platform

VECTOR is a chemoinformatics platform that generates novel chemical structures directly from user-provided seed molecules. It is integrated with multidimensional molecular descriptors, heuristic priority-guided synthon assembly (BRICS), automated synthetic accessibility filtering (SAScore), multithreaded 3D distance geometry conformer embedding (ETKDGv3), force-field minimization (MMFF94), USRCAT 3D shape/pharmacophore matching, multi-objective Pareto-frontier optimization, and optional target receptor pocket docking (AutoDock Vina).

VECTOR can be used for general chemical discovery and lead optimization across arbitrary chemical domains (medicinal chemistry, agrochemicals, material precursors, and chemical biology).

## 1. Scope & Boundary Conditions

* **Target Domain**: Small molecules (Molecular Weight $\text{MW} < 700\text{ g/mol}$, heavy atom count $< 50$).
* **Explicit Exclusions**: VECTOR is strictly **not intended for macrocycles (rings $> 12$ members), cyclic peptides, linear peptides, or biologics**. Retrosynthetic cleavage rules, standard distance geometry bounds, and QED desirability models are calibrated specifically for small-molecule chemical space.
* **Seed Cohort Guidance**:
  * **$3 \le K \le 5$ Seeds**: Focused Analogue Exploration (suitable for sparse early-stage hit series).
  * **$6 \le K \le 10+$ Seeds**: Optimal Multidimensional Diversity (yields broad combinatorial cross-hybridization and robust centroid estimation).

## 2. Algorithmic Architecture

### 2.1 Heuristic Priority-Guided Chemical Sampling
Rather than unguided combinatorial search, VECTOR implements an **intelligent priority-guided beam search**:
1. Input molecules are decomposed into labeled retro-synthetic synthons via BRICS rules.
2. Individual fragments are pre-evaluated in normalized descriptor space against the target centroid $\vec{c}_{\text{norm}}$.
3. Synthon assembly prioritizes fragments and partial assemblies that lie along the vector pointing directly toward the target centroid.
4. Early Synthetic Accessibility Filtering (`SAScore <= 5.0`) prunes un-synthesizable, highly strained, or bridged structures prior to computationally demanding 3D calculations.

### 2.2 Automated Synthetic Accessibility Scoring (SAScore)
To ensure discovered candidates are practically synthesizable in a lab, VECTOR integrates Ertl & Schuffenhauer's Synthetic Accessibility Score:
* Quantifies molecular complexity based on fragment contributions derived from millions of commercially available chemicals combined with topological ring complexity penalties.
* Candidates with $\text{SAScore} > 5.0$ are eliminated before 3D conformer embedding.

### 2.3 3-Stage 3D Conformer Generation & Biophysical Profiling
For all molecules passing 2D descriptor and SAScore filters, 3D conformers are generated in parallel across all CPU cores:
1. **Stage 1 (Primary)**: Distance Geometry with Experimental Torsion Knowledge (`ETKDGv3`, `randomSeed=0xf00d`, failure tracking).
2. **Stage 2 (Fallback 1)**: Extended conformational search (`maxAttempts=5000`) for strained ring systems.
3. **Stage 3 (Fallback 2)**: Random coordinate projection (`useRandomCoords=True`).
4. **Force-Field Minimization**: MMFF94 force-field relaxation (500 iterations) to relieve steric strain and van der Waals clashes, recording internal strain energy ($E_{\text{strain}} = E_{\text{initial}} - E_{\text{opt}}$).
5. **USRCAT 3D Pharmacophore Similarity**: 60-dimensional shape and pharmacophore distance moments tracking geometric, charge, and hydrogen bonding distributions relative to the seed cohort conformers.

### 2.4 Multi-Objective Pareto-Frontier Optimization
Rather than rigid linear sorting, VECTOR identifies the **non-dominated 2D Pareto frontier** balancing:
1. **Minimizing Euclidean Descriptor Distance** $d(\vec{x}, \vec{c}_{\text{norm}})$.
2. **Maximizing Drug-Likeness (QED)**.

A compound $A$ dominates $B$ if $d_A \le d_B$ and $\text{QED}_A \ge \text{QED}_B$ with at least one strict inequality. Pareto-optimal candidates are surfaced at the top of the discovery table.

### 2.5 Optional Target Protein Pocket Docking (AutoDock Vina)
When a target protein structure is available, users can upload a receptor file (`.pdb` or `.pdbqt`) or click **Load Example Protein** (`protein.pdb`):
* PDB files are converted to PDBQT format using Open Babel and Meeko.
* AutoDock Vina computes target binding affinity ($\text{kcal/mol}$) within the defined binding grid.
* Docking scores are directly integrated into the candidate table and exported SDF tags.

## 3. Directory Structure

```
.
├── app.py              # Streamlit GUI, Pareto Visualizer, & Docking Interface
├── pipeline.py         # Chemoinformatics Engine, Priority Assembler, 3D Conformer, & Vina Module
├── run_app.sh          # Native Shell Launcher Script
├── README.md           # Technical Documentation & User Guide
├── protein.pdb         # Example Target Protein Structure (Ready for Docking)
└── environment.yml     # Complete Conda Environment Specification
```

## 4. Installation & Setup

### Requirements
* Linux / macOS / WSL
* Conda / Mamba (Miniforge recommended)

### Step 1: Environment Provisioning
```bash
cd /home/igris/molgen
conda env create --prefix ./env -f environment.yml
```

### Step 2: Launch Platform
```bash
./run_app.sh
```
Or directly via the python environment:
```bash
./env/bin/streamlit run app.py
```
Open your browser at: `http://localhost:8501`

## 5. Usage Instructions

1. **Input Seed Molecules**:
   * Click **Load Example Seed Molecules** for a 1-click test suite.
   * Or upload a `.txt` file containing SMILES (one per line).
   * Or paste canonical SMILES directly into the text area.
2. **Adjust Optimization Parameters**:
   * Configure Maximum Euclidean Distance, Minimum QED, Maximum SAScore, and Lipinski Rule of 5 violations.
3. **Optional Target Protein Docking**:
   * Check **Enable Target Protein Pocket Docking**.
   * Click **Load Example Protein (protein.pdb)** to load the local receptor structure and auto-populate pocket center coordinates, or upload your own `.pdb` / `.pdbqt` file.
4. **Generate & Inspect in 2D / 3D**:
   * Click **Generate Novel Molecules**.
   * Inspect the interactive **Side-by-Side 2D Diagram & 3D WebGL Molecular Viewer** (rotate, pan, zoom, toggle between Stick and Sphere styles).
   * Review Pareto-optimal candidates, 3D MMFF94 energies, USRCAT shape similarities, and optional Vina binding affinities.
5. **Export Discovery Results**:
   * Download the results as a **CSV Table** or as an **SDF File with Minimized 3D Coordinates** ready for molecular visualization (PyMOL, ChimeraX, Maestro, MOE).

## License

This project is licensed under the [MPL-2.0 License](LICENSE).
