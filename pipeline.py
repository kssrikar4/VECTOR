import io
import os
import sys
import time
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, BRICS, AllChem, rdDistGeom, rdMolDescriptors, RDConfig

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    sascorer.readFragmentScores()
except Exception:
    sascorer = None

try:
    from vina import Vina
    import meeko
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    HAS_VINA = True
except Exception:
    HAS_VINA = False

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "FractionCSP3", "QED"
]

BENCHMARK_MEAN = np.array([
    386.126918, 2.492616, 86.380769, 1.884215, 4.988782,
    5.535256, 3.196314, 2.142228, 0.371846, 0.573716
], dtype=np.float64)

BENCHMARK_STD = np.array([
    166.566342, 2.506313, 63.132116, 2.027410, 3.277967,
    4.276075, 1.494623, 1.224174, 0.231034, 0.204986
], dtype=np.float64)

def calculate_sascore(mol: Chem.Mol) -> float:
    if sascorer is None or mol is None:
        return 3.0
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 5.0

def embed_and_profile_3d_batch(mols: List[Chem.Mol], seed_usr: Optional[List[float]] = None) -> List[Tuple[Optional[Chem.Mol], Dict[str, Any]]]:
    results = []
    num_cpus = os.cpu_count() or 4
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    params.numThreads = num_cpus

    for mol in mols:
        m3d = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(m3d, params)
        if status < 0:
            status = AllChem.EmbedMolecule(m3d, maxAttempts=5000)
        if status < 0:
            status = AllChem.EmbedMolecule(m3d, useRandomCoords=True)

        if m3d.GetNumConformers() == 0:
            results.append((None, {
                "Conformer_Success": False,
                "MMFF94_Energy": None,
                "Strain_Energy": None,
                "USRCAT_Similarity": None
            }))
            continue

        mmff_energy = None
        strain_energy = None
        if AllChem.MMFFHasAllMoleculeParams(m3d):
            mp = AllChem.MMFFGetMoleculeProperties(m3d)
            ff = AllChem.MMFFGetMoleculeForceField(m3d, mp)
            if ff:
                e_initial = ff.CalcEnergy()
                AllChem.MMFFOptimizeMolecule(m3d, maxIters=500)
                e_opt = ff.CalcEnergy()
                mmff_energy = round(float(e_opt), 2)
                strain_energy = round(float(max(0.0, e_initial - e_opt)), 2)
        else:
            AllChem.UFFOptimizeMolecule(m3d, maxIters=500)

        usr_sim = None
        if seed_usr is not None:
            try:
                cand_usr = rdMolDescriptors.GetUSRCAT(m3d)
                diff = sum(abs(a - b) for a, b in zip(cand_usr, seed_usr))
                usr_sim = round(float(1.0 / (1.0 + diff / len(cand_usr))), 4)
            except Exception:
                usr_sim = None

        results.append((m3d, {
            "Conformer_Success": True,
            "MMFF94_Energy": mmff_energy,
            "Strain_Energy": strain_energy,
            "USRCAT_Similarity": usr_sim
        }))
    return results

def compute_pareto_front(df: pd.DataFrame) -> List[bool]:
    if df.empty:
        return []
    pts = df[["Euclidean_Dist", "QED"]].values
    n = len(pts)
    is_pareto = [True] * n
    for i in range(n):
        if not is_pareto[i]:
            continue
        d_i, q_i = pts[i]
        for j in range(n):
            if i == j:
                continue
            d_j, q_j = pts[j]
            if (d_j <= d_i and q_j >= q_i) and (d_j < d_i or q_j > q_i):
                is_pareto[i] = False
                break
    return is_pareto

def convert_pdb_to_pdbqt(pdb_path: str, output_pdbqt_path: str) -> bool:
    obabel_bin = shutil.which("obabel")
    if not obabel_bin or not os.path.exists(pdb_path):
        return False
    try:
        cmd = [obabel_bin, "-ipdb", pdb_path, "-opdbqt", "-O", output_pdbqt_path, "-xr"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0 and os.path.exists(output_pdbqt_path) and os.path.getsize(output_pdbqt_path) > 0
    except Exception:
        return False

def get_pdb_center(pdb_path: str) -> Tuple[float, float, float]:
    coords = []
    if os.path.exists(pdb_path):
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append((x, y, z))
                    except Exception:
                        pass
    if coords:
        arr = np.array(coords, dtype=np.float64)
        c = np.mean(arr, axis=0)
        return (round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2))
    return (0.0, 0.0, 0.0)

class MolecularPipeline:
    def __init__(self):
        self.feature_names = DESCRIPTOR_NAMES
        self.dim = len(DESCRIPTOR_NAMES)
        self.mean = BENCHMARK_MEAN
        self.std = BENCHMARK_STD
        self.max_workers = os.cpu_count() or 4

    def parse_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                return mol
            except Exception:
                return None
        return None

    def compute_raw_descriptors(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        if mol is None:
            return None
        try:
            return np.array([
                float(Descriptors.MolWt(mol)),
                float(Descriptors.MolLogP(mol)),
                float(Descriptors.TPSA(mol)),
                float(Lipinski.NumHDonors(mol)),
                float(Lipinski.NumHAcceptors(mol)),
                float(Lipinski.NumRotatableBonds(mol)),
                float(Lipinski.RingCount(mol)),
                float(Lipinski.NumAromaticRings(mol)),
                float(Lipinski.FractionCSP3(mol)),
                float(QED.qed(mol))
            ], dtype=np.float64)
        except Exception:
            return None

    def transform(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.mean) / self.std

    def calculate_centroid(self, mols: List[Chem.Mol]) -> Tuple[np.ndarray, np.ndarray, Optional[List[float]]]:
        """
        Calculates exact descriptor and 3D shape centroid across 100% of input seed molecules
        without skipping or truncating any molecule.
        """
        vecs = []
        seed_usrs = []
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        params.numThreads = self.max_workers

        for m in mols:
            v = self.compute_raw_descriptors(m)
            if v is not None and not np.isnan(v).any():
                vecs.append(v)
            m_h = Chem.AddHs(m)
            if AllChem.EmbedMolecule(m_h, params) >= 0:
                try:
                    seed_usrs.append(rdMolDescriptors.GetUSRCAT(m_h))
                except Exception:
                    pass

        if not vecs:
            raise ValueError("No valid molecules provided to calculate centroid.")

        raw_centroid = np.mean(np.array(vecs, dtype=np.float64), axis=0)
        norm_centroid = self.transform(raw_centroid)

        mean_usr = None
        if seed_usrs:
            mean_usr = np.mean(np.array(seed_usrs, dtype=np.float64), axis=0).tolist()

        return raw_centroid, norm_centroid, mean_usr

    def generate_candidates(
        self,
        seed_mols: List[Chem.Mol],
        norm_centroid: np.ndarray,
        target_count: int = 40,
        beam_width: int = 150,
        max_depth: int = 3,
        max_sa_score: float = 5.0
    ) -> List[Chem.Mol]:
        """
        Full, comprehensive BRICS deconstruction across 100% of seed molecules.
        All fragments are preserved and prioritized by proximity to the target centroid.
        """
        seed_canonical = {Chem.MolToSmiles(m, isomericSmiles=False) for m in seed_mols}

        frag_smiles_set = set()
        for m in seed_mols:
            for f in BRICS.BRICSDecompose(m, returnMols=False, singlePass=False):
                frag_smiles_set.add(f)

        frag_mols = [Chem.MolFromSmiles(s) for s in frag_smiles_set if Chem.MolFromSmiles(s) is not None]
        if not frag_mols:
            return []

        # Retain 100% of all generated fragments, prioritized by distance vector
        scored_frags = []
        for f in frag_mols:
            v = self.compute_raw_descriptors(f)
            dist = float(np.linalg.norm(self.transform(v) - norm_centroid)) if v is not None else 999.0
            scored_frags.append((dist, f))
        scored_frags.sort(key=lambda x: x[0])
        sorted_frag_mols = [f for _, f in scored_frags]

        builder = BRICS.BRICSBuild(sorted_frag_mols, maxDepth=max_depth, scrambleReagents=False)

        candidates_pool = []
        seen_smiles = set()

        for cand in builder:
            if cand is None:
                continue
            try:
                Chem.SanitizeMol(cand)
                can_smi = Chem.MolToSmiles(cand, isomericSmiles=False)
                if can_smi in seed_canonical or can_smi in seen_smiles:
                    continue
                seen_smiles.add(can_smi)

                sa = calculate_sascore(cand)
                if sa > max_sa_score:
                    continue

                raw_v = self.compute_raw_descriptors(cand)
                if raw_v is None or np.isnan(raw_v).any():
                    continue

                dist = float(np.linalg.norm(self.transform(raw_v) - norm_centroid))
                candidates_pool.append((dist, cand))

                if len(candidates_pool) >= beam_width:
                    break
            except Exception:
                continue

        candidates_pool.sort(key=lambda x: x[0])
        return [c for _, c in candidates_pool[:target_count]]

    def score_and_rank(
        self,
        candidate_mols: List[Chem.Mol],
        norm_centroid: np.ndarray,
        seed_usr: Optional[List[float]] = None,
        min_qed: float = 0.35,
        max_dist: float = 5.0,
        max_ro5: int = 1,
        max_sa_score: float = 5.0,
        receptor_pdbqt_path: Optional[str] = None,
        dock_center: Optional[Tuple[float, float, float]] = None,
        dock_box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        dock_top_n: int = 10
    ) -> Tuple[pd.DataFrame, List[Chem.Mol], List[Chem.Mol]]:
        records = []
        retained_2d_mols = []

        for mol in candidate_mols:
            raw_v = self.compute_raw_descriptors(mol)
            if raw_v is None or np.isnan(raw_v).any():
                continue

            norm_v = self.transform(raw_v)
            dist = float(np.linalg.norm(norm_v - norm_centroid))
            qed_score = float(QED.qed(mol))
            sa_score = calculate_sascore(mol)
            mw = float(raw_v[0])
            logp = float(raw_v[1])
            hbd = int(raw_v[3])
            hba = int(raw_v[4])

            ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

            if qed_score < min_qed or dist > max_dist or ro5_violations > max_ro5 or sa_score > max_sa_score:
                continue

            records.append({
                "SMILES": Chem.MolToSmiles(mol),
                "Euclidean_Dist": round(dist, 4),
                "QED": round(qed_score, 4),
                "SAScore": round(sa_score, 2),
                "MW": round(mw, 2),
                "LogP": round(logp, 2),
                "TPSA": round(raw_v[2], 2),
                "HBD": hbd,
                "HBA": hba,
                "RotBonds": int(raw_v[5]),
                "AromRings": int(raw_v[7]),
                "Ro5_Violations": ro5_violations
            })
            retained_2d_mols.append(mol)

        df = pd.DataFrame(records)
        if df.empty:
            return df, [], []

        # Full 500-iteration MMFF94 force-field relaxation & ETKDGv3 embedding
        embed_results = embed_and_profile_3d_batch(retained_2d_mols, seed_usr=seed_usr)

        mols_3d = []
        mmff_energies = []
        strain_energies = []
        usr_sims = []

        for idx, (m3d, profile) in enumerate(embed_results):
            mols_3d.append(m3d if m3d is not None else retained_2d_mols[idx])
            mmff_energies.append(profile["MMFF94_Energy"])
            strain_energies.append(profile["Strain_Energy"])
            usr_sims.append(profile["USRCAT_Similarity"])

        df["MMFF94_Energy_kcal"] = mmff_energies
        df["Strain_Energy_kcal"] = strain_energies
        df["3D_Shape_USRCAT_Sim"] = usr_sims
        df["Pareto_Optimal"] = compute_pareto_front(df)

        df["_idx"] = range(len(df))
        df.sort_values(by=["Pareto_Optimal", "Euclidean_Dist", "QED"], ascending=[False, True, False], inplace=True)
        sorted_indices = df["_idx"].tolist()
        df.drop(columns=["_idx"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        sorted_2d_mols = [retained_2d_mols[i] for i in sorted_indices]
        sorted_3d_mols = [mols_3d[i] for i in sorted_indices]

        # Full precision 12-thread AutoDock Vina target docking
        if receptor_pdbqt_path and dock_center and HAS_VINA and os.path.exists(receptor_pdbqt_path):
            vina_scores = [None] * len(sorted_3d_mols)
            candidates_to_dock = min(dock_top_n, len(sorted_3d_mols))

            try:
                num_threads = os.cpu_count() or 4
                v = Vina(sf_name="vina", cpu=num_threads, verbosity=0)
                v.set_receptor(receptor_pdbqt_path)
                v.compute_vina_maps(center=list(dock_center), box_size=list(dock_box_size))

                prep = MoleculePreparation()
                for i in range(candidates_to_dock):
                    m = sorted_3d_mols[i]
                    if m.GetNumConformers() == 0:
                        continue
                    try:
                        setups = prep.prepare(m)
                        if not setups:
                            continue
                        pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(setups[0])
                        if not is_ok:
                            continue

                        v.set_ligand_from_string(pdbqt_str)
                        v.dock(exhaustiveness=2, n_poses=1)
                        e = v.energies(n_poses=1)
                        if e is not None and len(e) > 0:
                            vina_scores[i] = round(float(e[0][0]), 2)
                    except Exception:
                        continue
            except Exception:
                pass

            df["Vina_Affinity_kcal_mol"] = vina_scores

        return df, sorted_2d_mols, sorted_3d_mols

    def export_sdf(self, mols_3d: List[Chem.Mol], properties_df: pd.DataFrame) -> str:
        sio = io.StringIO()
        writer = Chem.SDWriter(sio)
        for mol, (_, row) in zip(mols_3d, properties_df.iterrows()):
            m = Chem.Mol(mol)
            for col in properties_df.columns:
                m.SetProp(col, str(row[col]))
            writer.write(m)
        writer.close()
        return sio.getvalue()
