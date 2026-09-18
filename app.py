import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import py3Dmol
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from pipeline import MolecularPipeline, convert_pdb_to_pdbqt, get_pdb_center, HAS_VINA

RDLogger.DisableLog("rdApp.*")

st.set_page_config(
    page_title="VECTOR - Molecular Trajectory & Recombination Platform",
    layout="wide"
)

EXAMPLE_SMILES = (
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O\n"
    "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O\n"
    "CC(C1=CC=CC(=C1)C(=O)C2=CC=CC=C2)C(=O)O\n"
    "CC(C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O\n"
    "O=C(O)CCC1=NC(=C(O1)C2=CC=CC=C2)C3=CC=CC=C3"
)

LOCAL_PROTEIN_PDB = os.path.join(os.path.dirname(__file__), "protein.pdb")

def render_3dmol(molblock: str, style: str = "stick", width: int = 500, height: int = 400):
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(molblock, "mol")
    if style == "stick":
        viewer.setStyle({"stick": {"colorscheme": "greenCarbon"}})
    elif style == "sphere":
        viewer.setStyle({"sphere": {"scale": 0.3}, "stick": {}})
    else:
        viewer.setStyle({"line": {}})
    viewer.zoomTo()
    viewer.setBackgroundColor("white")
    html = viewer._make_html()
    components.html(html, height=height + 20)

@st.cache_resource
def load_pipeline():
    return MolecularPipeline()

pipeline = load_pipeline()

st.title("VECTOR Platform")
st.markdown("**V**ector-Guided **E**uclidean **C**hemical **T**rajectory & **O**rganic **R**ecombination")

with st.sidebar:
    st.header("Generation & Filter Parameters")
    max_dist = st.slider("Max Euclidean Descriptor Distance:", min_value=1.0, max_value=8.0, value=4.5, step=0.1)
    min_qed = st.slider("Minimum QED (Drug-likeness):", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    max_sa = st.slider("Max Synthetic Accessibility (SAScore):", min_value=2.0, max_value=8.0, value=5.0, step=0.5)
    max_ro5 = st.selectbox("Lipinski Rule of 5 Max Violations:", [0, 1, 2], index=1)
    target_count = st.slider("Candidate Molecules to Generate:", min_value=10, max_value=120, value=40, step=5)

    st.divider()
    st.header("Optional Protein Target & Docking Grid")
    enable_docking = st.checkbox("Enable Target Protein Pocket Docking (AutoDock Vina)", value=False)
    
    receptor_file = None
    dock_center = (0.0, 0.0, 0.0)
    dock_box_size = (20.0, 20.0, 20.0)

    if enable_docking:
        if os.path.exists(LOCAL_PROTEIN_PDB):
            if st.button("Load Example Protein (protein.pdb)", width="stretch"):
                st.session_state["use_example_protein"] = True
                st.session_state["protein_center"] = get_pdb_center(LOCAL_PROTEIN_PDB)
                st.rerun()

        if st.session_state.get("use_example_protein", False):
            st.success("Loaded local example: protein.pdb")
            if st.button("Unload Example Protein", width="stretch"):
                st.session_state["use_example_protein"] = False
                st.rerun()

        receptor_file = st.file_uploader("Upload Custom Target Receptor (.pdb or .pdbqt):", type=["pdb", "pdbqt"])

        default_center = st.session_state.get("protein_center", (0.0, 0.0, 0.0))
        st.caption("Docking Grid Center Coordinates (X, Y, Z):")
        c1, c2, c3 = st.columns(3)
        center_x = c1.number_input("Center X", value=float(default_center[0]), step=1.0)
        center_y = c2.number_input("Center Y", value=float(default_center[1]), step=1.0)
        center_z = c3.number_input("Center Z", value=float(default_center[2]), step=1.0)
        dock_center = (center_x, center_y, center_z)

        st.caption("Grid Box Dimensions (Angstroms):")
        b1, b2, b3 = st.columns(3)
        size_x = b1.number_input("Size X", value=20.0, step=2.0)
        size_y = b2.number_input("Size Y", value=20.0, step=2.0)
        size_z = b3.number_input("Size Z", value=20.0, step=2.0)
        dock_box_size = (size_x, size_y, size_z)

st.subheader("1. Input Seed Molecules")
st.caption("Paste canonical SMILES (one per line) or upload a .txt file. VECTOR is designed for small molecules (MW < 700 g/mol, heavy atoms < 50) and is not intended for macrocycles or peptides.")

if "input_smiles" not in st.session_state:
    st.session_state["input_smiles"] = ""

col_action1, col_action2, _ = st.columns([1, 1, 3])
with col_action1:
    if st.button("Load Example Seed Molecules", width="stretch"):
        st.session_state["input_smiles"] = EXAMPLE_SMILES
        st.rerun()
with col_action2:
    if st.button("Clear Input", width="stretch"):
        st.session_state["input_smiles"] = ""
        st.rerun()

txt_file = st.file_uploader("Upload Seed SMILES (.txt file):", type=["txt"])
if txt_file is not None:
    try:
        content = txt_file.read().decode("utf-8")
        st.session_state["input_smiles"] = content
        st.info("Loaded seed SMILES from uploaded .txt file.")
    except Exception as e:
        st.error(f"Failed to decode text file: {e}")

input_smiles_raw = st.text_area(
    "Seed SMILES:",
    value=st.session_state["input_smiles"],
    height=140,
    placeholder="Paste at least 3 SMILES strings here (or click 'Load Example Seed Molecules' or upload a .txt file)..."
)

st.session_state["input_smiles"] = input_smiles_raw

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_btn = st.button("Generate Novel Molecules", type="primary", width="stretch")

if run_btn:
    lines = [s.strip() for s in input_smiles_raw.splitlines() if s.strip()]
    parsed_seeds = [pipeline.parse_smiles(s) for s in lines]
    valid_seeds = [m for m in parsed_seeds if m is not None]

    if len(valid_seeds) < 3:
        st.error(f"Insufficient seed molecules: provided {len(valid_seeds)}, but at least 3 are required to guarantee retro-synthetic fragment diversity and prevent combinatorial collapse.")
        st.stop()

    if len(valid_seeds) < 6:
        st.info(f"Loaded {len(valid_seeds)} seeds: Operating in Focused Analogue Exploration mode. For broader scaffold diversity, providing 6 to 10+ seeds is recommended.")
    else:
        st.success(f"Loaded {len(valid_seeds)} seeds: Operating in Optimal Multidimensional Diversity mode across {pipeline.max_workers} CPU threads.")

    raw_centroid, norm_centroid, seed_usr = pipeline.calculate_centroid(valid_seeds)

    with st.expander("Multidimensional Descriptor Centroid Profile", expanded=True):
        stat_cols = st.columns(5)
        stat_cols[0].metric("MW (g/mol)", f"{raw_centroid[0]:.1f}")
        stat_cols[1].metric("LogP", f"{raw_centroid[1]:.2f}")
        stat_cols[2].metric("TPSA", f"{raw_centroid[2]:.1f}")
        stat_cols[3].metric("HBD / HBA", f"{int(round(raw_centroid[3]))} / {int(round(raw_centroid[4]))}")
        stat_cols[4].metric("QED Mean", f"{raw_centroid[9]:.2f}")

    receptor_pdbqt_path = None
    temp_pdbqt_cleanup = None

    if enable_docking:
        if st.session_state.get("use_example_protein", False) and os.path.exists(LOCAL_PROTEIN_PDB):
            temp_pdbqt = tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt")
            temp_pdbqt.close()
            success = convert_pdb_to_pdbqt(LOCAL_PROTEIN_PDB, temp_pdbqt.name)
            if success:
                receptor_pdbqt_path = temp_pdbqt.name
                temp_pdbqt_cleanup = temp_pdbqt.name
            else:
                st.warning("Could not auto-convert protein.pdb to PDBQT. Skipping docking calculation.")
        elif receptor_file is not None:
            fname = receptor_file.name.lower()
            if fname.endswith(".pdbqt"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt")
                tfile.write(receptor_file.read())
                tfile.close()
                receptor_pdbqt_path = tfile.name
                temp_pdbqt_cleanup = tfile.name
            elif fname.endswith(".pdb"):
                tpdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
                tpdb.write(receptor_file.read())
                tpdb.close()
                tpdbqt = tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt")
                tpdbqt.close()
                if convert_pdb_to_pdbqt(tpdb.name, tpdbqt.name):
                    receptor_pdbqt_path = tpdbqt.name
                    temp_pdbqt_cleanup = tpdbqt.name
                else:
                    st.warning("Could not convert uploaded PDB to PDBQT. Skipping docking.")
                try:
                    os.remove(tpdb.name)
                except Exception:
                    pass
        else:
            st.warning("Target docking was enabled, but no receptor file was loaded. Proceeding with descriptor & 3D biophysical profiling.")

    with st.spinner("Executing priority-guided chemical sampling, SAScore pre-filtering, 3D conformer embedding, and Pareto optimization..."):
        novel_pool = pipeline.generate_candidates(
            valid_seeds,
            norm_centroid=norm_centroid,
            target_count=target_count,
            max_sa_score=max_sa
        )
        df_ranked, retained_2d, retained_3d = pipeline.score_and_rank(
            novel_pool,
            norm_centroid=norm_centroid,
            seed_usr=seed_usr,
            min_qed=min_qed,
            max_dist=max_dist,
            max_ro5=max_ro5,
            max_sa_score=max_sa,
            receptor_pdbqt_path=receptor_pdbqt_path,
            dock_center=dock_center,
            dock_box_size=dock_box_size
        )

    if temp_pdbqt_cleanup and os.path.exists(temp_pdbqt_cleanup):
        try:
            os.remove(temp_pdbqt_cleanup)
        except Exception:
            pass

    if df_ranked.empty:
        st.warning("No candidate structures satisfied the distance, SAScore, or drug-likeness criteria. Consider relaxing Max Distance, SAScore, or QED threshold.")
    else:
        st.session_state["results_df"] = df_ranked
        st.session_state["results_2d"] = retained_2d
        st.session_state["results_3d"] = retained_3d

if "results_df" in st.session_state and not st.session_state["results_df"].empty:
    df_ranked = st.session_state["results_df"]
    retained_2d = st.session_state["results_2d"]
    retained_3d = st.session_state["results_3d"]

    st.subheader(f"2. Novel Candidates Ranked by Pareto Optimality & Descriptor Proximity ({len(df_ranked)} Compounds)")
    
    display_cols = [
        "SMILES", "Pareto_Optimal", "Euclidean_Dist", "QED", "SAScore", "MW", "LogP", "TPSA",
        "MMFF94_Energy_kcal", "Strain_Energy_kcal", "3D_Shape_USRCAT_Sim", "Ro5_Violations"
    ]
    if "Vina_Affinity_kcal_mol" in df_ranked.columns:
        display_cols.insert(2, "Vina_Affinity_kcal_mol")
    
    st.dataframe(
        df_ranked[display_cols].style.highlight_min(subset=["Euclidean_Dist"], color="#d4edda")
                                     .highlight_max(subset=["QED"], color="#d1ecf1"),
        width="stretch"
    )

    st.subheader("3. Molecular Visualization: Side-by-Side 2D & Interactive 3D Shapes")
    
    cand_options = [
        f"Rank {i+1}: SMILES={df_ranked.loc[i, 'SMILES'][:28]}... | Dist={df_ranked.loc[i, 'Euclidean_Dist']} | QED={df_ranked.loc[i, 'QED']}"
        for i in range(len(df_ranked))
    ]
    selected_cand_idx = st.selectbox("Select Candidate to Inspect (2D & 3D):", range(len(cand_options)), format_func=lambda i: cand_options[i])

    col_2d, col_3d = st.columns([1, 1])
    
    with col_2d:
        st.markdown("**2D Structure Diagram**")
        mol_2d = retained_2d[selected_cand_idx]
        img = Draw.MolToImage(mol_2d, size=(450, 380))
        st.image(img, caption=f"2D Graph: {df_ranked.loc[selected_cand_idx, 'SMILES']}", width="stretch")

    with col_3d:
        st.markdown("**Interactive 3D Conformation (Rotate, Pan & Zoom)**")
        style_choice = st.radio("3D Display Style:", ["Stick", "Sphere/Spacefill", "Line"], horizontal=True)
        mol_3d = retained_3d[selected_cand_idx]
        if mol_3d.GetNumConformers() > 0:
            mb = Chem.MolToMolBlock(mol_3d)
            style_map = {"Stick": "stick", "Sphere/Spacefill": "sphere", "Line": "line"}
            render_3dmol(mb, style=style_map[style_choice], width=450, height=360)
            st.caption(f"MMFF94 Energy: {df_ranked.loc[selected_cand_idx, 'MMFF94_Energy_kcal']} kcal/mol | USRCAT 3D Shape Sim: {df_ranked.loc[selected_cand_idx, '3D_Shape_USRCAT_Sim']}")
        else:
            st.info("3D conformation not available for this candidate.")

    st.subheader("4. Top Candidate Overview Grid (2D)")
    top_n = min(6, len(retained_2d))
    grid_img = Draw.MolsToGridImage(
        retained_2d[:top_n],
        molsPerRow=3,
        subImgSize=(350, 250),
        legends=[
            f"P:{df_ranked.loc[i, 'Pareto_Optimal']} | D:{df_ranked.loc[i, 'Euclidean_Dist']} | Q:{df_ranked.loc[i, 'QED']} | SA:{df_ranked.loc[i, 'SAScore']}"
            for i in range(top_n)
        ]
    )
    st.image(grid_img, caption=f"Top {top_n} Novel Structures (Pareto Frontier & Proximity)", width="stretch")

    st.subheader("5. Export Discovery Results")
    csv_bytes = df_ranked.to_csv(index=False).encode('utf-8')
    sdf_data = pipeline.export_sdf(retained_3d, df_ranked)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download Results as CSV Table",
            data=csv_bytes,
            file_name="vector_molecular_candidates.csv",
            mime="text/csv",
            width="stretch"
        )
    with dl_col2:
        st.download_button(
            label="Download 3D Minimized Structures as SDF (Ready for Docking / CAD)",
            data=sdf_data,
            file_name="vector_molecular_candidates_3d.sdf",
            mime="chemical/x-mdl-sdfile",
            width="stretch"
        )
