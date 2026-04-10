from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Dict, Any
import csv
import re

from rdkit import Chem
from rdkit.Chem import AllChem

from meeko import MoleculePreparation, PDBQTWriterLegacy


def safe_name(name, fallback="molecule"):
    name = (name or "").strip()
    if not name:
        name = fallback
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name[:200]


def has_3d_conformer(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    try:
        return conf.Is3D()
    except Exception:
        zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
        return len(set(round(z, 3) for z in zs)) > 1


def copy_mol(mol):
    return Chem.Mol(mol)


def prepare_rdkit_mol(
    mol,
    add_hs=True,
    force_3d=True,
    minimize=True,
    max_embed_attempts=3,
    random_seed=42,
):
    if mol is None:
        raise ValueError("Input molecule is None")
    mol = copy_mol(mol)
    Chem.SanitizeMol(mol)

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    needs_3d = force_3d or (not has_3d_conformer(mol))

    if needs_3d:
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True

        embedded = False
        for _ in range(max_embed_attempts):
            mol_try = copy_mol(mol)
            code = AllChem.EmbedMolecule(mol_try, params)
            if code == 0:
                mol = mol_try
                embedded = True
                break
        if not embedded:
            raise RuntimeError("RDKit failed to generate a 3D conformer")

    if minimize:
        mmff_ok = AllChem.MMFFHasAllMoleculeParams(mol)
        if mmff_ok:
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    return mol


def mol_to_pdbqt_text(
    mol,
    mol_name=None,
    merge_hydrogens=True,
    keep_input_name_prop=True,
):
    mol = copy_mol(mol)

    if mol_name:
        mol.SetProp("_Name", mol_name)
    elif keep_input_name_prop and not mol.HasProp("_Name"):
        mol.SetProp("_Name", "molecule")

    merge_types = ("H",) if merge_hydrogens else ()
    preparator = MoleculePreparation(
        merge_these_atom_types=merge_types,
        rigid_macrocycles=False,
        min_ring_size=6,
        max_ring_size=33,
        double_bond_penalty=50,
    )
    setups = preparator.prepare(mol)

    if not setups:
        raise RuntimeError("Meeko returned no MoleculeSetup objects")

    pdbqt_blocks = []
    for setup in setups:
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        if not is_ok:
            raise RuntimeError(f"Meeko PDBQT writing failed: {error_msg}")
        pdbqt_blocks.append(pdbqt_string)

    return "\n".join(pdbqt_blocks)


def write_pdbqt_file(
    mol,
    output_path,
    mol_name=None,
    add_hs=True,
    force_3d=True,
    minimize=True,
    merge_hydrogens=True,
):
    prepared = prepare_rdkit_mol(
        mol,
        add_hs=add_hs,
        force_3d=force_3d,
        minimize=minimize,
    )
    pdbqt_text = mol_to_pdbqt_text(
        prepared,
        mol_name=mol_name,
        merge_hydrogens=merge_hydrogens,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pdbqt_text, encoding="utf-8")
    return output_path


def iter_sdf_molecules(sdf_path, remove_hs=False):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=remove_hs)
    for idx, mol in enumerate(supplier):
        yield idx, mol


def convert_sdf_file(
    sdf_path,
    output_dir,
    name_prop_candidates=("_Name", "ID", "Name", "TITLE"),
    add_hs=True,
    force_3d=False,
    minimize=True,
    merge_hydrogens=True,
    write_report_csv=True,
):
    sdf_path = Path(sdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    ok_count = 0
    fail_count = 0

    for idx, mol in iter_sdf_molecules(sdf_path, remove_hs=False):
        rec = {
            "index": idx,
            "source": str(sdf_path),
            "status": "ok",
            "name": None,
            "output": None,
            "error": None,
        }

        try:
            if mol is None:
                raise ValueError("RDKit could not parse record from SDF")

            mol_name = None
            for prop in name_prop_candidates:
                if mol.HasProp(prop):
                    val = mol.GetProp(prop).strip()
                    if val:
                        mol_name = val
                        break

            if not mol_name:
                mol_name = f"{sdf_path.stem}_{idx+1}"

            out_path = output_dir / f"{safe_name(mol_name)}.pdbqt"
            write_pdbqt_file(
                mol=mol,
                output_path=out_path,
                mol_name=mol_name,
                add_hs=add_hs,
                force_3d=force_3d,
                minimize=minimize,
                merge_hydrogens=merge_hydrogens,
            )
            rec["name"] = mol_name
            rec["output"] = str(out_path)
            ok_count += 1

        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = str(e)
            fail_count += 1

        results.append(rec)

    if write_report_csv:
        report_path = output_dir / f"{sdf_path.stem}_conversion_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "source", "status", "name", "output", "error"],
            )
            writer.writeheader()
            writer.writerows(results)

    return {
        "input": str(sdf_path),
        "output_dir": str(output_dir),
        "ok": ok_count,
        "failed": fail_count,
        "total": ok_count + fail_count,
        "results": results,
    }


def smiles_to_mol(smiles, name=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if name:
        mol.SetProp("_Name", name)
    return mol


def read_smiles_table(
    smiles_path,
    delimiter=None,
    smiles_col=0,
    name_col=1,
    has_header=False,
):
    smiles_path = Path(smiles_path)
    with smiles_path.open("r", encoding="utf-8") as f:
        if delimiter is None:
            delimiter = "\t" if smiles_path.suffix.lower() in {".smi", ".tsv"} else ","
        reader = csv.reader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            if not row:
                continue
            if has_header and idx == 0:
                continue
            smiles = row[smiles_col].strip()
            name = (
                row[name_col].strip()
                if (name_col is not None and len(row) > name_col)
                else f"mol_{idx+1}"
            )
            yield idx, smiles, name


def convert_smiles_file(
    smiles_path,
    output_dir,
    delimiter=None,
    smiles_col=0,
    name_col=1,
    has_header=False,
    add_hs=True,
    force_3d=True,
    minimize=True,
    merge_hydrogens=True,
    write_report_csv=True,
):
    smiles_path = Path(smiles_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    ok_count = 0
    fail_count = 0

    for idx, smiles, name in read_smiles_table(
        smiles_path,
        delimiter=delimiter,
        smiles_col=smiles_col,
        name_col=name_col,
        has_header=has_header,
    ):
        rec = {
            "index": idx,
            "source": str(smiles_path),
            "status": "ok",
            "name": name,
            "smiles": smiles,
            "output": None,
            "error": None,
        }

        try:
            mol = smiles_to_mol(smiles, name=name)
            out_path = output_dir / f"{safe_name(name)}.pdbqt"
            write_pdbqt_file(
                mol=mol,
                output_path=out_path,
                mol_name=name,
                add_hs=add_hs,
                force_3d=force_3d,
                minimize=minimize,
                merge_hydrogens=merge_hydrogens,
            )
            rec["output"] = str(out_path)
            ok_count += 1

        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = str(e)
            fail_count += 1

        results.append(rec)

    if write_report_csv:
        report_path = output_dir / f"{smiles_path.stem}_conversion_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "source",
                    "status",
                    "name",
                    "smiles",
                    "output",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

    return {
        "input": str(smiles_path),
        "output_dir": str(output_dir),
        "ok": ok_count,
        "failed": fail_count,
        "total": ok_count + fail_count,
        "results": results,
    }


if __name__ == "__main__":
    # конвертация SDF -> PDBQT
    convert_sdf_file(
        r"C:\Users\zu200\example.sdf",
        r"C:\Users\zu200\pdbqt_from_sdf",
        add_hs=True,
        force_3d=False,
        minimize=True,
    )

    # конвертация SMILES -> PDBQT
    convert_smiles_file(
        r"C:\Users\zu200\example.smi",
        r"C:\Users\zu200\pdbqt_from_smiles",
        add_hs=True,
        force_3d=True,
        minimize=True,
    )
