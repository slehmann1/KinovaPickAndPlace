#!/usr/bin/env bash
set -euo pipefail

# ACRONYM setup helper for Linux / WSL.
#
# This script:
# 1. clones the official ACRONYM repo into the project
# 2. extracts a locally downloaded ACRONYM grasp archive
# 3. locates the referenced source meshes under a ShapeNetSem-style mesh root
# 4. builds watertight, simplified OBJ meshes into data/acronym/meshes
#
# Assumptions:
# - git, tar, python3, manifold, and simplify are installed and available on PATH
# - you already downloaded the official ACRONYM grasp archive separately
# - you already downloaded the corresponding ShapeNetSem meshes separately
#
# Example:
#   bash scripts/setup_acronym.sh \
#     --grasp-archive /path/to/acronym.tar.gz \
#     --mesh-source /path/to/ShapeNetSem

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACRONYM_REPO_URL="https://github.com/NVlabs/acronym.git"
ACRONYM_REPO_DIR="${PROJECT_ROOT}/data/acronym_repo"
ACRONYM_DATA_DIR="${PROJECT_ROOT}/data/acronym"
ACRONYM_GRASP_DIR="${ACRONYM_DATA_DIR}/grasps"
ACRONYM_MESH_DIR="${ACRONYM_DATA_DIR}/meshes"
TMP_DIR="${ACRONYM_DATA_DIR}/_tmp"

GRASP_ARCHIVE=""
MESH_SOURCE=""
FORCE_REPO=0
FORCE_GRASPS=0
FORCE_MESHES=0
KEEP_TMP=0
MAX_MESHES=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_acronym.sh --grasp-archive <acronym.tar.gz> --mesh-source <ShapeNetSem_root> [options]

Required arguments:
  --grasp-archive PATH   Local path to the downloaded ACRONYM grasp archive (.tar.gz)
  --mesh-source PATH     Root directory containing the original ShapeNetSem OBJ meshes

Optional arguments:
  --project-root PATH    Override project root. Defaults to the parent of this script.
  --force-repo           Re-clone the ACRONYM repo even if it already exists
  --force-grasps         Re-extract the grasp archive even if data/acronym/grasps exists
  --force-meshes         Rebuild processed meshes even if outputs already exist
  --keep-tmp             Keep temporary extraction files under data/acronym/_tmp
  --max-meshes N         Limit mesh processing to the first N grasp files for smoke tests
  --help                 Show this help message

Notes:
  - Full ACRONYM installation still requires you to obtain the grasp archive and
    ShapeNetSem meshes yourself from the official sources.
  - 'manifold' and 'simplify' are expected to be installed separately.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

resolve_abs_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grasp-archive)
      GRASP_ARCHIVE="$2"
      shift 2
      ;;
    --mesh-source)
      MESH_SOURCE="$2"
      shift 2
      ;;
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --force-repo)
      FORCE_REPO=1
      shift
      ;;
    --force-grasps)
      FORCE_GRASPS=1
      shift
      ;;
    --force-meshes)
      FORCE_MESHES=1
      shift
      ;;
    --keep-tmp)
      KEEP_TMP=1
      shift
      ;;
    --max-meshes)
      MAX_MESHES="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${GRASP_ARCHIVE}" ]] || die "--grasp-archive is required"
[[ -n "${MESH_SOURCE}" ]] || die "--mesh-source is required"

PROJECT_ROOT="$(resolve_abs_path "${PROJECT_ROOT}")"
GRASP_ARCHIVE="$(resolve_abs_path "${GRASP_ARCHIVE}")"
MESH_SOURCE="$(resolve_abs_path "${MESH_SOURCE}")"

ACRONYM_REPO_DIR="${PROJECT_ROOT}/data/acronym_repo"
ACRONYM_DATA_DIR="${PROJECT_ROOT}/data/acronym"
ACRONYM_GRASP_DIR="${ACRONYM_DATA_DIR}/grasps"
ACRONYM_MESH_DIR="${ACRONYM_DATA_DIR}/meshes"
TMP_DIR="${ACRONYM_DATA_DIR}/_tmp"

need_cmd git
need_cmd tar
need_cmd python3
need_cmd manifold
need_cmd simplify

[[ -f "${GRASP_ARCHIVE}" ]] || die "Grasp archive not found: ${GRASP_ARCHIVE}"
[[ -d "${MESH_SOURCE}" ]] || die "Mesh source directory not found: ${MESH_SOURCE}"

mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${ACRONYM_DATA_DIR}"
mkdir -p "${TMP_DIR}"

if [[ ${FORCE_REPO} -eq 1 && -d "${ACRONYM_REPO_DIR}" ]]; then
  rm -rf "${ACRONYM_REPO_DIR}"
fi

if [[ ! -d "${ACRONYM_REPO_DIR}/.git" ]]; then
  echo "Cloning ACRONYM repo into ${ACRONYM_REPO_DIR}"
  git clone "${ACRONYM_REPO_URL}" "${ACRONYM_REPO_DIR}"
else
  echo "Using existing ACRONYM repo at ${ACRONYM_REPO_DIR}"
fi

if [[ ${FORCE_GRASPS} -eq 1 && -d "${ACRONYM_GRASP_DIR}" ]]; then
  rm -rf "${ACRONYM_GRASP_DIR}"
fi

if [[ ! -d "${ACRONYM_GRASP_DIR}" ]]; then
  echo "Extracting grasp archive into ${ACRONYM_DATA_DIR}"
  EXTRACT_DIR="${TMP_DIR}/grasp_extract"
  rm -rf "${EXTRACT_DIR}"
  mkdir -p "${EXTRACT_DIR}"
  tar -xzf "${GRASP_ARCHIVE}" -C "${EXTRACT_DIR}"

  mkdir -p "${ACRONYM_GRASP_DIR}"
  mapfile -t H5_FILES < <(find "${EXTRACT_DIR}" -type f -name '*.h5' | sort)
  [[ ${#H5_FILES[@]} -gt 0 ]] || die "No .h5 grasp files found in extracted archive"

  for grasp_file in "${H5_FILES[@]}"; do
    cp "${grasp_file}" "${ACRONYM_GRASP_DIR}/"
  done
else
  echo "Using existing grasp directory at ${ACRONYM_GRASP_DIR}"
fi

echo "Preparing watertight mesh outputs under ${ACRONYM_MESH_DIR}"
mkdir -p "${ACRONYM_MESH_DIR}"

python3 - "${ACRONYM_GRASP_DIR}" "${MESH_SOURCE}" "${ACRONYM_MESH_DIR}" "${TMP_DIR}" "${FORCE_MESHES}" "${MAX_MESHES}" <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np


def decode_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_string(value.item())
    if isinstance(value, np.ndarray) and value.size == 1:
        return decode_string(value.reshape(-1)[0])
    return str(value)


def load_optional_first_available(container, candidate_paths, default=None):
    for candidate in candidate_paths:
        try:
            return container[candidate][()]
        except Exception:
            continue
    return default


def resolve_mesh_candidates(mesh_root: Path, object_id: str, mesh_rel_path: str | None):
    candidates = []

    if mesh_rel_path:
        rel = Path(mesh_rel_path.replace("\\", "/").strip())
        basename = rel.name
        stem = rel.stem
        candidates.extend(
            [
                mesh_root / rel,
                mesh_root / basename,
                mesh_root / stem / "model.obj",
                mesh_root / stem / f"{stem}.obj",
                mesh_root / stem / "model_watertight.obj",
                mesh_root / f"{stem}.obj",
                mesh_root / f"{stem}_watertight.obj",
            ]
        )

    stem = object_id
    if "_" in object_id:
        maybe_scale = object_id.rsplit("_", 1)[-1]
        try:
            float(maybe_scale)
            stem = object_id.rsplit("_", 1)[0]
        except ValueError:
            stem = object_id

    candidates.extend(
        [
            mesh_root / f"{stem}.obj",
            mesh_root / stem / "model.obj",
            mesh_root / stem / f"{stem}.obj",
            mesh_root / stem / "model_watertight.obj",
            mesh_root / f"{stem}_watertight.obj",
        ]
    )
    return candidates


def resolve_mesh(mesh_root: Path, object_id: str, mesh_rel_path: str | None):
    checked = []
    for candidate in resolve_mesh_candidates(mesh_root, object_id, mesh_rel_path):
        checked.append(candidate)
        if candidate.exists():
            return candidate.resolve()

    mesh_names = []
    if mesh_rel_path:
        mesh_names.append(Path(mesh_rel_path).name)
    mesh_names.append(f"{object_id}.obj")
    if "_" in object_id:
        mesh_names.append(f"{object_id.rsplit('_', 1)[0]}.obj")

    for mesh_name in mesh_names:
        matches = list(mesh_root.rglob(mesh_name))
        if matches:
            return matches[0].resolve()

    checked_str = "\n".join(str(path) for path in checked[:20])
    raise FileNotFoundError(
        "Unable to resolve mesh for ACRONYM object\n"
        f"object_id={object_id}\n"
        f"mesh_rel_path={mesh_rel_path}\n"
        f"First checked candidates:\n{checked_str}"
    )


grasp_dir = Path(sys.argv[1]).resolve()
mesh_root = Path(sys.argv[2]).resolve()
mesh_out_root = Path(sys.argv[3]).resolve()
tmp_dir = Path(sys.argv[4]).resolve()
force_meshes = bool(int(sys.argv[5]))
max_meshes = int(sys.argv[6])

mesh_out_root.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

mesh_paths = (
    "object/file",
    "object/path",
    "object/mesh/file",
    "mesh/file",
    "mesh/path",
    "object_file",
)
scale_paths = (
    "object/scale",
    "mesh/scale",
    "object_scale",
    "scale",
)

grasp_files = sorted(grasp_dir.glob("*.h5"))
if max_meshes > 0:
    grasp_files = grasp_files[:max_meshes]

if not grasp_files:
    raise FileNotFoundError(f"No ACRONYM grasp files found under {grasp_dir}")

processed = 0
skipped = 0

for grasp_file in grasp_files:
    object_id = grasp_file.stem
    with h5py.File(grasp_file, "r") as handle:
        mesh_rel_path = load_optional_first_available(handle, mesh_paths)
        object_scale = load_optional_first_available(handle, scale_paths, default=1.0)

    mesh_rel_path = None if mesh_rel_path is None else decode_string(mesh_rel_path)
    object_scale = float(np.asarray(object_scale).reshape(-1)[0])
    source_mesh = resolve_mesh(mesh_root, object_id, mesh_rel_path)

    object_dir_name = object_id
    output_dir = mesh_out_root / object_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    final_mesh = output_dir / "model.obj"

    if final_mesh.exists() and not force_meshes:
      skipped += 1
      continue

    tmp_mesh = tmp_dir / f"{object_dir_name}_watertight.obj"
    if tmp_mesh.exists():
        tmp_mesh.unlink()

    print(f"[mesh] processing {object_id}")
    print(f"       source: {source_mesh}")
    print(f"       scale:  {object_scale}")
    print(f"       output: {final_mesh}")

    subprocess.run(
        ["manifold", str(source_mesh), str(tmp_mesh), "-s"],
        check=True,
    )
    subprocess.run(
        ["simplify", "-i", str(tmp_mesh), "-o", str(final_mesh), "-m", "-r", "0.02"],
        check=True,
    )

    metadata = {
        "object_id": object_id,
        "source_mesh": str(source_mesh),
        "mesh_rel_path": mesh_rel_path,
        "object_scale": object_scale,
        "processed_mesh": str(final_mesh),
    }
    (output_dir / "metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2),
        encoding="utf-8",
    )
    processed += 1

print(f"Processed meshes: {processed}")
print(f"Skipped existing meshes: {skipped}")
PY

if [[ ${KEEP_TMP} -eq 0 ]]; then
  rm -rf "${TMP_DIR}"
fi

echo
echo "ACRONYM setup complete."
echo "Repo:   ${ACRONYM_REPO_DIR}"
echo "Grasps: ${ACRONYM_GRASP_DIR}"
echo "Meshes: ${ACRONYM_MESH_DIR}"
echo
echo "Next check:"
echo "  python .\\src\\grasp_pipeline\\acronym_tools.py --root-dir .\\data\\acronym --mesh-root .\\data\\acronym\\meshes --object-index 0"
