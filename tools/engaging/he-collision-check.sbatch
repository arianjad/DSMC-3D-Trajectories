#!/usr/bin/env bash
#SBATCH --job-name=dsmc-he-collision-check
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=he-collision-%j.log
set -euo pipefail
root=$SLURM_SUBMIT_DIR
source "$root/smoke-setup.sh"
exec > >(tee "$root/he-collision-$SLURM_JOB_ID-progress.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$root/he-collision-$SLURM_JOB_ID.exit"' EXIT
hostname
sha256sum "$root/sparta-build/src/spa_mpi_bigbig"
# Same check sparta-smoke.sbatch runs as part of its acceptance verdict.
bash "$root/he-collision-runs.sh" "$root" "$root/he-collision-$SLURM_JOB_ID"
