#!/usr/bin/env python3
"""Prepare, inspect and bind saved frames for versioned 3D DSMC cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dsmc_case import inspect_geometry, load_case, load_profile, prepare_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="resolve a case into a fresh directory")
    prepare.add_argument("case", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    cad_prepare = commands.add_parser("cad-workflow-prepare", help="bind checked CAD geometry and explicit transport inputs")
    cad_prepare.add_argument("checked", type=Path)
    cad_prepare.add_argument("--helium-config", type=Path, required=True)
    cad_prepare.add_argument("--transport-config", type=Path, required=True)
    cad_prepare.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="check prepared file identities and summarize geometry")
    inspect.add_argument("prepared", type=Path)
    inspect.add_argument("--json", type=Path, help="new JSON inspection result path")
    inspect.add_argument("--solver-check", action="store_true",
                         help="run an explicit SPARTA run-0 check and freeze a new canonical preparation")
    inspect.add_argument("--profile", type=Path)
    inspect.add_argument("--out", type=Path, help="fresh output directory for the solver-checked derivative")
    frame = commands.add_parser("frame", help="select an independently complete saved helium frame")
    frame.add_argument("run_dir", type=Path)
    choice = frame.add_mutually_exclusive_group(required=True)
    choice.add_argument("--timestep", type=int)
    choice.add_argument("--last-complete", action="store_true")
    frame.add_argument("--json", type=Path, help="new JSON frame-selection receipt path")
    field = commands.add_parser("field", help="freeze and convert one selected helium frame")
    field.add_argument("prepared", type=Path, help="solver-checked preparation")
    field.add_argument("--parent", type=Path, required=True, help="helium run directory")
    field_choice = field.add_mutually_exclusive_group(required=True)
    field_choice.add_argument("--timestep", type=int)
    field_choice.add_argument("--last-complete", action="store_true")
    field.add_argument("--gas-policy", required=True,
                       choices=("positive-T-only", "positive-T-and-empty-ballistic",
                                "carry-T", "carry-T-and-empty-ballistic"))
    field.add_argument("--splitmap", type=Path,
                       help="solver-exported split3d map for exact cut-cell lookup")
    field.add_argument("--splitmap-ids", type=Path,
                       help="receipt-bound split parent ID list for --splitmap")
    field.add_argument("--out", type=Path, required=True, help="new child directory")
    molecule = commands.add_parser("molecule", help="trace molecules through one prepared field")
    molecule.add_argument("field_dir", type=Path)
    molecule.add_argument("--out", type=Path, required=True, help="new child directory")
    molecule.add_argument("--count", type=int, help="molecules to launch (smoke override)")
    molecule.add_argument("--trace-count", type=int, default=0, metavar="K",
                          help="record full paths of molecules 1..K to paths.csv (default 0: off)")
    molecule_batch = commands.add_parser(
        "molecule-batch", help="run named molecule trials on one checked field")
    molecule_batch.add_argument("field_dir", type=Path)
    molecule_batch.add_argument("--trials", type=Path, required=True,
                                help="JSON with named source/species trial overrides")
    molecule_batch.add_argument("--out", type=Path, required=True,
                                help="new batch directory")
    run = commands.add_parser("run", help="run checked local helium, field and molecules")
    run.add_argument("prepared", type=Path, help="solver-checked preparation")
    run.add_argument("--profile", type=Path, required=True, help="local machine profile")
    run.add_argument("--gas-policy", required=True,
                     choices=("positive-T-only", "positive-T-and-empty-ballistic",
                              "carry-T", "carry-T-and-empty-ballistic"))
    run.add_argument("--helium-backend", choices=("native", "wsl"), default="native",
                     help="launch helium on this host or through WSL (Windows)")
    run.add_argument("--out", type=Path, required=True, help="new workflow directory")
    run.add_argument("--splitmap", type=Path)
    run.add_argument("--splitmap-ids", type=Path)
    run.add_argument("--splitmap-solver", help="separate native diagnostic executable; generate map before field conversion")
    finish = commands.add_parser("finish", help="run downstream stages from a retained complete helium receipt")
    finish.add_argument("prepared", type=Path)
    finish.add_argument("--helium-receipt", type=Path, required=True)
    finish.add_argument("--profile", type=Path, required=True)
    finish.add_argument("--gas-policy", required=True)
    finish.add_argument("--out", type=Path, required=True)
    finish.add_argument("--splitmap", type=Path)
    finish.add_argument("--splitmap-ids", type=Path)
    finish.add_argument("--splitmap-solver")
    finish.add_argument("--splitmap-probe", type=Path,
                        help="retain and revalidate an existing complete diagnostic directory without rerunning SPARTA")
    finish.add_argument("--splitmap-backend", choices=("native", "wsl"), default="native")
    cad_import = commands.add_parser("bind-cad-helium", help="import a completed CAD pilot for downstream reuse")
    cad_import.add_argument("prepared", type=Path)
    cad_import.add_argument("--pilot-result", type=Path, required=True)
    cad_import.add_argument("--out", type=Path, required=True)
    normalize = commands.add_parser("normalize-resume", help="bind verified resumed helium for downstream stages")
    normalize.add_argument("resume_dir", type=Path)
    normalize.add_argument("--parent", type=Path, required=True)
    acceptance = commands.add_parser("acceptance", help="run local/handoff acceptance or render/submit one Slurm workflow")
    acceptance.add_argument("--suite", required=True, choices=("local", "handoff", "slurm"))
    acceptance.add_argument("--profile", type=Path, required=True)
    acceptance.add_argument("--helium-backend", choices=("auto", "native", "wsl"), default="auto")
    acceptance.add_argument("--prepared", type=Path, help="checked B5 input for Slurm dry run")
    acceptance.add_argument("--dry-run", action="store_true", help="render only; never submit")
    acceptance.add_argument("--submit", action="store_true", help="render and submit once; reconciliation is on demand")
    acceptance.add_argument("--out", type=Path, required=True, help="new acceptance directory")
    slurm_submit = commands.add_parser("slurm-submit", help="submit one rendered Slurm bundle exactly once")
    slurm_submit.add_argument("bundle", type=Path)
    slurm_submit.add_argument("--receipt", type=Path, required=True)
    slurm_status = commands.add_parser("slurm-status", help="query and reconcile one Slurm attempt receipt")
    slurm_status.add_argument("--receipt", type=Path, required=True)
    slurm_benchmark = commands.add_parser("slurm-benchmark", help="collect cost evidence from one submitted job and verified archive; no simulation")
    slurm_benchmark.add_argument("submission", type=Path)
    slurm_benchmark.add_argument("--archive", type=Path, required=True)
    slurm_benchmark.add_argument("--out", type=Path, required=True)
    slurm_render = commands.add_parser("slurm-render", help="render a full workflow, checkpoint producer, or recovery job")
    slurm_render.add_argument("prepared", type=Path)
    slurm_render.add_argument("--profile", type=Path, required=True)
    slurm_render.add_argument("--out", type=Path, required=True)
    kind = slurm_render.add_mutually_exclusive_group()
    kind.add_argument("--helium-only", action="store_true")
    kind.add_argument("--recovery-attempt", type=Path)
    export = commands.add_parser("export", help="copy a verified B5 analysis bundle")
    export.add_argument("run_dir", type=Path, help="completed workflow directory")
    export.add_argument("--out", type=Path, required=True, help="new local bundle directory")
    archive = commands.add_parser("archive", help="copy a verified full B5 workflow")
    archive.add_argument("run_dir", type=Path, help="completed workflow directory")
    archive.add_argument("--out", type=Path, required=True, help="new full archive directory")
    archive_verify = commands.add_parser("archive-verify", help="verify a retrieved full archive without its original paths")
    archive_verify.add_argument("archive", type=Path)
    archive_verify.add_argument("--receipt", type=Path, required=True, help="new verification receipt outside the archive")
    resume = commands.add_parser("resume", help="prepare a fresh B5 final-checkpoint continuation")
    resume.add_argument("archive", type=Path, help="verified full B5 archive")
    resume.add_argument("--checkpoint", type=Path, required=True, help="archive's final.restart.base")
    resume.add_argument("--seed", type=int, required=True, help="new positive SPARTA seed")
    resume.add_argument("--target-step", type=int, required=True, help="absolute SPARTA timestep to reach")
    resume.add_argument("--ranks", type=int, help="MPI ranks for the continuation; defaults to checkpoint rank count")
    resume.add_argument("--out", type=Path, required=True, help="new unrun resume directory")
    resume_run = commands.add_parser("resume-run", help="launch one prepared B5 restart through a local profile")
    resume_run.add_argument("resume_dir", type=Path, help="fresh directory made by resume")
    resume_run.add_argument("--parent-archive", type=Path, required=True)
    resume_run.add_argument("--profile", type=Path, required=True, help="local solver/MPI profile")
    checkpoint = commands.add_parser("checkpoint-archive", help="publish an allocated native-read-verified periodic checkpoint")
    checkpoint.add_argument("attempt", type=Path)
    checkpoint.add_argument("--checkpoint", type=Path, help="select one base file; default tries latest usable generation")
    checkpoint.add_argument("--profile", type=Path, required=True)
    checkpoint.add_argument("--out", type=Path, required=True)
    checkpoint_resume = commands.add_parser("resume-checkpoint", help="prepare a continuation from an independent checkpoint bundle")
    checkpoint_resume.add_argument("bundle", type=Path)
    checkpoint_resume.add_argument("--seed", type=int, required=True)
    checkpoint_resume.add_argument("--target-step", type=int, required=True)
    checkpoint_resume.add_argument("--ranks", type=int)
    checkpoint_resume.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare_case(load_case(args.case), args.out)
            print(f"Prepared: {path}")
            print("Next: python tools/dsmc.py inspect " + str(path.parent))
        elif args.command == "cad-workflow-prepare":
            from dsmc_cad_workflow import prepare_cad_workflow
            print(prepare_cad_workflow(args.checked, args.helium_config,
                                       args.transport_config, args.out))
        elif args.command == "slurm-benchmark":
            from benchmark_3d import collect_slurm_benchmark
            print(collect_slurm_benchmark(args.submission, args.archive, args.out))
        elif args.command == "frame":
            from dsmc_frames import select_frame
            selection = ({"last_complete": True} if args.last_complete else
                         {"timestep": args.timestep})
            result = select_frame(args.run_dir, selection)
            if args.json:
                if args.json.exists() or args.json.is_symlink():
                    raise FileExistsError(f"frame: refusing existing JSON path {args.json}")
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8", newline="\n")
            print(f"Frame: {result['path']} | step {result['timestep']} | {result['rows']} rows")
            print(f"Parent: {result['parent_manifest_status']} | SHA-256: {result['sha256']}")
            if result["rejected_newer_frames"]:
                print(f"Skipped newer invalid frames: {len(result['rejected_newer_frames'])}")
            if args.json:
                print(f"Selection JSON: {args.json}")
        elif args.command == "field":
            from dsmc_field import prepare_field
            selection = ({"last_complete": True} if args.last_complete else
                         {"timestep": args.timestep})
            receipt = prepare_field(args.prepared, args.parent, selection,
                                    args.gas_policy, args.out,
                                    splitmap=args.splitmap,
                                    splitmap_ids=args.splitmap_ids)
            print(f"Field preparation: {receipt}")
            print("Next for checked B5 circle/slit: python tools/dsmc.py molecule "
                  + str(receipt.parent) + " --out NEW_MOLECULE_DIRECTORY")
        elif args.command == "molecule":
            from dsmc_molecule import run_molecules
            receipt = run_molecules(args.field_dir, args.out, count=args.count,
                                    trace_count=args.trace_count)
            print(f"Molecule run: {receipt}")
        elif args.command == "molecule-batch":
            from dsmc_molecule import run_molecule_batch
            receipt = run_molecule_batch(args.field_dir, args.trials, args.out)
            print(f"Molecule batch: {receipt}")
        elif args.command == "bind-cad-helium":
            from dsmc_cad_workflow import bind_cad_helium
            print(bind_cad_helium(args.prepared, args.pilot_result, args.out))
        elif args.command == "run":
            from dsmc_pipeline import run_pipeline
            receipt = run_pipeline(args.prepared, args.profile, args.out,
                                   gas_policy=args.gas_policy,
                                   helium_backend=args.helium_backend,
                                   splitmap=args.splitmap, splitmap_ids=args.splitmap_ids,
                                   splitmap_solver=args.splitmap_solver)
            print(f"Full local workflow: {receipt}")
        elif args.command == "finish":
            from dsmc_pipeline import finish_pipeline
            print(finish_pipeline(args.prepared, args.helium_receipt, args.profile,
                                  args.out, gas_policy=args.gas_policy,
                                  splitmap=args.splitmap, splitmap_ids=args.splitmap_ids,
                                  splitmap_solver=args.splitmap_solver,
                                  splitmap_backend=args.splitmap_backend,
                                  splitmap_probe=args.splitmap_probe))
        elif args.command == "normalize-resume":
            from dsmc_resume_parent import normalize_resume
            print(normalize_resume(args.resume_dir, args.parent))
        elif args.command == "acceptance":
            if args.suite == "local":
                if args.dry_run or args.submit or args.prepared:
                    raise ValueError("local acceptance runs the tracked case; omit --dry-run/--submit/--prepared")
                from dsmc_acceptance import run_local_acceptance
                receipt = run_local_acceptance(args.profile, args.out,
                                               helium_backend=args.helium_backend)
                print(f"Local acceptance: {receipt}")
            elif args.suite == "handoff":
                if args.dry_run or args.submit or args.prepared:
                    raise ValueError("handoff acceptance runs the tracked case; omit --dry-run/--submit/--prepared")
                from dsmc_handoff import run_handoff_acceptance
                receipt = run_handoff_acceptance(args.profile, args.out,
                                                 helium_backend=args.helium_backend)
                print(f"Handoff rehearsal: {receipt}")
            else:
                if not args.prepared or args.helium_backend != "auto" or args.dry_run == args.submit:
                    raise ValueError("Slurm acceptance needs exactly one of --dry-run/--submit and --prepared, without a helium backend")
                from dsmc_slurm import _exclusive_json, _json, _now, render_slurm, submit_slurm
                site_profile = json.loads(args.profile.read_text(encoding="utf-8"))
                if args.dry_run:
                    script = render_slurm(args.prepared, site_profile, args.out)
                    print(f"Slurm dry-run script: {script}")
                else:
                    out = args.out.resolve()
                    if out.exists() or out.is_symlink():
                        raise FileExistsError(f"Slurm acceptance output already exists: {out}")
                    out.mkdir(parents=True)
                    acceptance_receipt = out / "acceptance.json"
                    _exclusive_json(acceptance_receipt, {
                        "schema_version": "dsmc-3d-slurm-acceptance-v1", "status": "rendering",
                        "created_utc": _now(), "suite": "slurm", "prepared": str(args.prepared.resolve()),
                        "profile": {"path": str(args.profile.resolve())},
                    })
                    try:
                        script = render_slurm(args.prepared, site_profile, out / "bundle")
                        attempt = submit_slurm(script.parent, out / "submission.json")
                        receipt = json.loads(acceptance_receipt.read_text(encoding="utf-8"))
                        receipt.update({"status": attempt["status"], "rendered_bundle": "bundle",
                                        "submission_receipt": {"path": "submission.json"},
                                        "updated_utc": _now()})
                        _json(acceptance_receipt, receipt)
                    except BaseException as exc:
                        receipt = json.loads(acceptance_receipt.read_text(encoding="utf-8"))
                        receipt.update({"status": "failed", "failure": str(exc), "updated_utc": _now()})
                        _json(acceptance_receipt, receipt)
                        raise
                    print(f"Slurm submitted: {attempt['status']} | receipt: {acceptance_receipt}")
        elif args.command == "slurm-submit":
            from dsmc_slurm import submit_slurm
            result = submit_slurm(args.bundle, args.receipt)
            print(json.dumps(result, indent=2))
        elif args.command == "slurm-status":
            from dsmc_slurm import status_slurm
            print(json.dumps(status_slurm(args.receipt), indent=2))
        elif args.command == "export":
            from dsmc_export import export as export_workflow
            manifest = export_workflow(args.run_dir, args.out)
            print(f"Analysis export: {manifest}")
        elif args.command == "archive":
            from dsmc_archive import archive_workflow
            manifest = archive_workflow(args.run_dir, args.out)
            print(f"Full archive: {manifest}")
        elif args.command == "archive-verify":
            from dsmc_archive import verify_retrieval
            print(json.dumps(verify_retrieval(args.archive, args.receipt), indent=2))
        elif args.command == "resume":
            from dsmc_resume import prepare_resume
            receipt = prepare_resume(args.archive, args.checkpoint, args.out,
                                     seed=args.seed, target_step=args.target_step,
                                     ranks=args.ranks)
            print(f"Prepared B5 restart: {receipt}")
        elif args.command == "resume-run":
            from dsmc_resume_run import run_resume, run_resume_checkpoint
            record = json.loads((args.resume_dir / "resume.json").read_text(encoding="utf-8"))
            runner = (run_resume_checkpoint if record.get("schema_version") ==
                      "dsmc-checkpoint-resume-preparation-v1" else run_resume)
            verification = runner(args.resume_dir, args.parent_archive, args.profile)
            print(f"Verified 3D restart: {verification}")
        elif args.command == "checkpoint-archive":
            from dsmc_checkpoint import archive_checkpoint
            print(archive_checkpoint(args.attempt, args.checkpoint, args.out, profile=args.profile))
        elif args.command == "resume-checkpoint":
            from dsmc_resume import prepare_resume_checkpoint
            print(prepare_resume_checkpoint(args.bundle, args.out, seed=args.seed,
                                             target_step=args.target_step, ranks=args.ranks))
        elif args.command == "slurm-render":
            from dsmc_slurm import render_slurm
            print(render_slurm(args.prepared, json.loads(args.profile.read_text(encoding="utf-8")),
                               args.out, helium_only=args.helium_only, recovery_attempt=args.recovery_attempt))
        else:
            if args.solver_check:
                if not args.profile:
                    raise ValueError("inspect --solver-check requires --profile")
                from dsmc_solver import solver_check
                original = Path(args.prepared).resolve()
                if original.is_file():
                    original = original.parent
                target = args.out or original.with_name(original.name + "-solver-checked")
                checked = solver_check(args.prepared, args.profile, target)
                args.prepared = checked.parent
                print(f"Solver-checked preparation: {checked}")
            elif args.out:
                raise ValueError("inspect --out requires --solver-check")
            elif args.profile:
                load_profile(args.profile)
            result = inspect_geometry(args.prepared)
            if args.json:
                if args.json.exists() or args.json.is_symlink():
                    raise FileExistsError(f"inspect: refusing existing JSON path {args.json}")
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            check = result["solver_check"]
            check_status = check["status"] if isinstance(check, dict) else check
            print(f"Case: {result['case_id']} | {result['state']} | solver check: {check_status}")
            area = result["geometry"]["physical_boundary"]["aperture_area_m2"]
            print(f"Aperture area: {area:.12g} m^2")
            if result["state"] == "solver-checked":
                print("Canonical solver triangles: " + str(result["geometry"]["physical_boundary"]["triangles"]))
            else:
                print("Pre-clip physical triangles: " + str(result["geometry"]["physical_boundary"]["triangles"]))
                print("Canonical solver triangles: pending explicit solver check")
            if args.json:
                print(f"Inspection JSON: {args.json}")
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        print(f"dsmc {args.command}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
