#!/usr/bin/env python3
"""Local self-check for the tracked Engaging second-user setup (Linux/WSL only).

    python3 tools/engaging/test_engaging_setup.py            # offline checks
    DSMC_TEST_NETWORK=1 python3 tools/engaging/test_engaging_setup.py  # + real pip install

Covers: bash syntax; the sbatch SPARTA pin equals tools/wsl/build_sparta_cpu.sh;
write_profiles.py output passes dsmc_slurm.validate_profile / dsmc_case.load_profile
and its negative controls; ctest_gate.py on the synthetic BIGBIG log in
tools/wsl/fixtures/ (PASS) and on a truncated copy (FAIL); sparta-smoke.sbatch run end to
end with stub cmake/ctest/mpirun so its clone, pin check, re-submit refusal and
acceptance exit status (runtime smokes + the real collision-rate analyzer on
stubbed solver output; the ctest gate is information only) are exercised; setup-env.sh refusals, and (network mode) a full venv build with a
stub Julia plus a Manifest-rewriting Julia that must be caught. Python >= 3.11.
No SPARTA, Slurm, module system or Julia is needed.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
from dsmc_case import load_profile  # noqa: E402
from dsmc_slurm import validate_profile  # noqa: E402

SBATCH = HERE / "sparta-smoke.sbatch"
SETUP = HERE / "setup-env.sh"
WRITER = HERE / "write_profiles.py"
WSL_LOG = REPO / "tools/wsl/fixtures/ctest-synthetic-bigbig.log"
GIT = ["git", "-c", "safe.directory=*", "-c", "user.name=t", "-c", "user.email=t@t"]


def run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, timeout=900, **kw)


def script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


class Syntax(unittest.TestCase):
    def test_bash_n(self):
        for path in (SETUP, SBATCH, HERE / "he-collision-runs.sh", HERE / "he-collision-check.sbatch"):
            r = run(["bash", "-n", str(path)])
            self.assertEqual(r.returncode, 0, (path, r.stderr))

    def test_pin_matches_build_recipe(self):
        pat = r"^SPARTA_{}=\$\{{SPARTA_{}:-(\S+)\}}$"
        for key in ("URL", "TAG", "COMMIT"):
            ours = re.findall(pat.format(key, key), SBATCH.read_text(), re.M)
            recipe = re.findall(pat.format(key, key), (REPO / "tools/wsl/build_sparta_cpu.sh").read_text(), re.M)
            self.assertEqual(len(ours), 1, key)
            self.assertEqual(ours, recipe, key)


class Profiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "w5"
        for d in ("repo", "python-venv/bin", "config"):
            (self.root / d).mkdir(parents=True)
        (self.root / "python-venv/bin/python").write_text("")
        (self.root / "config/setup.sh").write_text("")
        self.solver = self.tmp / "spa_mpi_bigbig"
        self.solver.write_bytes(b"solver")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, *extra, root=None):
        return run([sys.executable, str(WRITER), "--root", str(root or self.root), "--account", "mit_general",
                    "--partition", "mit_normal", "--solver", str(self.solver), *extra])

    def test_output_passes_maintained_validators(self):
        r = self.write()
        self.assertEqual(r.returncode, 0, r.stderr)
        site = json.loads((self.root / "site-profile.json").read_text())
        validate_profile(site)  # dsmc_slurm.py:74-134
        self.assertIs(site["site_runtime_verified"], False)
        self.assertEqual((site["account"], site["partition"], site["ranks"]), ("mit_general", "mit_normal", 4))
        self.assertEqual(site["setup_script"], str(self.root.resolve() / "config/setup.sh"))
        launch = load_profile(self.root / "solver-check-profile.json")
        self.assertEqual(launch["solver_executable"], str(self.solver.resolve()))
        inputs = json.loads((self.root / "profile-inputs.json").read_text())
        import hashlib
        for key, name in (("site_profile_sha256", "site-profile.json"),
                          ("solver_check_profile_sha256", "solver-check-profile.json")):
            self.assertEqual(inputs[key], hashlib.sha256((self.root / name).read_bytes()).hexdigest())
        again = self.write()
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("refusing to overwrite", again.stderr)

    def test_account_none_is_null(self):
        self.assertEqual(self.write("--site-id", "x").returncode, 0)
        shutil.rmtree(self.root / "repo")
        (self.root / "repo").mkdir()
        for name in ("site-profile.json", "solver-check-profile.json", "profile-inputs.json"):
            (self.root / name).unlink()
        r = run([sys.executable, str(WRITER), "--root", str(self.root), "--account", "none",
                 "--partition", "mit_normal", "--solver", str(self.solver)])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIsNone(json.loads((self.root / "site-profile.json").read_text())["account"])

    def test_negative_controls(self):
        bad = self.tmp / "CHANGE_ME" / "w5"
        shutil.copytree(self.root, bad)
        r = self.write(root=bad)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("placeholder", r.stderr)
        self.assertFalse((bad / "site-profile.json").exists())
        r = run([sys.executable, str(WRITER), "--root", str(self.root), "--account", "mit_general",
                 "--partition", "bad partition", "--solver", str(self.solver)])
        self.assertIn("invalid Slurm partition", r.stderr)
        self.assertFalse((self.root / "site-profile.json").exists())
        (self.root / "config/setup.sh").unlink()
        self.assertIn("setup-env.sh first", self.write().stderr)


class Gate(unittest.TestCase):
    def gate(self, log):
        return run([sys.executable, str(REPO / "tools/wsl/ctest_gate.py"), str(log), "--variant", "bigbig"])

    def test_synthetic_bigbig_log_passes(self):
        r = self.gate(WSL_LOG)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("PASS", r.stdout)

    def test_truncated_log_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "truncated.log"
            bad.write_text(WSL_LOG.read_text().split("\n75% tests passed")[0])
            r = self.gate(bad)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("no CTest summary line", r.stdout)


def collide_log(ncoll):
    """SPARTA-shaped stats for the stage2-collide deck at 4 K. analyze.py expects
    ~102.9 collisions/step there, so ncoll 103 gives R ~ 1.00 and 80 gives R ~ 0.78."""
    rows = [f"{s} 0.1 50000 200 {ncoll + (1 if s % 400 else -1)} 4.0" for s in range(0, 20001, 200)]
    return "Step CPU Np Natt Ncoll c_myT\n" + "\n".join(rows) + "\nLoop time of 1 on 1 procs\n"


class SbatchWiring(unittest.TestCase):
    """Run sparta-smoke.sbatch with stubs; the clone/pin/gate/acceptance logic,
    he-collision-runs.sh and the collision analyzer are real, the solver is not."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        up = self.tmp / "upstream"
        up.mkdir()
        (up / "README").write_text("stub sparta\n")
        run(GIT + ["-C", str(up), "init", "-q"])
        run(GIT + ["-C", str(up), "add", "README"])
        run(GIT + ["-C", str(up), "commit", "-qm", "stub"])
        self.url, self.commit = str(up), run(GIT + ["-C", str(up), "rev-parse", "HEAD"]).stdout.strip()
        stub = self.tmp / "stubbin"
        stub.mkdir()
        script(stub / "cmake", 'b=; while (($#)); do [ "$1" = -B ] && b=$2; shift; done\n'
               'if [ -n "$b" ]; then mkdir -p "$b/src"; printf "#!/bin/sh\\n" > "$b/src/spa_mpi_bigbig"\n'
               '  chmod +x "$b/src/spa_mpi_bigbig"; echo SPARTA_MACHINE:STRING=mpi_bigbig > "$b/CMakeCache.txt"; fi')
        script(stub / "ctest", 'cat "$CTEST_FIXTURE"; exit 8')
        # The collision deck gets the fixture's stats; everything else just echoes.
        script(stub / "mpirun", 'case "$*" in *in.he_collide*) cat "$COLLIDE_FIXTURE"; exit "${COLLIDE_RC:-0}";; esac\n'
               'echo stub-mpirun')
        for name in ("lscpu", "ldd", "module"):
            script(stub / name, "echo stub-" + name)
        self.stub = stub
        self.good = self.tmp / "collide-good.log"
        self.good.write_text(collide_log(103))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def submit(self, name, fixture, prepare=None, collide=None, reuse=False, **extra):
        root = self.tmp / name
        if not reuse:
            root.mkdir()
            shutil.copy(SBATCH, root / "sparta-smoke.sbatch")
            shutil.copy(HERE / "he-collision-runs.sh", root / "he-collision-runs.sh")
            script(root / "smoke-setup.sh", f"export PATH={self.stub}:$PATH")
            script(root / "sparta-runtime-smokes.sh", 'mkdir "$2"; exit "${SMOKE_RC:-0}"')
            (root / "repo").symlink_to(REPO)  # the walkthrough links the clone here the same way
        if prepare:
            prepare(root)
        env = {**os.environ, "SLURM_SUBMIT_DIR": str(root), "SPARTA_URL": self.url,
               "SPARTA_COMMIT": self.commit, "CTEST_FIXTURE": str(fixture),
               "COLLIDE_FIXTURE": str(collide or self.good), **extra,
               "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*"}
        r = run(["bash", str(root / "sparta-smoke.sbatch")], cwd=root, env=env)
        return root, r

    def assert_accepted(self, root, r):
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((root / "sparta.exit").read_text().strip(), "0")
        text = (root / "acceptance.txt").read_text()
        self.assertTrue(text.startswith("acceptance PASS"), text)
        for ranks in ("1", "4"):
            self.assertIn("VERDICT: PASS", (root / "he-collision" / ranks / "analysis.txt").read_text())
        return text

    def test_clone_pin_and_gate_pass(self):
        root, r = self.submit("pass", WSL_LOG)
        text = self.assert_accepted(root, r)
        self.assertEqual(run(GIT + ["-C", str(root / "sparta-source"), "rev-parse", "HEAD"]).stdout.strip(), self.commit)
        self.assertIn(self.commit, (root / "sparta-source.pin").read_text())
        self.assertEqual((root / "ctest.exit").read_text().strip(), "8")  # raw ctest nonzero
        self.assertEqual((root / "ctest-gate.exit").read_text().strip(), "0")
        self.assertIn("ctest gate PASS (bigbig)", (root / "ctest-gate.txt").read_text())
        self.assertIn("information only", text)
        # 4-rank copy gets the decomposition line; the 1-rank copy stays the original deck.
        self.assertIn("balance_grid rcb cell", (root / "he-collision/4/in.he_collide").read_text())
        self.assertNotIn("balance_grid", (root / "he-collision/1/in.he_collide").read_text())

    def test_ctest_gate_failure_is_information_only(self):
        bad = self.tmp / "truncated.log"
        bad.write_text(WSL_LOG.read_text().split("\n75% tests passed")[0])
        root, r = self.submit("truncated", bad)
        text = self.assert_accepted(root, r)
        self.assertEqual((root / "ctest-gate.exit").read_text().strip(), "1")
        self.assertIn("ctest gate FAIL", (root / "ctest-gate.txt").read_text())
        self.assertIn("ctest gate FAIL", text)  # recorded in the verdict file, not acted on

    def test_science_failure_is_the_exit_status(self):
        slow = self.tmp / "collide-slow.log"
        slow.write_text(collide_log(80))  # R ~ 0.78: analyze.py's FAIL band
        cases = {"rate": dict(collide=slow), "crash": dict(COLLIDE_RC="1"), "smoke": dict(SMOKE_RC="1")}
        for name, kw in cases.items():
            with self.subTest(name):
                root, r = self.submit("sci-" + name, WSL_LOG, **kw)  # a PASSing gate must not rescue it
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual((root / "sparta.exit").read_text().strip(), str(r.returncode))
                self.assertTrue((root / "acceptance.txt").read_text().startswith("acceptance FAIL"))
                self.assertIn("ctest gate PASS", (root / "ctest-gate.txt").read_text())  # still written
                self.assertTrue((root / "he-collision/4/analysis.txt").is_file())  # both checks ran

    def test_resubmit_into_used_dir_refused(self):
        root, r = self.submit("reuse", WSL_LOG)
        self.assert_accepted(root, r)
        before = {n: (root / n).read_text() for n in ("sparta.exit", "sparta-progress.log", "acceptance.txt")}
        root, r = self.submit("reuse", WSL_LOG, reuse=True)
        self.assertEqual(r.returncode, 6, r.stdout + r.stderr)
        self.assertIn("refusing to reuse", r.stderr)
        for n, text in before.items():
            self.assertEqual((root / n).read_text(), text, n)  # earlier evidence untouched

    def test_dirty_source_refused(self):
        def dirty(root):
            run(GIT + ["clone", "-q", self.url, str(root / "sparta-source")])
            (root / "sparta-source/extra").write_text("x")
        root, r = self.submit("dirty", WSL_LOG, dirty)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertFalse((root / "sparta-build").exists())

    def test_missing_gate_refused_before_build(self):
        def no_repo(root):
            (root / "repo").unlink()
        root, r = self.submit("nogate", WSL_LOG, no_repo)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertFalse((root / "sparta-build").exists())


@unittest.skipUnless(sys.version_info >= (3, 11), "setup-env.sh needs Python 3.11+ as python3")
class SetupEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "w5"
        self.root.mkdir()
        r = run(GIT + ["clone", "-q", "--depth", "1", "file://" + str(REPO), str(self.root / "repo")])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.stub = self.tmp / "stubbin"
        self.stub.mkdir()
        script(self.stub / "python3", f'exec {sys.executable} "$@"')
        script(self.stub / "mpirun", "echo stub-mpirun")
        self.modules = script(self.tmp / "modules.sh", f"export PATH={self.stub}:$PATH")

    def tearDown(self):
        subprocess.run(["chmod", "-R", "u+w", str(self.tmp)])
        shutil.rmtree(self.tmp)

    def setup_env(self, julia_body="echo 'Status stub'"):
        script(self.stub / "julia", julia_body)
        env = {**os.environ, "DSMC_ROOT": str(self.root), "SITE_MODULE_SETUP": str(self.modules),
               "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*"}
        return run(["bash", str(SETUP)], env=env)

    def test_existing_output_refused(self):
        (self.root / "python-venv").mkdir()
        r = self.setup_env()
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing existing", r.stderr)
        self.assertFalse((self.root / "config").exists())

    def test_dirty_source_refused_before_install(self):
        (self.root / "repo/tools/untracked_tool.py").write_text("")
        r = self.setup_env()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("untracked implementation files", r.stderr)
        self.assertFalse((self.root / "logs/pip-install.log").exists())

    def test_wrong_expected_commit_refused(self):
        os.environ["EXPECTED_COMMIT"] = "0" * 40
        try:
            r = self.setup_env()
        finally:
            del os.environ["EXPECTED_COMMIT"]
        self.assertIn("EXPECTED_COMMIT", r.stderr)
        self.assertFalse((self.root / "python-venv").exists())

    @unittest.skipUnless(os.environ.get("DSMC_TEST_NETWORK") == "1", "set DSMC_TEST_NETWORK=1 for pip installs")
    def test_full_install_then_manifest_rewrite_caught(self):
        r = self.setup_env()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        record = json.loads((self.root / "env/environment.json").read_text())
        self.assertEqual(record["status"], "environment-ready")
        self.assertEqual(record["python"]["sys_prefix"], str(self.root / "python-venv"))
        check = run(["sha256sum", "-c", str(self.root / "env/inputs.sha256")])
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
        text = (self.root / "config/setup.sh").read_text()
        self.assertNotIn("set -e", text)
        self.assertIn(f"JULIA_DEPOT_PATH={self.root}/julia-depot", text)
        # Negative control: a resolver that rewrites the Manifest must fail verify-after.
        shutil.rmtree(self.root)
        self.root.mkdir()
        run(GIT + ["clone", "-q", "--depth", "1", "file://" + str(REPO), str(self.root / "repo")])
        r = self.setup_env('m=$JULIA_PROJECT/Manifest.toml; chmod u+w "$m"; echo "# rewritten" >> "$m"')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.root / "env/environment.json").exists())
        self.assertFalse((self.root / "config/setup.sh").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
