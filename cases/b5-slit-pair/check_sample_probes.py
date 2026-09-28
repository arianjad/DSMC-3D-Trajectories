#!/usr/bin/env python3
"""Check both sampled probes, including streaming their complete field records."""
import gzip, hashlib, json, math, os, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
from equilibrium import read_dump

TAG = os.environ.get('PAIR_TAG', '20260918')

def field_path(d, step):
    for name in (f'fields.{step}.gz', f'fields.{step}'):
        if (d / name).exists():
            return d / name
    raise AssertionError((d, f'fields.{step}[.gz] missing'))

def fopen(path):
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()

def check_field(path, expected_step):
    with fopen(path) as stream:
        assert stream.readline().strip() == 'ITEM: TIMESTEP'
        assert int(stream.readline()) == expected_step
        assert stream.readline().strip() == 'ITEM: NUMBER OF CELLS'
        expected_rows = int(stream.readline())
        for line in stream:
            if line.startswith('ITEM: CELLS '):
                columns = line.split()[2:]
                break
        else:
            raise AssertionError('missing CELLS header')
        ni = columns.index('f_afields[1]')
        rhoi = columns.index('f_afields[2]')
        ti = columns.index('f_afields[9]')
        rows = occupied = gas_zero_t = 0
        max_rho = 0.0
        for line in stream:
            vals = line.split()
            assert len(vals) == len(columns), (path, rows, 'truncated row')
            numbers = list(map(float, vals[1:]))  # idstr may be hierarchical
            assert all(math.isfinite(v) for v in numbers), (path, rows, 'nonfinite')
            n, rho, temp = float(vals[ni]), float(vals[rhoi]), float(vals[ti])
            # thermal/grid forms <v^2> - <v>^2; in 1-3-sample cells this cancels to
            # O(-1e-15) K (measured 2026-09-18 on both sampled probes: 12-19 k cells
            # per frame, all above -1e-14). Read those as 0 K, reject anything colder.
            assert n >= 0 and rho >= 0 and temp >= -1e-9, (path, rows, n, rho, temp)
            rows += 1
            occupied += rho > 0
            gas_zero_t += rho > 0 and temp <= 0
            max_rho = max(max_rho, rho)
        assert rows == expected_rows, (path, rows, expected_rows)
        assert occupied > 0, (path, 'all-zero density')
    return dict(file=path.name, rows=rows, occupied=occupied, gas_zero_T=gas_zero_t,
                max_rho=max_rho, bytes=path.stat().st_size)

def main():
    root = Path(sys.argv[1]); result = []
    for arm in ('circle', 'slit'):
        d = root / f'b5-pair-s25-{arm}-sampleprobe-{TAG}'
        assert (d / 'rc.sentinel').read_text().strip() == '0'
        manifest = json.loads((d / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        # Stage comparison strips CR exactly as run_lean.sh does.
        for p in (HERE / arm).iterdir():
            if p.name == 'in.sampled' or p.suffix == '.surf':
                assert p.read_bytes().replace(b'\r', b'') == (d / p.name).read_bytes(), p
        log = (d / 'run.log').read_text()
        assert 'ERROR' not in log
        counts = re.findall(r'(\d+)\s+(\d+)\s+(\d+) = cells outside/inside/overlapping surfs', log)
        assert counts and all(int(a) > int(b) for a,b,c in counts)
        corners = re.findall(r'(\d+)\s+(\d+) = cells overlapping surfs, overlap cells with unmarked corner pts', log)
        assert corners and all(int(b) == 0 for a,b in corners), corners
        stations = {k: read_dump(d / f'equil-{k}.out') for k in ('tf','tr','af','ar')}
        assert all(v and max(v.values()) > 0 for v in stations.values())
        source = [line.split() for line in (d / 'equil-source.out').read_text().splitlines()
                  if line.strip() and not line.startswith('#')]
        assert source and any(float(row[1]) > 0 for row in source)
        loops = [(float(t), int(n)) for t,n in re.findall(r'Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps', log)]
        assert sum(n for _,n in loops) == 4800
        fields = [check_field(field_path(d, step), step) for step in (2400,4800)]
        result.append(dict(arm=arm, status='passed', classifications=counts, unmarked=corners,
                           loops=loops, fields=fields,
                           station_atoms_per_s={k:list(v.values())[-1] for k,v in stations.items()}))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()