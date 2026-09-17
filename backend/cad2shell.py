#!/usr/bin/env python3
"""cad2shell CLI -- part mesh in, investment shell mould out.

    python cad2shell.py part.stl -o out/part --alloy A356 --shell 6 --pitch 1.5
    python cad2shell.py --demo boss_plate -o out/demo
    python cad2shell.py --list-alloys
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cad2shell import Config, demo_parts, run
from cad2shell.config import ALLOYS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cad2shell",
        description="Generate a ceramic investment-shell mould with a full gating tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("part", nargs="?", help="part mesh (STL/OBJ/PLY/3MF)")
    p.add_argument("--demo", choices=sorted(demo_parts.DEMO_PARTS),
                   help="use a built-in demo part instead of a file")
    p.add_argument("-o", "--out", default="out/part", help="output prefix")
    p.add_argument("--alloy", default="A356", choices=sorted(ALLOYS))
    p.add_argument("--density", type=float, help="kg/mm^3, overrides the alloy preset")
    p.add_argument("--shell", type=float, default=6.0, help="shell thickness, mm")
    p.add_argument("--pitch", type=float, default=1.5, help="voxel pitch, mm")
    p.add_argument("--pour-time", type=float, default=6.0, help="target fill time, s")
    p.add_argument("--riser-factor", type=float, default=1.2, help="M_riser / M_section")
    p.add_argument("--max-risers", type=int, default=4)
    p.add_argument("--no-png", action="store_true", help="skip the cross-section render")
    p.add_argument("--json", metavar="PATH", help="also write the stats JSON here")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--list-alloys", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_alloys:
        for k, v in sorted(ALLOYS.items()):
            print(f"  {k:10} {v['density']:.2e} kg/mm^3   {v['label']}")
        return 0

    if not args.part and not args.demo:
        build_parser().error("give a part file or --demo NAME")

    cfg = Config(
        alloy=args.alloy, density=args.density, shell_thickness=args.shell,
        voxel_pitch=args.pitch, pour_time=args.pour_time,
        riser_modulus_factor=args.riser_factor, max_risers=args.max_risers,
        write_section_png=not args.no_png,
    )

    mesh = demo_parts.get(args.demo) if args.demo else None
    try:
        stats = run(part_path=args.part, out_prefix=args.out, cfg=cfg, mesh=mesh)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    d = stats.as_dict()
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(stats.to_json())

    if not args.quiet:
        th = d["shell"]["thickness_measured"]
        print(f"part            {d['part_volume']/1000:8.2f} cm^3   "
              f"modulus {d['casting_modulus']:.2f} mm")
        print(f"hot spots       {len(d['hot_spots'])}")
        print(f"choke           d={2*d['gating']['choke_radius']:.2f} mm  "
              f"A={d['gating']['choke_area']:.1f} mm^2  "
              f"v={d['gating']['velocity']/1000:.2f} m/s")
        print(f"sprue           {2*d['gating']['sprue_r_bottom']:.1f} -> "
              f"{2*d['gating']['sprue_r_top']:.1f} mm")
        print(f"runner/ingate   {2*d['gating']['runner_radius']:.2f} / "
              f"{2*d['gating']['ingate_radius']:.2f} mm  x{d['gating']['n_ingates']}")
        for r in d["risers"]:
            print(f"riser {r['index']}         d={r['diameter']:.1f} h={r['height']:.1f} mm  "
                  f"M={r['modulus']:.2f} vs {r['feeds_modulus']:.2f} (x{r['factor']:.2f})")
        print(f"shell wall      {th['median']:.2f} mm (target {d['shell']['thickness_target']:.1f})  "
              f"watertight={d['shell']['watertight']}")
        print(f"metal / yield   {d['metal_volume']/1000:.2f} cm^3   {d['yield_pct']:.1f} %")
        for w in d["warnings"]:
            print(f"warning: {w}")
        for k, v in d["files"].items():
            print(f"wrote {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
