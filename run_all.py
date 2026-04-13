"""Batch-run all FA PDFs through the pipeline and print a summary table."""

import asyncio
import glob
from pathlib import Path

from pipeline import run_pipeline


async def main():
    fa_files = sorted(glob.glob("finalassets/*.pdf"))
    print(f"Found {len(fa_files)} FA files\n")

    results = []
    for fa_path in fa_files:
        print(f"\n{'='*70}")
        print(f"Processing: {Path(fa_path).name}")
        print(f"{'='*70}")
        try:
            report = await run_pipeline(fa_path=fa_path, isi_dir="isi", debug=True)
            results.append(report)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"fa": Path(fa_path).name, "error": str(e)})

    # Summary table
    print(f"\n\n{'='*120}")
    print(f"{'FA':<65} {'ISI':<45} {'Cov':>5} {'Auth':>5} {'F1':>5}  {'Match'}")
    print(f"{'-'*120}")
    for r in results:
        if "error" in r:
            print(f"{r['fa']:<65} ERROR: {r['error']}")
            continue
        o = r.get("overall", {})
        isi_name = r.get("isi") or "None"
        print(
            f"{r['fa']:<65} {isi_name:<45} "
            f"{o.get('coverage',0):5.1f} {o.get('authenticity',0):5.1f} {o.get('f1',0):5.1f}  "
            f"{r.get('match_category','?')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
