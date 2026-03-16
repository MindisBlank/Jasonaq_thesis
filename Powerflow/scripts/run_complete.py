#!/usr/bin/env python3
"""
run_complete.py
================
Pipeline orchestrator: runs the full power flow analysis pipeline
from data translation through loss comparison.

Processes each substation listed in jason_config.yaml consecutively.

Steps (per substation):
  1. Jason_data_adapter.py       -- Translate ArcGIS topology to power flow format
  2. Jason_prepare_powerflow.py  -- Build per-phase load matrices from smart meter data
  3. Jason_run_powerflow.py      -- Run 3-phase power flow (unbalanced + balanced)
  4. Jason_compare_losses.py     -- Compare losses and generate plots

Usage:
    python run_complete.py                          # Full run (all IDs in config)
    python run_complete.py --max-timesteps 10       # Quick test (10 timesteps)
    python run_complete.py --skip-adapter --skip-prepare  # Re-run PF only
    python run_complete.py --mode unbalanced        # Only unbalanced run
"""

import subprocess
import sys
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(step_name, cmd):
    """Run a pipeline step and abort on failure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Step: {step_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"FAILED: {step_name} (exit code {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)

    logger.info(f"OK: {step_name} ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description='Run the complete power flow analysis pipeline')
    parser.add_argument('--max-timesteps', type=int, default=None,
                        help='Limit number of timesteps (for testing)')
    parser.add_argument('--skip-adapter', action='store_true',
                        help='Skip Jason_data_adapter.py (topology already translated)')
    parser.add_argument('--skip-prepare', action='store_true',
                        help='Skip Jason_prepare_powerflow.py (load matrices already built)')
    parser.add_argument('--mode', choices=['unbalanced', 'balanced', 'both'], default='both',
                        help='Power flow mode (default: both)')
    args = parser.parse_args()

    # Load substation IDs from config
    from Jason_config import load_substation_ids
    ids = load_substation_ids()

    logger.info("=" * 80)
    logger.info("Jason Power Flow Pipeline - Complete Run")
    logger.info(f"Substations: {ids}")
    logger.info("=" * 80)

    total_start = time.time()
    python = sys.executable

    for sub_id in ids:
        logger.info(f"\n{'#'*80}")
        logger.info(f"  SUBSTATION {sub_id}")
        logger.info(f"{'#'*80}")

        sub_start = time.time()

        # Step 1: Data adapter
        if not args.skip_adapter:
            run_step(f"Data Adapter [{sub_id}]",
                     [python, str(SCRIPT_DIR / "Jason_data_adapter.py"),
                      "--substation", sub_id])
        else:
            logger.info(f"\nSkipping data adapter for {sub_id} (--skip-adapter)")

        # Step 2: Prepare power flow
        if not args.skip_prepare:
            run_step(f"Prepare Power Flow [{sub_id}]",
                     [python, str(SCRIPT_DIR / "Jason_prepare_powerflow.py"),
                      "--substation", sub_id])
        else:
            logger.info(f"\nSkipping prepare step for {sub_id} (--skip-prepare)")

        # Step 3: Run power flow
        pf_cmd = [python, str(SCRIPT_DIR / "Jason_run_powerflow.py"), "--mode", args.mode]
        if args.max_timesteps:
            pf_cmd.extend(["--max-timesteps", str(args.max_timesteps)])
        run_step(f"Run Power Flow [{sub_id}]", pf_cmd)

        # Step 4: Compare losses (only if mode includes both runs)
        if args.mode == 'both':
            run_step(f"Compare Losses [{sub_id}]",
                     [python, str(SCRIPT_DIR / "Jason_compare_losses.py")])
        else:
            logger.info(f"\nSkipping loss comparison for {sub_id} (mode={args.mode}, need 'both')")

        sub_elapsed = time.time() - sub_start
        logger.info(f"\nSubstation {sub_id} done ({sub_elapsed:.1f}s / {sub_elapsed/60:.1f} min)")

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*80}")
    logger.info(f"Pipeline complete! {len(ids)} substation(s) in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
