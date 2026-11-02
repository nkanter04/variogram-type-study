"""Run the study outside Jupyter (results are cached in results/ for the notebooks).

    python run_study.py pilot     # 5 truths per scenario, separate seeds (~1 min)
    python run_study.py main      # 30 truths per scenario (~5 min)
    python run_study.py aoi       # AOI volumetrics with SGS, 100 realizations per variogram model (~2 min)
"""
import sys
import warnings

from vstudy.pipeline import run_study

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    stage = sys.argv[1] if len(sys.argv) > 1 else 'main'
    if stage == 'aoi':
        from vstudy import aoi
        aoi.run(force=True)
    else:
        run_study(stage, force=True)
