"""Run the study outside Jupyter (results are cached in results/<stage>/ for the notebooks).

    python run_study.py pilot     # 5 truths per scenario, separate seeds (~2 min)
    python run_study.py main      # 30 truths per scenario (~10 min)
"""
import sys
import warnings

from vstudy.pipeline import run_study

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    stage = sys.argv[1] if len(sys.argv) > 1 else 'main'
    run_study(stage, force=True)
