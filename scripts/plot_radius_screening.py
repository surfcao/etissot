"""Plot the radius-screening curves from the declared measurements.

Equivalent to `python -m etissot.reporting.radius_screening --data data --output outputs`.
To plot freshly collected curves, run scripts/radius_screening.py and copy
outputs/summaries/radius_screening.json to data/measurements/review/.
"""
import sys
from runpy import run_module

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv += ['--data', 'data', '--output', 'outputs']
    run_module('etissot.reporting.radius_screening', run_name='__main__')
