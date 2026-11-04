"""Areas of interest (AOIs) for volumetrics.

In practice the AOI (lease, development block, drainage area) is fixed by decision makers before any
modelling, so the study's AOIs are fixed rectangles on the grid, identical for every truth model and
never chosen from data or results. Rectangles aligned with the grid are NumPy slices.
"""
from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass(frozen=True)
class AOI:
    name: str
    rows: slice
    cols: slice

    def mean(self, f):
        """Average over the AOI of a field (NY, NX) or a stack (n, NY, NX)."""
        return np.asarray(f)[..., self.rows, self.cols].mean(axis=(-2, -1))

    def ooip(self, f):
        """OOIP inside the AOI, MMSTB."""
        return C.ooip_cells(np.asarray(f)[..., self.rows, self.cols]).sum(axis=(-2, -1)) / 1e6

    def extent(self):
        x0, x1 = C.XMIN + self.cols.start * C.XSIZ, C.XMIN + self.cols.stop * C.XSIZ
        y1, y0 = C.YMAX - self.rows.start * C.YSIZ, C.YMAX - self.rows.stop * C.YSIZ
        return x0, x1, y0, y1

    def to_dict(self):
        return dict(name=self.name, rows=[int(self.rows.start), int(self.rows.stop)],
                    cols=[int(self.cols.start), int(self.cols.stop)])

    @staticmethod
    def from_dict(d):
        return AOI(d['name'], slice(*d['rows']), slice(*d['cols']))


STUDY_AOIS = [AOI(name, slice(r0, r1), slice(c0, c1)) for name, (r0, r1, c0, c1) in C.AOIS.items()]
LOCAL = [a for a in STUDY_AOIS if a.name in C.AOI_CHOICE]          # candidates for the development choice
