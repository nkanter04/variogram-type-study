"""All study settings in one place.

Conventions (GSLIB): azimuth clockwise from north (+Y); it 1 = spherical, 2 = exponential,
3 = Gaussian (practical ranges); cell centres at xmn = ymn = 5 m; grid row 0 = NORTH edge.
"""
import numpy as np

# ---------------------------------------------------------------- grid and reservoir
NX, NY = 100, 100
XSIZ, YSIZ = 10.0, 10.0
XMIN, YMIN = 0.0, 0.0
XMAX, YMAX = XMIN + NX * XSIZ, YMIN + NY * YSIZ
XC = XMIN + (np.arange(NX) + 0.5) * XSIZ                 # column centres, west -> east
YC = YMAX - (np.arange(NY) + 0.5) * YSIZ                 # row centres, north -> south
XX, YY = np.meshgrid(XC, YC)
GRID_XY = np.c_[XX.ravel(), YY.ravel()]

H_FT, BO = 50.0, 1.2
SW_A, SW_B, SW_LO, SW_HI = 0.55, 1.5, 0.15, 1.0
PHI_LO, PHI_HI = 0.01, 0.40
CELL_ACRES = XSIZ * YSIZ / 4046.8564224

# ---------------------------------------------------------------- truth scenarios
# Each scenario matches a theory about one variogram type. Variograms in Gaussian units (sill 1).
SCENARIOS = {
    # baseline: every type should work; nugget and range resolvable at 50+ wells
    'gaussian':    dict(transform='gaussian', mean=0.17, std=0.035,
                        vario=dict(it=1, nug=0.10, azi=0.0, hmaj=400.0, hmin=400.0)),
    # right-skewed porosity: relative, log and normal-score variograms are meant to help
    'skewed':      dict(transform='lognormal', median=0.12, log_std=0.45,
                        vario=dict(it=1, nug=0.10, azi=0.0, hmaj=400.0, hmin=250.0)),
    # strong anisotropy: tests azimuth inference and short minor range
    'anisotropic': dict(transform='gaussian', mean=0.17, std=0.035,
                        vario=dict(it=1, nug=0.05, azi=45.0, hmaj=600.0, hmin=150.0)),
    # bimodal sand/shale mixture (truncated Gaussian facies): robust types are meant to help
    'facies':      dict(transform='facies', sand_prop=0.60, sand=(0.21, 0.030), shale=(0.06, 0.015),
                        vario=dict(it=1, nug=0.05, azi=-60.0, hmaj=500.0, hmin=200.0),
                        facies_vario=dict(it=1, nug=0.0, azi=-60.0, hmaj=600.0, hmin=250.0)),
}
SCEN_NAMES = list(SCENARIOS)

N_REAL = 30                     # truth models per scenario (main study)
N_REAL_PILOT = 5                # truth models per scenario (pilot, separate seeds)
SEED_MAIN, SEED_PILOT = 20261014, 777
WELL_COUNTS = [25, 50, 100]
DESIGNS = ['random', 'preferential']
WT_MIN, WT_MAX = 0.1, 1.0       # preferential sampling weight for lowest / highest porosity rank

# ---------------------------------------------------------------- variogram calculation
# gamv types from the course list (1, 4, 5, 6, 7, 8) + cumulative + normal-score semivariogram
VTYPES = ['semivariogram', 'correlogram', 'general_relative', 'pairwise_relative',
          'log', 'rodogram', 'cumulative', 'normal_score']
AZIMUTHS = np.arange(0.0, 180.0, 15.0)   # candidate major directions
ATOL = 22.5
MAX_LAG = 600.0                          # calculate lags out to ~ half the field
MIN_PAIRS = 5
BAND_FACTOR = 3.0                        # bandwidth = BAND_FACTOR x lag spacing

# ---------------------------------------------------------------- model fitting (same for every type)
FIT_STRUCTURES = (1, 2)                  # spherical, exponential (Gaussian without nugget makes kriging unstable)
FIT_NUGS = np.round(np.arange(0.0, 0.9001, 0.025), 3)
FIT_RANGES = np.arange(30.0, 1500.1, 20.0)
SCORE_HSHORT = 250.0                     # variogram accuracy scored on lags <= 250 m

# ---------------------------------------------------------------- uncertainty and decisions
N_SIM = 50                               # conditional simulations (normal-score model) for the OOIP range
N_QUAD = 20                              # Gauss-Hermite nodes for the expected-value (E-type) map
K_WELLS, R_DRAIN = 3, 150.0
RF, NETBACK, WELL_COST = 0.30, 35.0, 6.0  # recovery factor, $/bbl netback, $MM per well
CAND_STEP = 2                            # candidate sites every 2 cells (20 m)
N_RANDOM = 200
TRIM_SAME_SITES = 90.0                   # pilot rule: merge types that pick identical sites this often (%)

LABEL = {'semivariogram': 'Semivariogram', 'correlogram': 'Correlogram', 'general_relative': 'General relative',
         'pairwise_relative': 'Pairwise relative', 'log': 'Log semivariogram', 'rodogram': 'Rodogram',
         'cumulative': 'Cumulative', 'normal_score': 'Normal-score semivariogram', 'loocv': 'Cross-validation pick',
         'truth_variogram': 'True variogram', 'idw': 'Inverse distance (no variogram)',
         'simulation_ev': 'Simulation, expected value', 'random': 'Random sites', 'naive': 'Declustered mean'}


def sw_from_phi(phi):
    return np.clip(SW_A - SW_B * np.asarray(phi), SW_LO, SW_HI)


def ooip_cells(phi):
    """STB per cell (7758 A h phi (1 - Sw) / Bo)."""
    phi = np.asarray(phi)
    return 7758.0 * CELL_ACRES * H_FT * phi * (1.0 - sw_from_phi(phi)) / BO


def ooip_total(phi):
    """MMSTB over the grid (or per realization if phi is a stack)."""
    return ooip_cells(phi).sum(axis=(-2, -1)) / 1e6


def to_money(mmstb):
    return mmstb * RF * NETBACK
