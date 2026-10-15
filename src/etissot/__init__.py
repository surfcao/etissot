"""eTissot: Tissot geometry for geographic embeddings without import-time dataset loading."""
from .geometry.fitting import fit_ellipse, fit_many, fitted_tensor, refine_radius
from .geometry.probing import destination_point, probe_points
from .geometry.indicatrix import measure_indicatrix, measure_locations
from .models.interface import EncoderAdapter
__version__ = '0.2.0'
__all__ = ['fit_ellipse', 'fit_many', 'fitted_tensor', 'refine_radius', 'destination_point', 'probe_points', 'measure_indicatrix', 'measure_locations', 'EncoderAdapter']
