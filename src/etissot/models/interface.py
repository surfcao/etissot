"""Framework-independent adapters for user-supplied location encoders."""
from dataclasses import dataclass, field
from copy import deepcopy
import json
from typing import Callable, Mapping, Any
import numpy as np


def feature_matrix(output, rows):
    """Validate real feature rows without importing an ML framework."""
    values = np.asarray(output)
    if np.iscomplexobj(values):
        raise ValueError('Convert complex features to explicit real components before measurement.')
    # Own the rows: an encoder may reuse an output buffer on its next call.
    values = np.array(values, dtype=float, copy=True)
    if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
        raise ValueError('Encoder must return one nonempty feature row per requested location.')
    if not np.isfinite(values).all():
        raise ValueError('Encoder returned nonfinite features.')
    return values


@dataclass(frozen=True)
class EncoderAdapter:
    """Adapt a batched model to canonical latitude/longitude in radians.

    ``encode`` receives an (N, 2) coordinate array in the declared units/order.
    ``feature_transform`` may select an output field or convert framework
    tensors to NumPy. No model is loaded, moved, cast or switched to eval mode.
    ``domain_check(lat, lon, radius_km)`` must return a Boolean for the complete
    probe disc; it is optional and cannot be inferred from the model output.
    """
    encode: Callable
    name: str = 'custom'
    input_units: str = 'radians'
    coordinate_order: str = 'latlon'
    wrap_longitude: bool = False
    normalization: str = 'raw'
    feature_transform: Callable | None = None
    domain_check: Callable | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not callable(self.encode) or not isinstance(self.name, str) or not self.name:
            raise ValueError('A callable encoder and a nonempty name are required.')
        if self.input_units not in {'radians', 'degrees'}:
            raise ValueError('Input units must be radians or degrees.')
        if self.coordinate_order not in {'latlon', 'lonlat'}:
            raise ValueError('Coordinate order must be latlon or lonlat.')
        if self.normalization not in {'raw', 'unit'}:
            raise ValueError('Normalization must be raw or unit.')
        if not isinstance(self.wrap_longitude, (bool, np.bool_)):
            raise ValueError('Longitude wrapping must be Boolean.')
        for callback in [self.feature_transform, self.domain_check]:
            if callback is not None and not callable(callback):
                raise ValueError('Optional transforms and domain checks must be callable.')
        try:
            metadata = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        except (ValueError, TypeError) as error:
            raise ValueError('Metadata must contain finite JSON-compatible values.') from error
        object.__setattr__(self, 'metadata', metadata)

    def describe(self):
        """Return an independent record of the selected representation."""
        return dict(name=self.name, input_units=self.input_units,
                    coordinate_order=self.coordinate_order,
                    wrap_longitude=bool(self.wrap_longitude), normalization=self.normalization,
                    feature_transform_supplied=self.feature_transform is not None,
                    domain_check_supplied=self.domain_check is not None,
                    metadata=deepcopy(self.metadata))

    def validate_disc(self, latitude, longitude, radius_km):
        if self.domain_check is not None:
            accepted = self.domain_check(latitude, longitude, radius_km)
            if not isinstance(accepted, (bool, np.bool_)):
                raise ValueError('Domain check must return a Boolean for the entire probe disc.')
            if not accepted:
                raise ValueError('Encoder domain check rejected this probe disc.')

    def encode_locations(self, latitudes, longitudes):
        latitude, longitude = [np.asarray(v, dtype=float) for v in (latitudes, longitudes)]
        if (latitude.ndim != 1 or longitude.shape != latitude.shape
                or not np.isfinite(latitude).all() or not np.isfinite(longitude).all()
                or np.any(abs(latitude) > np.pi / 2)):
            raise ValueError('Finite aligned one-dimensional latitude/longitude arrays in radians are required.')
        if self.wrap_longitude:
            longitude = (longitude + np.pi) % (2 * np.pi) - np.pi
        coordinates = np.column_stack([latitude, longitude])
        if self.coordinate_order == 'lonlat':
            coordinates = coordinates[:, ::-1]
        if self.input_units == 'degrees':
            coordinates = np.degrees(coordinates)
        output = self.encode(coordinates)
        if self.feature_transform is not None:
            output = self.feature_transform(output)
        features = feature_matrix(output, len(latitude))
        if self.normalization == 'unit':
            # Scale first so norm computation cannot overflow on large vectors.
            scale = np.max(abs(features), axis=1, keepdims=True)
            if np.any(scale == 0):
                raise ValueError('Unit normalization requires nonzero feature vectors.')
            scaled = features / scale
            features = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
        return features

    def __call__(self, latitude, longitude):
        return self.encode_locations(np.array([latitude]), np.array([longitude]))[0]


def iter_feature_batches(encoder, latitudes, longitudes, batch_size):
    """Evaluate canonical positions with either an adapter or scalar callable."""
    dimension = None
    for start in range(0, len(latitudes), batch_size):
        lat, lon = latitudes[start:start+batch_size], longitudes[start:start+batch_size]
        if isinstance(encoder, EncoderAdapter):
            features = encoder.encode_locations(lat, lon)
        else:
            vectors = [np.array(encoder(a, b), copy=True) for a, b in zip(lat, lon)]
            if any(v.ndim != 1 or v.size == 0 for v in vectors):
                raise ValueError('Scalar encoder must return a nonempty one-dimensional feature vector.')
            try:
                features = feature_matrix(np.stack(vectors), len(lat))
            except ValueError as error:
                raise ValueError('Scalar encoder must return finite feature vectors of consistent dimension.') from error
        if dimension is not None and features.shape[1] != dimension:
            raise ValueError('Feature dimension changed between encoder batches.')
        dimension = features.shape[1]
        yield features
