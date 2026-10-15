"""Nonstationary Matérn Gaussian process and matched multistart optimisation."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from pyproj import Proj
WINDOWS = {'CONUS': (25, 50, -125, -65), 'Europe': (35, 60, -10, 30), 'Australia': (-40, -12, 113, 154), 'TropSAm': (-15, 5, -75, -45)}
TARGETS = ['ELEVATION', 'TREE_COVER', 'SOIL_MOISTURE']
VARIANTS = ['iso', 'embedding', 'SatCLIP-L40', 'shuffled', 'oracle', 'SINR', 'Climplicit', 'GeoCLIP-1km', 'spatial+embedding', 'tensor+embedding']
T = lambda x: torch.as_tensor(x, dtype=torch.float64)

class GaussianProcess:

    def __init__(self, x, y, a, b, e, kind):
        if kind not in VARIANTS:
            raise ValueError(f'Unknown Gaussian-process variant: {kind}')
        x, y, a, b = [np.asarray(v, dtype=float) for v in (x, y, a, b)]
        if (y.ndim != 1 or y.size < 2 or x.shape != (y.size, 2)
                or a.shape != y.shape or b.shape != y.shape
                or not all(np.isfinite(v).all() for v in (x, y, a, b)) or np.any(a <= 0)):
            raise ValueError('Finite aligned coordinates, targets, positive anisotropy and bearings are required.')
        if e is not None:
            e = np.asarray(e, dtype=float)
            if e.ndim != 2 or e.shape[0] != y.size or e.shape[1] == 0 or not np.isfinite(e).all():
                raise ValueError('Embedding features must be finite and aligned with training sites.')
        if 'embedding' in kind and e is None:
            raise ValueError('This variant requires embedding features.')
        self.kind = kind
        self.x, self.a, self.b = map(T, (x, a, b))
        self.ymu, self.ysd = (float(y.mean()), float(y.std() + 1e-09))
        self.y = T((y - self.ymu) / self.ysd)
        self.eye = torch.eye(len(y), dtype=torch.float64)
        self.dx = self.x[:, None, 0] - self.x[None, :, 0]
        self.dy = self.x[:, None, 1] - self.x[None, :, 1]
        self.e = e
        self.ed = T(cdist(e, e)) if e is not None else None
        self.is_combo = kind in ['spatial+embedding', 'tensor+embedding']
        self.tensor = kind not in ['iso', 'embedding', 'spatial+embedding']
        self.bounds = [(np.log(20), np.log(5000)), (-4, 4), (-8, 2)]
        if kind == 'embedding':
            self.bounds[0] = (None, None)
        elif self.tensor:
            self.bounds.append((0.0, 3.0))
        if self.is_combo:
            self.bounds += [(None, None), (-4, 4)]

    def kernel(self, p, dx, dy, a1, b1, a2, b2, ed):

        def matern(d, length, variance):
            z = np.sqrt(3) * d / torch.exp(length)
            return torch.exp(variance) * (1 + z) * torch.exp(-z)
        if self.kind == 'embedding':
            return matern(ed, p[0], p[1])
        if self.tensor:
            beta = p[3]

            def entries(a, b):
                q = torch.exp(-beta * torch.log(a))
                r = 1 / q
                sn, cs = (torch.sin(b), torch.cos(b))
                return (q * sn ** 2 + r * cs ** 2, (q - r) * sn * cs, q * cs ** 2 + r * sn ** 2)
            u, v, w = entries(a1, b1)
            uu, vv, ww = entries(a2, b2)
            aa = (u[:, None] + uu[None, :]) / 2
            bb = (v[:, None] + vv[None, :]) / 2
            cc = (w[:, None] + ww[None, :]) / 2
            det = aa * cc - bb ** 2
            q = (cc * dx ** 2 - 2 * bb * dx * dy + aa * dy ** 2) / det
            dist = torch.sqrt(torch.clamp(q, min=1e-28))
            k = matern(dist, p[0], p[1]) / torch.sqrt(det)
        else:
            k = matern(torch.sqrt(torch.clamp(dx ** 2 + dy ** 2, min=1e-28)), p[0], p[1])
        if self.is_combo:
            k = k + matern(ed, p[-2], p[-1])
        return k

    def objective(self, values):
        if np.shape(values) != (len(self.bounds),):
            raise ValueError('Parameter count does not match the Gaussian-process variant.')
        if not np.isfinite(values).all():
            return (1e+20, np.zeros(len(values)))
        p = T(values).requires_grad_(True)
        k = self.kernel(p, self.dx, self.dy, self.a, self.b, self.a, self.b, self.ed)
        k = k + (torch.exp(p[2]) + 1e-06) * self.eye
        try:
            l = torch.linalg.cholesky(k)
        except RuntimeError:
            return (1e+20, np.zeros(len(values)))
        alpha = torch.cholesky_solve(self.y[:, None], l)[:, 0]
        loss = 0.5 * self.y @ alpha + torch.log(l.diagonal()).sum() + len(self.y) / 2 * np.log(2 * np.pi)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(p.grad).all():
            return (1e+20, np.zeros(len(values)))
        return (float(loss.detach()), p.grad.detach().numpy().copy())

    def starts(self, iso=None, emb=None, tensor=None, spatial=None):
        emed = np.median(cdist(self.e[:200], self.e[:200])) if self.e is not None else 1.0
        base = np.array([np.log(300), 0.0, -2.0])
        if self.kind == 'embedding':
            return [np.array([np.log(max(emed * c, 1e-09)), 0.0, -2.0]) for c in (0.3, 1.0, 3.0)]
        if not self.tensor and (not self.is_combo):
            return [np.array([np.log(r), 0.0, -2.0]) for r in (80, 300, 1000)]
        iso = np.array(iso if iso is not None else base)
        if not self.is_combo:
            return [np.r_[iso, 0.0], np.r_[base, 0.5], np.r_[np.log(1000), 0.0, -2.0, 1.5]]
        ep = np.array(emb)
        if self.tensor:
            return [np.r_[tensor, np.log(max(emed, 1e-09)), -4.0], np.insert(np.asarray(spatial), 3, 0.0), np.r_[base, 1.5, ep[0], ep[1]]]
        return [np.r_[iso, ep[0], -4.0], np.r_[iso, ep[0], ep[1]], np.r_[np.log(1000), 0.0, -2.0, ep[0], ep[1]]]

    def fit(self, starts):
        runs = []
        for p0 in starts:
            initial = self.objective(p0)[0]
            opt = minimize(self.objective, p0, jac=True, method='L-BFGS-B', bounds=self.bounds, options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-05, 'maxls': 40})
            accept = np.isfinite(opt.fun) and np.isfinite(opt.x).all() and opt.fun <= initial
            p = opt.x if accept else p0
            nll = float(opt.fun if accept else initial)
            runs.append({'p': p.tolist(), 'nll': nll, 'initial_nll': initial, 'success': bool(opt.success), 'message': str(opt.message), 'iterations': int(opt.nit), 'start': np.asarray(p0).tolist()})
        if not runs or all(r['nll'] >= 1e+20 for r in runs):
            raise RuntimeError('No numerically valid Gaussian-process fit was found.')
        best = min(runs, key=lambda x: x['nll'])
        return {'p': best['p'], 'nll': best['nll'], 'starts': runs, 'nll_spread': max((r['nll'] for r in runs)) - best['nll']}

    def predict(self, fit, xte, yte, ate, bte, ete):
        xte, yte, ate, bte = [np.asarray(v, dtype=float) for v in (xte, yte, ate, bte)]
        if (yte.ndim != 1 or yte.size == 0 or xte.shape != (yte.size, 2)
                or ate.shape != yte.shape or bte.shape != yte.shape
                or not all(np.isfinite(v).all() for v in (xte, yte, ate, bte)) or np.any(ate <= 0)):
            raise ValueError('Finite aligned test coordinates, targets, positive anisotropy and bearings are required.')
        if self.kind == 'embedding' or self.is_combo:
            ete = np.asarray(ete, dtype=float)
            if ete.shape != (yte.size, self.e.shape[1]) or not np.isfinite(ete).all():
                raise ValueError('Test embedding features must match training feature dimensions.')
        if np.shape(fit['p']) != (len(self.bounds),) or not np.isfinite(fit['p']).all():
            raise ValueError('Prediction requires a finite parameter vector for this variant.')
        with torch.no_grad():
            p = T(fit['p'])
            x = T(xte)
            k = self.kernel(p, self.dx, self.dy, self.a, self.b, self.a, self.b, self.ed)
            l = torch.linalg.cholesky(k + (torch.exp(p[2]) + 1e-06) * self.eye)
            dx = x[:, None, 0] - self.x[None, :, 0]
            dy = x[:, None, 1] - self.x[None, :, 1]
            ed = T(cdist(ete, self.e)) if ete is not None else None
            ks = self.kernel(p, dx, dy, T(ate), T(bte), self.a, self.b, ed)
            mu = ks @ torch.cholesky_solve(self.y[:, None], l)[:, 0]
            v = torch.linalg.solve_triangular(l, ks.T, upper=False)
            kss = torch.exp(p[1]) + (torch.exp(p[-1]) if self.is_combo else 0.0)
            var = torch.clamp(kss - (v * v).sum(0), min=1e-09) + torch.exp(p[2])
            yt = T((yte - self.ymu) / self.ysd)
            fit['rmse'] = float(torch.sqrt(torch.mean((mu - yt) ** 2)))
            fit['nlpd'] = float(torch.mean(0.5 * torch.log(2 * np.pi * var) + 0.5 * (mu - yt) ** 2 / var))
            fit['beta'] = float(p[3]) if self.tensor else None
        return fit
