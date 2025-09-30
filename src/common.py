# src/common.py
import yaml
import numpy as np
import healpy as hp
import pymaster as nmt

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_maps(paths_cfg):
    T = hp.read_map(paths_cfg["T_map"], verbose=False)
    T_mask = hp.read_map(paths_cfg["T_mask"], verbose=False)
    phi = hp.read_map(paths_cfg["phi_map"], verbose=False)
    phi_mask = hp.read_map(paths_cfg["phi_mask"], verbose=False)
    return T, T_mask, phi, phi_mask

def build_bins(nside, bins_cfg):
    lmin = int(bins_cfg.get("lmin", 8))
    lmax = int(bins_cfg.get("lmax", 100))
    nlb  = int(bins_cfg.get("nlb", 10))
    return nmt.NmtBin.from_nside_linear(nside=nside, nlb=nlb, lmin=lmin, lmax=lmax)

def apodize(mask, aporad_deg=1.0):
    return nmt.mask_apodization(mask.astype(float), aporad_deg, apotype="C1")

def lowell_filter(alm, lmax_keep):
    lmax = hp.Alm.getlmax(alm.size)
    fl = np.zeros(lmax+1)
    fl[:lmax_keep+1] = 1.0
    return hp.almxfl(alm, fl)

def map_lowell(T_map, lmax_keep, nside):
    alm = hp.map2alm(T_map, lmax=3*nside-1, iter=0)
    alm_f = lowell_filter(alm, lmax_keep=lmax_keep)
    return hp.alm2map(alm_f, nside=nside, verbose=False)

def compute_cross_cls(field_a, field_b, bins):
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(field_a, field_b, bins)
    cl_coup = nmt.compute_coupled_cell(field_a, field_b)
    cl_dec  = w.decouple_cell(cl_coup)
    ells = bins.get_effective_ells()
    return ells, cl_dec[0]
