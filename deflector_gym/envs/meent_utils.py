
from typing import Tuple
import random
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import meent
from meent.on_torch.modeler.modeling import find_nk_index, read_material_table

import torch

def get_field(
        pattern_input,
        wavelength=900,
        deflected_angle=50,
        fourier_order=40,
        field_res=(256, 1, 32)  # (100, 1, 20)
):
    period = [abs(wavelength / np.sin(deflected_angle / 180 * np.pi))]
    n_ridge = 'p_si__real'
    n_groove = 1
    wavelength = np.array([wavelength])
    grating_type = 0
    thickness = [325] * 8

    if type(n_ridge) is str:
        mat_table = read_material_table()
        n_ridge = find_nk_index(n_ridge, mat_table, wavelength)
    ucell = np.array([[pattern_input]])
    ucell = (ucell + 1) / 2
    ucell = ucell * (n_ridge - n_groove) + n_groove
    ucell_new = np.ones((len(thickness), 1, ucell.shape[-1]))
    ucell_new[0:2] = 1.45
    ucell_new[2] = ucell

    mee = meent.call_mee(
        mode=0, wavelength=wavelength, period=period, grating_type=0, n_I=1.45, n_II=1.,
        theta=0, phi=0, psi=0, fourier_order=fourier_order, pol=1,
        thickness=thickness,
        ucell=ucell_new
    )
    # Calculate field distribution: OLD
    de_ri, de_ti, field_cell = mee.conv_solve_field(
        res_x=field_res[0], res_y=field_res[1], res_z=field_res[2],
    )
    if grating_type == 0:
        center = de_ti.shape[0] // 2
        de_ri_cut = de_ri[center - 1:center + 2]
        de_ti_cut = de_ti[center - 1:center + 2]
    else:
        x_c, y_c = np.array(de_ti.shape) // 2
        de_ri_cut = de_ri[x_c - 1:x_c + 2, y_c - 1:y_c + 2]
        de_ti_cut = de_ti[x_c - 1:x_c + 2, y_c - 1:y_c + 2]

    field_ex = np.flipud(field_cell[:, 0, :, 1])

    return de_ti_cut, field_ex


def get_efficiency(
        pattern_input,
        wavelength=900,
        deflected_angle=50,
        fourier_order=40
):

    period = [abs(wavelength / np.sin(deflected_angle / 180 * np.pi))]
    n_ridge = 'p_si__real'
    n_groove = 1
    wavelength = torch.tensor([wavelength])
    grating_type = 0
    thickness = [325]

    if type(n_ridge) is str:
        mat_table = read_material_table()
        n_ridge = find_nk_index(n_ridge, mat_table, wavelength)
    ucell = torch.tensor(np.array([[pattern_input]]))
    ucell = (ucell + 1) / 2
    ucell = ucell * (n_ridge - n_groove) + n_groove

    mee = meent.call_mee(
        backend=2, wavelength=wavelength, period=period, grating_type=0, n_I=1.45, n_II=1.,
        theta=0, phi=0, psi=0, fourier_order=fourier_order, pol=1,
        thickness=thickness,
        ucell=ucell
    )
    de_ri, de_ti = mee.conv_solve()
    rayleigh_r = mee.rayleigh_r
    rayleigh_t = mee.rayleigh_t

    # diffraction efficiency
    if grating_type == 0:
        center = de_ti.shape[0] // 2
        de_ri_cut = de_ri[center - 1:center + 2]
        de_ti_cut = de_ti[center - 1:center + 2]
        de_ti_interest = de_ti[center+1]

    else:
        x_c, y_c = np.array(de_ti.shape) // 2
        de_ri_cut = de_ri[x_c - 1:x_c + 2, y_c - 1:y_c + 2]
        de_ti_cut = de_ti[x_c - 1:x_c + 2, y_c - 1:y_c + 2]
        de_ti_interest = de_ti[x_c+1, y_c]

    # data for neural operator experiment
    # W, V, q = mee.neuralop_data[0]
    # if grating_type == 0:
    #     [W], [V], [q] = W, V, q

    # elif grating_type == 1:
    #     [W_1, W_2], [V_11, V_12, V_21, V_22], [q_1, q_2] = W, V, q

    # elif grating_type == 2:
    #     [W_11, W_12, W_21, W_22], [V_11, V_12, V_21, V_22], [q] = W, V, q

    # else:
    #     raise ValueError

    return de_ti_interest