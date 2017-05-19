"""Atomic structure.

This mudule contains helper functions for setting up nanotubes and
graphene nanoribbons."""

import warnings
from math import sqrt

import numpy as np

from ase.atoms import Atoms, string2symbols
from ase.data import covalent_radii
from ase.utils import gcd


def nanotube(n, m, length=1, bond=1.42, symbol='C', verbose=False):
    if n < m:
        m, n = n, m
        sign = -1
    else:
        sign = 1

    nk = 6000
    sq3 = sqrt(3.0)
    a = sq3 * bond
    l2 = n * n + m * m + n * m
    l = sqrt(l2)

    nd = gcd(n, m)
    if (n - m) % (3 * nd) == 0:
        ndr = 3 * nd
    else:
        ndr = nd

    nr = (2 * m + n) / ndr
    ns = -(2 * n + m) / ndr
    nn = 2 * l2 / ndr

    ichk = 0
    if nr == 0:
        n60 = 1
    else:
        n60 = nr * 4

    absn = abs(n60)
    nnp = []
    nnq = []
    for i in range(-absn, absn + 1):
        for j in range(-absn, absn + 1):
            j2 = nr * j - ns * i
            if j2 == 1:
                j1 = m * i - n * j
                if j1 > 0 and j1 < nn:
                    ichk += 1
                    nnp.append(i)
                    nnq.append(j)

    if ichk == 0:
        raise RuntimeError('not found p, q strange!!')
    if ichk >= 2:
        raise RuntimeError('more than 1 pair p, q strange!!')

    nnnp = nnp[0]
    nnnq = nnq[0]

    if verbose:
        print 'the symmetry vector is', nnnp, nnnq

    lp = nnnp * nnnp + nnnq * nnnq + nnnp * nnnq
    r = a * sqrt(lp)
    c = a * l
    t = sq3 * c / ndr

    if 2 * nn > nk:
        raise RuntimeError('parameter nk is too small!')

    rs = c / (2.0 * np.pi)

    if verbose:
        print 'radius=', rs, t

    q1 = np.arctan((sq3 * m) / (2 * n + m))
    q2 = np.arctan((sq3 * nnnq) / (2 * nnnp + nnnq))
    q3 = q1 - q2

    q4 = 2.0 * np.pi / nn
    q5 = bond * np.cos((np.pi / 6.0) - q1) / c * 2.0 * np.pi

    h1 = abs(t) / abs(np.sin(q3))
    h2 = bond * np.sin((np.pi / 6.0) - q1)

    ii = 0
    x, y, z = [], [], []
    for i in range(nn):
        x1, y1, z1 = 0, 0, 0

        k = np.floor(i * abs(r) / h1)
        x1 = rs * np.cos(i * q4)
        y1 = rs * np.sin(i * q4)
        z1 = (i * abs(r) - k * h1) * np.sin(q3)
        kk2 = abs(np.floor((z1 + 0.0001) / t))
        if z1 >= t - 0.0001:
            z1 -= t * kk2
        elif z1 < 0:
            z1 += t * kk2
        ii += 1

        x.append(x1)
        y.append(y1)
        z.append(z1)
        z3 = (i * abs(r) - k * h1) * np.sin(q3) - h2
        ii += 1

        if z3 >= 0 and z3 < t:
            x2 = rs * np.cos(i * q4 + q5)
            y2 = rs * np.sin(i * q4 + q5)
            z2 = (i * abs(r) - k * h1) * np.sin(q3) - h2
            x.append(x2)
            y.append(y2)
            z.append(z2)
        else:
            x2 = rs * np.cos(i * q4 + q5)
            y2 = rs * np.sin(i * q4 + q5)
            z2 = (i * abs(r) - (k + 1) * h1) * np.sin(q3) - h2
            kk = abs(np.floor(z2 / t))
            if z2 >= t - 0.0001:
                z2 -= t * kk
            elif z2 < 0:
                z2 += t * kk
            x.append(x2)
            y.append(y2)
            z.append(z2)

    ntotal = 2 * nn
    X = []
    for i in range(ntotal):
        X.append([x[i], y[i], sign * z[i]])

    if length > 1:
        xx = X[:]
        for mnp in range(2, length + 1):
            for i in range(len(xx)):
                X.append(xx[i][:2] + [xx[i][2] + (mnp - 1) * t])

    TransVec = t
    NumAtom = ntotal * length
    Diameter = rs * 2
    ChiralAngle = np.arctan((sq3 * n) / (2 * m + n)) / (np.pi * 180)

    cell = [Diameter * 2, Diameter * 2, length * t]
    atoms = Atoms(symbol + str(NumAtom), positions=X, cell=cell,
                  pbc=[False, False, True])
    atoms.center()
    if verbose:
        print 'translation vector =', TransVec
        print 'diameter = ', Diameter
        print 'chiral angle = ', ChiralAngle
    return atoms


def graphene_nanoribbon(n, m, type='zigzag', saturated=False, C_H=1.09,
                        C_C=1.42, vacuum=2.5, magnetic=None, initial_mag=1.12,
                        sheet=False, main_element='C', saturate_element='H',
                        vacc=None):
    """Create a graphene nanoribbon.

    Creates a graphene nanoribbon in the x-z plane, with the nanoribbon
    running along the z axis.

    Parameters:

    n: int
        The width of the nanoribbon.
    m: int
        The length of the nanoribbon.
    type: str
        The orientation of the ribbon.  Must be either 'zigzag'
        or 'armchair'.
    saturated: bool
        If true, hydrogen atoms are placed along the edge.
    C_H: float
        Carbon-hydrogen bond length.  Default: 1.09 Angstrom.
    C_C: float
        Carbon-carbon bond length.  Default: 1.42 Angstrom.
    vacuum: float
        Amount of vacuum added to both sides.  Default 2.5 Angstrom.
    magnetic: bool
        Make the edges magnetic.
    initial_mag: float
        Magnitude of magnetic moment if magnetic=True.
    sheet: bool
        If true, make an infinite sheet instead of a ribbon.
    """

    if vacc is not None:
        warnings.warn('Use vacuum=%f' % (0.5 * vacc))
        vacuum = 0.5 * vacc

    assert vacuum > 0
    b = sqrt(3) * C_C / 4
    arm_unit = Atoms(main_element + '4',
                     pbc=(1, 0, 1),
                     cell=[4 * b, 2 * vacuum, 3 * C_C])
    arm_unit.positions = [[0, 0, 0],
                          [b * 2, 0, C_C / 2.],
                          [b * 2, 0, 3 * C_C / 2.],
                          [0, 0, 2 * C_C]]
    zz_unit = Atoms(main_element + '2',
                    pbc=(1, 0, 1),
                    cell=[3 * C_C / 2.0, 2 * vacuum, b * 4])
    zz_unit.positions = [[0, 0, 0],
                         [C_C / 2.0, 0, b * 2]]
    atoms = Atoms()
    if sheet:
        vacuum2 = 0.0
    else:
        vacuum2 = vacuum
    if type == 'zigzag':
        edge_index0 = np.arange(m) * 2 + 1
        edge_index1 = (n - 1) * m * 2 + np.arange(m) * 2
        if magnetic:
            mms = np.zeros(m * n * 2)
            for i in edge_index0:
                mms[i] = initial_mag
            for i in edge_index1:
                mms[i] = -initial_mag

        for i in range(n):
            layer = zz_unit.repeat((1, 1, m))
            layer.positions[:, 0] -= 3 * C_C / 2 * i
            if i % 2 == 1:
                layer.positions[:, 2] += 2 * b
                layer[-1].position[2] -= b * 4 * m
            atoms += layer
        if magnetic:
            atoms.set_initial_magnetic_moments(mms)
        if saturated:
            H_atoms0 = Atoms(saturate_element + str(m))
            H_atoms0.positions = atoms[edge_index0].positions
            H_atoms0.positions[:, 0] += C_H
            H_atoms1 = Atoms(saturate_element + str(m))
            H_atoms1.positions = atoms[edge_index1].positions
            H_atoms1.positions[:, 0] -= C_H
            atoms += H_atoms0 + H_atoms1
        atoms.cell = [n * 3 * C_C / 2 + 2 * vacuum2, 2 * vacuum, m * 4 * b]

    elif type == 'armchair':
        for i in range(n):
            layer = arm_unit.repeat((1, 1, m))
            layer.positions[:, 0] -= 4 * b * i
            atoms += layer
        if saturated:
            arm_right_saturation = Atoms(saturate_element + '2', pbc=(1, 0, 1),
                                         cell=[4 * b, 2 * vacuum, 3 * C_C])
            arm_right_saturation.positions = [
                [- sqrt(3) / 2 * C_H, 0, C_H * 0.5], 
                [- sqrt(3) / 2 * C_H, 0, 2 * C_C - C_H * 0.5]]
            arm_left_saturation = Atoms(saturate_element + '2', pbc=(1, 0, 1),
                                        cell=[4 * b, 2 * vacuum, 3 * C_C])
            arm_left_saturation.positions = [
                [b * 2 + sqrt(3) / 2 * C_H, 0, C_C / 2 - C_H * 0.5], 
                [b * 2 + sqrt(3) / 2 * C_H, 0, 3 * C_C / 2.0 + C_H * 0.5]]
            arm_right_saturation.positions[:, 0] -= 4 * b * (n - 1)
            atoms += arm_right_saturation.repeat((1, 1, m))
            atoms += arm_left_saturation.repeat((1, 1, m))

        atoms.cell = [b * 4 * n + 2 * vacuum2, 2 * vacuum, 3 * C_C * m]

    atoms.center()
    atoms.set_pbc([sheet, False, True])
    return atoms


def molecule(name, data=None, **kwargs):
    """Create formula base on data. If data is None assume G2 set.
    kwargs currently not used.  """
    if name in extra:
        kwargs.update(extra[name])
        return Atoms(**kwargs)
    
    if data is None:
        from ase.data.g2 import data
    if name not in data.keys():
        raise NotImplementedError('%s not in data.' % (name))
    args = data[name].copy()
    # accept only the following Atoms constructor arguments
    # XXX: should we accept all Atoms arguments?
    for k in args.keys():
        if k not in [
            'symbols', 'positions', 'numbers',
            'tags', 'masses',
            'magmoms', 'charges',
            'info',
            ]:
            args.pop(k)
    # kwargs overwrites data
    args.update(kwargs)
    return Atoms(**args)


extra = {
    'Be2': {
        'symbols': 'BeBe',
        'positions': [[0, 0, 1.0106],
                      [0, 0, -1.0106]]},
    'C7NH5': {
        'symbols': 'C7NH5',
        'positions': [[-1.593581, -1.142601, 0.],
                      [-2.235542, 0.095555, 0.],
                      [-0.204885, -1.210726, 0.],
                      [0.549645, -0.025355, 0.],
                      [1.976332, -0.085321, 0.],
                      [-0.099258, 1.220706, 0.],
                      [-1.488628, 1.273345, 0.],
                      [3.136871, -0.128138, 0.],
                      [-2.177996, -2.060896, 0.],
                      [-3.323594, 0.141242, 0.],
                      [0.301694, -2.173705, 0.],
                      [0.488716, 2.136782, 0.],
                      [-1.987765, 2.240495, 0.]]},
    'BDA': {
        # 1,4-Benzodiamine
        # aka p-Aminoaniline; p-Benzenediamine; p-Diaminobenzene;
        #     p-Phenylenediamine; Paraphenylen-diamine
        # PBE-gpaw relaxed
        'symbols': 'C6H4N2H4',
        'positions': [[0.004212, 1.406347, 0.061073],
                      [1.193490, 0.687096, 0.029481],
                      [1.190824, -0.690400, -0.028344],
                      [0.000295, -1.406191, -0.059503],
                      [-1.186974, -0.685668, -0.045413],
                      [-1.185376, 0.690203, 0.009452],
                      [2.147124, 1.219997, 0.064477],
                      [2.141593, -1.227477, -0.054266],
                      [-2.138408, -1.222814, -0.095050],
                      [-2.137740, 1.226930, 0.023036],
                      [-0.006314, 2.776024, 0.186278],
                      [-0.007340, -2.777839, -0.159936],
                      [0.844710, -3.256543, 0.110098],
                      [-0.854965, -3.253324, 0.130125],
                      [0.845826, 3.267270, -0.055549],
                      [-0.854666, 3.254654, -0.092676]]},
    'biphenyl': {
        # PBE-gpaw relaxed
        'symbols': 'C6H5C6H5',
        'positions': [[-0.74081, -0.00000, -0.00003],
                      [-1.46261, -1.20370, -0.00993],
                      [-2.85531, -1.20350, -0.00663],
                      [-3.55761, -0.00000, -0.00003],
                      [-2.85531, 1.20350, 0.00667],
                      [-1.46261, 1.20370, 0.00997],
                      [-0.92071, -2.14850, 0.00967],
                      [-3.38981, -2.15110, -0.00083],
                      [-4.64571, -0.00000, -0.00003],
                      [-3.38981, 2.15110, 0.00077],
                      [-0.92071, 2.14850, -0.00963],
                      [3.55849, -0.00000, -0.00003],
                      [2.85509, -0.86640, -0.83553],
                      [1.46289, -0.87000, -0.83153],
                      [0.73969, -0.00000, -0.00003],
                      [1.46289, 0.87000, 0.83157],
                      [2.85509, 0.86640, 0.83547],
                      [4.64659, -0.00000, -0.00003],
                      [3.39189, -1.53770, -1.50253],
                      [0.91869, -1.53310, -1.50263],
                      [0.91869, 1.53310, 1.50267],
                      [3.39189, 1.53770, 1.50257]]},
    'C60': {
        # Buckminsterfullerene, I*h symm.
        # The Buckyball has two degrees of freedom, the C-C bond, and the
        # C=C bond. This is an LDA-gpaw relaxed structure with bond lengths
        # 1.437 and 1.385.
        # Experimentally, the two bond lengths are 1.45 and 1.40 Angstrom.
        'symbols': 'C60',
        'positions': [[2.2101953, 0.5866631, 2.6669504],
                      [3.1076393, 0.1577008, 1.6300286],
                      [1.3284430, -0.3158939, 3.2363232],
                      [3.0908709, -1.1585005, 1.2014240],
                      [3.1879245, -1.4574599, -0.1997005],
                      [3.2214623, 1.2230966, 0.6739440],
                      [3.3161210, 0.9351586, -0.6765151],
                      [3.2984981, -0.4301142, -1.1204138],
                      [-0.4480842, 1.3591484, 3.2081020],
                      [0.4672056, 2.2949830, 2.6175264],
                      [-0.0256575, 0.0764219, 3.5086259],
                      [1.7727917, 1.9176584, 2.3529691],
                      [2.3954623, 2.3095689, 1.1189539],
                      [-0.2610195, 3.0820935, 1.6623117],
                      [0.3407726, 3.4592388, 0.4745968],
                      [1.6951171, 3.0692446, 0.1976623],
                      [-2.1258394, -0.8458853, 2.6700963],
                      [-2.5620990, 0.4855202, 2.3531715],
                      [-0.8781521, -1.0461985, 3.2367302],
                      [-1.7415096, 1.5679963, 2.6197333],
                      [-1.6262468, 2.6357030, 1.6641811],
                      [-3.2984810, 0.4301871, 1.1204208],
                      [-3.1879469, 1.4573895, 0.1996030],
                      [-2.3360261, 2.5813627, 0.4760912],
                      [-0.5005210, -2.9797771, 1.7940308],
                      [-1.7944338, -2.7729087, 1.2047891],
                      [-0.0514245, -2.1328841, 2.7938830],
                      [-2.5891471, -1.7225828, 1.6329715],
                      [-3.3160705, -0.9350636, 0.6765268],
                      [-1.6951919, -3.0692581, -0.1976564],
                      [-2.3954901, -2.3096853, -1.1189862],
                      [-3.2214182, -1.2231835, -0.6739581],
                      [2.1758234, -2.0946263, 1.7922529],
                      [1.7118619, -2.9749681, 0.7557198],
                      [1.3130656, -1.6829416, 2.7943892],
                      [0.3959024, -3.4051395, 0.7557638],
                      [-0.3408219, -3.4591883, -0.4745610],
                      [2.3360057, -2.5814499, -0.4761050],
                      [1.6263757, -2.6357349, -1.6642309],
                      [0.2611352, -3.0821271, -1.6622618],
                      [-2.2100844, -0.5868636, -2.6670300],
                      [-1.7726970, -1.9178969, -2.3530466],
                      [-0.4670723, -2.2950509, -2.6175105],
                      [-1.3283500, 0.3157683, -3.2362375],
                      [-2.1759882, 2.0945383, -1.7923294],
                      [-3.0909663, 1.1583472, -1.2015749],
                      [-3.1076090, -0.1578453, -1.6301627],
                      [-1.3131365, 1.6828292, -2.7943639],
                      [0.5003224, 2.9799637, -1.7940203],
                      [-0.3961148, 3.4052817, -0.7557272],
                      [-1.7120629, 2.9749122, -0.7557988],
                      [0.0512824, 2.1329478, -2.7937450],
                      [2.1258630, 0.8460809, -2.6700534],
                      [2.5891853, 1.7227742, -1.6329562],
                      [1.7943010, 2.7730684, -1.2048262],
                      [0.8781323, 1.0463514, -3.2365313],
                      [0.4482452, -1.3591061, -3.2080510],
                      [1.7416948, -1.5679557, -2.6197714],
                      [2.5621724, -0.4853529, -2.3532026],
                      [0.0257904, -0.0763567, -3.5084446]]}}


def bulk(name, crystalstructure, a=None, c=None, covera=None,
         orthorhombic=False, cubic=False):
    """Helper function for creating bulk systems.

    name: str
        Chemical symbol or symbols as in 'MgO' or 'NaCl'.
    crystalstructure: str
        Must be one of sc, fcc, bcc, hcp, diamond, zincblende or
        rocksalt.
    a: float
        Lattice constant.
    c: float
        Lattice constant.
    covera: float
        c/a raitio used for hcp.  Defaults to ideal ratio.
    orthorhombic: bool
        Construct orthorhombic unit cell instead of primitive cell
        which is the default.
    cubic: bool
        Construct cubic unit cell.
    """

    warnings.warn('This function is deprecated.  Use the ' +
                  'ase.lattice.bulk() function instead.')

    if a is not None:
        a = float(a)
    if c is not None:
        c = float(c)

    if covera is not None and  c is not None:
        raise ValueError("Don't specify both c and c/a!")

    if covera is None and c is None:
        covera = sqrt(8.0 / 3.0)

    if a is None:
        a = estimate_lattice_constant(name, crystalstructure, covera)

    if covera is None and c is not None:
        covera = c / a

    x = crystalstructure.lower()

    if orthorhombic and x != 'sc':
        return _orthorhombic_bulk(name, x, a, covera)

    if cubic and x == 'bcc':
        return _orthorhombic_bulk(name, x, a, covera)

    if cubic and x != 'sc':
        return _cubic_bulk(name, x, a)

    if x == 'sc':
        atoms = Atoms(name, cell=(a, a, a), pbc=True)
    elif x == 'fcc':
        b = a / 2
        atoms = Atoms(name, cell=[(0, b, b), (b, 0, b), (b, b, 0)], pbc=True)
    elif x == 'bcc':
        b = a / 2
        atoms = Atoms(name, cell=[(-b, b, b), (b, -b, b), (b, b, -b)],
                      pbc=True)
    elif x == 'hcp':
        atoms = Atoms(2 * name,
                      scaled_positions=[(0, 0, 0),
                                        (1.0 / 3.0, 1.0 / 3.0, 0.5)],
                      cell=[(a, 0, 0),
                            (a / 2, a * sqrt(3) / 2, 0),
                            (0, 0, covera * a)],
                      pbc=True)
    elif x == 'diamond':
        atoms = bulk(2 * name, 'zincblende', a)
    elif x == 'zincblende':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a)
        atoms.positions[1] += a / 4
    elif x == 'rocksalt':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a)
        atoms.positions[1, 0] += a / 2
    else:
        raise ValueError('Unknown crystal structure: ' + crystalstructure)

    return atoms


def estimate_lattice_constant(name, crystalstructure, covera):
    atoms = bulk(name, crystalstructure, 1.0, covera)
    v0 = atoms.get_volume()
    v = 0.0
    for Z in atoms.get_atomic_numbers():
        r = covalent_radii[Z]
        v += 4 * np.pi / 3 * r**3 * 1.5
    return (v / v0)**(1.0 / 3)


def _orthorhombic_bulk(name, x, a, covera=None):
    if x == 'fcc':
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif x == 'bcc':
        atoms = Atoms(2 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif x == 'hcp':
        atoms = Atoms(4 * name,
                      cell=(a, a * sqrt(3), covera * a),
                      scaled_positions=[(0, 0, 0),
                                        (0.5, 0.5, 0),
                                        (0.5, 1.0 / 6.0, 0.5),
                                        (0, 2.0 / 3.0, 0.5)],
                      pbc=True)
    elif x == 'diamond':
        atoms = _orthorhombic_bulk(2 * name, 'zincblende', a)
    elif x == 'zincblende':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0, 0.25),
                                        (0.5, 0.5, 0.5), (0, 0.5, 0.75)])
    elif x == 'rocksalt':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0),
                                        (0.5, 0.5, 0.5), (0, 0, 0.5)])
    else:
        raise RuntimeError

    return atoms


def _cubic_bulk(name, x, a):
    if x == 'fcc':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0, 0.5, 0.5),
                                        (0.5, 0, 0.5), (0.5, 0.5, 0)])
    elif x == 'diamond':
        atoms = _cubic_bulk(2 * name, 'zincblende', a)
    elif x == 'zincblende':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25),
                                        (0, 0.5, 0.5), (0.25, 0.75, 0.75),
                                        (0.5, 0, 0.5), (0.75, 0.25, 0.75),
                                        (0.5, 0.5, 0), (0.75, 0.75, 0.25)])
    elif x == 'rocksalt':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0, 0),
                                        (0, 0.5, 0.5), (0.5, 0.5, 0.5),
                                        (0.5, 0, 0.5), (0, 0, 0.5),
                                        (0.5, 0.5, 0), (0, 0.5, 0)])
    else:
        raise RuntimeError

    return atoms
