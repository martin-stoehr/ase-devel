import numpy as np
from ase import Atoms
from ase.formula import Formula


assert Atoms('MoS2').get_chemical_formula() == 'MoS2'
assert Atoms('SnO2').get_chemical_formula(mode='metal') == 'SnO2'


for sym in ['', 'Pu', 'Pu2', 'U2Pu2', 'U2((Pu2)2H)']:
    for mode in ['all', 'reduce', 'hill', 'metal']:
        for empirical in [False, True]:
            if empirical and mode in ['all', 'reduce']:
                continue
            atoms = Atoms(sym)
            formula = atoms.get_chemical_formula(mode=mode,
                                                 empirical=empirical)
            atoms2 = Atoms(formula)
            print(repr(sym), '->', repr(formula))
            n1 = np.sort(atoms.numbers)
            n2 = np.sort(atoms2.numbers)
            if empirical and len(atoms) > 0:
                reduction = len(n1) // len(n2)
                n2 = np.repeat(n2, reduction)
            assert (n1 == n2).all()


for x in ['H2O', '10H2O', '2(CuO2(H2O)2)10', 'Cu20+H2', 'H' * 15,
          'AuBC2', '']:
    f = Formula(x)
    y = str(f)
    assert y == x
    print(f.count(), '{:latex}'.format(f))
    a, b = divmod(f, 'H2O')
    assert a * Formula('H2O') + b == f
    assert f != 117
