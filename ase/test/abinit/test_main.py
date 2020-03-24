import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.calculators.abinit import Abinit
from ase.units import Hartree
from ase.utils import workdir


required_quantities = {'eigenvalues',
                       'fermilevel',
                       'version',
                       'forces',
                       'energy',
                       'free_energy',
                       'stress',
                       'ibz_kpoints',
                       'kpoint_weights'}


def run(atoms, name):
    dirname = 'test-abinit/{}'.format(name)
    with workdir(dirname, mkdir=True):
        header = 'test {} in {}'.format(name, dirname)
        print()
        print(header)
        print('=' * len(header))
        print('input:', atoms.calc.parameters)
        atoms.get_potential_energy()
        atoms.get_forces()
        print(sorted(atoms.calc.results))
        for key, value in atoms.calc.results.items():
            if isinstance(value, np.ndarray):
                print(key, value.shape, value.dtype)
            else:
                print(key, value)

        for name in required_quantities:
            assert name in atoms.calc.results

    return atoms.calc.results


def abinit(**kwargs):
    kw = dict(ecut=150,
              chksymbreak=0,
              toldfe=1e-3)
    kw.update(kwargs)
    return Abinit(**kw)


def test_si():
    atoms = bulk('Si')
    atoms.calc = abinit(nbands=4 * len(atoms))
    run(atoms, 'bulk-si')


def run_au(pps, **kwargs):
    atoms = bulk('Au')
    atoms.calc = abinit(nbands=10 * len(atoms), pps=pps,
                        tsmear=0.1,
                        occopt=3,
                        kpts=[2, 2, 2],
                        **kwargs)
    run(atoms, 'bulk-au-{}'.format(pps))


def _test_fe(name, **kwargs):
    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([1])
    calc = abinit(nbands=8,
                  kpts=[2, 2, 2], **kwargs)
    atoms.calc = calc
    run(atoms, name)
    # Grrr we want to test magmoms but the caching doesn't work.
    # We should fix this.
    #magmom = atoms.get_magnetic_moment()
    #print('magmom', magmom)
    # The calculator base class thinks it is smart, returning 0 magmom
    # automagically when not otherwise given.  This means we get bogus zeros
    # if/when we didn't parse the magmoms.  This happens when the magmoms
    # are fixed.  Not going to fix this right now though.


def test_fe_fixed_magmom():
    _test_fe('bulk-spin-fixmagmom', spinmagntarget=2.3)


def test_fe_any_magmom():
    _test_fe('bulk-spin-anymagmom', occopt=7)


def test_h2o():
    atoms = molecule('H2O', vacuum=2.5)
    atoms.calc = abinit(nbands=8)
    run(atoms, 'molecule')


def test_o2():
    atoms = molecule('O2', vacuum=2.5)
    atoms.calc = abinit(nbands=8, occopt=7)
    run(atoms, 'molecule-spin')
    magmom = atoms.get_magnetic_moment()
    print('magmom', magmom)


@pytest.mark.skip('expensive')
def test_manykpts():
    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.01)
    atoms.symbols[:2] = 'Cu'
    atoms.calc = abinit(nbands=len(atoms) * 7,
                        kpts=[8, 8, 8])
    run(atoms, 'manykpts')


@pytest.mark.skip('expensive')
def test_manyatoms():
    atoms = bulk('Ne', cubic=True) * (4, 2, 2)
    atoms.rattle(stdev=0.01)
    atoms.calc = abinit(nbands=len(atoms) * 5)
    run(atoms, 'manyatoms')


#test_many()
#test_big()
#test_si()
def test_au_fhi():
    run_au(pps='fhi')

def test_au_paw():
    run_au(pps='paw', pawecutdg=6.0 * Hartree)
#test_fe_fixed_magmom()
#test_fe_any_magmom()
#test_h2o()
#test_o2()
