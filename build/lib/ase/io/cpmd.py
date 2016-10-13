import numpy as np

from ase.atoms import Atom, Atoms
#from ase.calculators.cpmd import CPMD
from ase.units import Hartree, Bohr, Ry
from ase.parallel import paropen
from ase import data

from collections import defaultdict
from itertools import izip

import math

"""Module to read cell and geometry from CPMD.out-file"""


def read_cpmd(fileobj, index=-1):
    """READ CPMD .out-file."""

    from ase.calculators.cpmd import CPMD
    #  Definition of strings for parsing CPMD-output
    str_isolated_system = 'THIS IS AN ISOLATED SYSTEM CALCULATION'
    str_poisson_solver = 'POISSON EQUATION SOLVER'
    str_symmetry = 'SYMMETRY'
    str_geom_opt = ' OPTIMIZATION OF IONIC POSITIONS'
    
    str_lattice_vec = 'LATTICE VECTOR A1'
    str_final_results = ' *                        FINAL RESULTS                         *'
    str_stars = ' ****************************************************************'
    str_atomic_coords = ' *                      ATOMIC COORDINATES                      *'
    str_conv_wf = 'CONVERGENCE CRITERIA FOR WAVEFUNCTION OPTIMIZATION:'
    str_conv_geom = 'CONVERGENCE CRITERIA FOR GEOMETRY OPTIMIZATION:'
    str_vdw_correction = ' EMPIRICAL VAN DER WAALS CORRECTION AFTER:'
    str_charge = 'CHARGE:    '
    str_cut_off = 'WAVEFUNCTION CUTOFF(RYDBERG):'

    filename = ""
    data_lines = []
    if isinstance(fileobj, str):
        fileobj = paropen(fileobj)
        data_lines = fileobj.readlines()
        filename = str(fileobj.name).split(".")[0]

    isolated = False
    pbc = True
    symmetry = None
    calc_type = "singlepoint"
    geo_opt = False
    conv_wf = None
    conv_geom = None
    charge = None
    cut_off = None
    lsd = False
    vdw_correction = False
    poisson_method = "TUCKERMAN" # pre-define standard, even if poisson_solver = False

    for line in data_lines:
        if str_symmetry in line:
            try:
                symmetry = line.split()[1]
            except:
                symmetry = line.strip()
        elif str_isolated_system in line:
            isolated = True
            pbc = False
    	elif str_poisson_solver in line:
            poisson_method = line.split(':')[1].strip()
            if "TUCKERMAN" in str(poisson_method):
                poisson_method = "TUCKERMAN"
#        elif str_symmetry in line:
#            symmetry = line.split()[1]
        elif str_geom_opt in line:
            calc_type = "geometry_opt"
            geo_opt = True
        elif str_conv_wf in line:
            conv_wf = line.split(':')[1].strip()
        elif str_conv_geom in line:
            conv_geom = line.split(':')[1].strip()
        elif str_charge in line:
            charge = int(round(float(line.split(':')[1].strip()), 0))
        elif str_cut_off in line:
            cut_off = float(line.split(':')[1].strip())*Ry
        elif str_vdw_correction in line:
            vdw_correction = True

    # Set symmetry for ISOLATED (in input it has to be "ISOLATED", not the actual one)
    if isolated == True:
        symmetry = "ISOLATED"

    #  Get lattice vectors in bohrs and convert them to angstrom
    bohr2angstrom = Bohr  # taken from ase.units

    lattice_raw = []
    lattice_pos = []
    lattice_string = []
    lattice_vec = []
    for num, line in enumerate(data_lines, 0):
        if str_lattice_vec in line:
            lattice_pos = num
    lattice_raw = data_lines[lattice_pos:lattice_pos+3]
    for line in lattice_raw:
        lattice_string.append(' '.join(line.split()).split(' '))
    for value in lattice_string:
        lattice_vec.append(value[3:])
    #  Convert lattice vectors to numpy array and angstrom
    lattice_vec = bohr2angstrom*np.array(lattice_vec, dtype=float)

    #  Get right part of the output-file for final results.
    final_pos = []  # position of the FINAL RESULTS flag
    coord_pos_start = []  # position of the ATOMIC COORDINATES section flag
    coord_pos_end = []  # position of the end-flag for coordinates
    atomic_information_raw = []

    for num, line in enumerate(data_lines, 0):
        if str_final_results in line:
            final_pos = num

    if geo_opt is True:
        coord_pos_start = final_pos+5
        for num, line in enumerate(data_lines[coord_pos_start:], 0):
            if str_stars in line:
                coord_pos_end = num
                break
        atomic_information_raw = data_lines[coord_pos_start:coord_pos_start+coord_pos_end-1]
    else:

        for num, line in enumerate(data_lines[final_pos:], 0):
            if str_atomic_coords in line:
                coord_pos_start = final_pos+num
        for num, line in enumerate(data_lines[coord_pos_start+2:], 0):
            if str_stars in line:
                coord_pos_end = num
                break
        #  explanation for following slice:
        #  coord_pos_start+2 to get the first atom after 'ATOMIC COORDINATES' line
        #  coord_pos_start+2+cood_pos_end due to relative counting of coord_pos_end
        atomic_information_raw = data_lines[coord_pos_start+2:coord_pos_start+2+coord_pos_end]

    #  get energy from output file

    for num, line in enumerate(data_lines[final_pos:], 0):
        if 'TOTAL ENERGY' in line:
            tot_energy = Hartree*float(data_lines[final_pos+num].split()[4])

        if 'VDW ENERGY' in line:
            vdw_correction_energy = Hartree*float(data_lines[final_pos+num].split()[3])

    #  make it readable
    atomic_information = []  # list for final coordinate lines
    for item in atomic_information_raw:
        atomic_information.append(' '.join(item.split()).split(' '))
    #  put information into different lists
    atomic_symbols = []
    atomic_number = []  # could add the count later as well... maybe.
    atomic_positions_xyz = []
    atomic_forces = []

    for i in atomic_information:
        atomic_number.append(i[0])
        atomic_symbols.append(i[1])
        atomic_positions_xyz.append([float(i[2]), float(i[3]), float(i[4])])
        if geo_opt is True:
            atomic_forces.append([float(i[5]), float(i[6]), float(i[7])])
        else:
            atomic_forces.append([0,0,0])
    #  convert xyz-positions from bohr to angstrom
    atomic_positions_xyz = bohr2angstrom*np.array(atomic_positions_xyz, dtype=float)

    #  Set atoms object - feed it with dataaaaa
    atoms = Atoms(symbols=atomic_symbols, cell=lattice_vec, pbc=pbc)
    atoms.set_positions(atomic_positions_xyz)

    # calculate valence electrons
    valence = get_valence_electrons(atoms) - charge
    # get standard-multiplicity
    if valence % 2:  # more complex multiplicities need to be done explicitly!
        lsd = True

    multiplicity = int((valence % 2)+1)
        

    value_dic = {
    "atoms" : atoms,
    "energy_zero" : tot_energy,
    "total_energy" : tot_energy,
    "charge" : charge,
    "forces" : atomic_forces,
    "calc_type" : calc_type,
    "conv_wf" : conv_wf,
    "conv_geom" : conv_geom,
    "vdw_correction" : vdw_correction,
    "cut_off" : cut_off,
    "symmetry" : symmetry,
    "filename" : filename,
    "valence_electrons" : valence,
    "multiplicity" : multiplicity,
    "lsd" : lsd,
    "poisson_solver" : isolated,
    "poisson_method" : poisson_method
               }

    #  Set calculator() with data from output-file
    atoms.set_calculator(CPMD(**value_dic))

#    atoms.set_calculator(CPMD(atoms=atoms, energy_zero=tot_energy, total_energy=tot_energy, charge=charge, forces=atomic_forces, calc_type=calc_type, conv_wf=conv_wf, conv_geom=conv_geom, vdw_correction=vdw_correction, cut_off=cut_off, symmetry=symmetry, filename=filename, valence_electrons=valence, multiplicity=multiplicity, lsd=lsd))

    # some conditional calculator settings
#    if isolated == True:
#        atoms.set_calculator(CPMD(atoms=atoms, energy_zero=tot_energy, total_energy=tot_energy, charge=charge, forces=atomic_forces, calc_type=calc_type, conv_wf=conv_wf, conv_geom=conv_geom, vdw_correction=vdw_correction, cut_off=cut_off, filename=filename, valence_electrons=valence, multiplicity=multiplicity, lsd=lsd, symmetry="ISOLATED", poisson_solver=isolated, poisson_method=poisson_method))
#
    return atoms

#function to calculate angle between two vectors
def angle_between(a,b):
    arccosInput = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    return math.acos(arccosInput)

#function to convert rad to degree
def rad2grad(rad):
    pi = math.pi
    deg = 180 * rad / pi
    return deg

#get number of valence electrons for pp_goed_pbe pseudopotentials
def get_valence_electrons(atoms):
    valence = float()
    for item in atoms:
        valence = valence + float(pp_goed_pbe[item.symbol])
    #valence = valence - atoms.calc.atoms_keys['charge'] #need to subtract charge later
    return valence

def write_info(atoms):

    a, b, c, alpha, beta, gamma, symmetry, symmetry_trig = get_symmetry(atoms)

    #  write info
    info_str = """&INFO
 CPMD input written by ASE.
 Calculated symmetry from cell vectors: {0}
&END""".format(symmetry)

    return info_str

def write_control(atoms):
    ### complex procedure, very explicit definitions..

    control_str = "&CPMD\n"
    # type of calculation
    # make a dictionary with the keywords for each kind of supported calculation

    #SINGLEPOINT WAVEFUNCTION OPTIMIZATION
    singlepoint_key = """ OPTIMIZE WAVEFUNCTION
 CONVERGENCE ORBITAL
 {0}\n""".format(atoms.calc.cpmd_params['conv_wf'])

    #STANDARD GEOMETRY OPTIMIZATION (IONIC)
    geom_opt_key = """ OPTIMIZE GEOMETRY XYZ
 CONVERGENCE ORBITAL
 {0}
 CONVERGENCE GEOMETRY
 {1}\n""".format(atoms.calc.cpmd_params['conv_wf'], atoms.calc.cpmd_params['conv_geom'])

    #FO-DFT FRAGMENT-KS
    fo_frag_ks_key = """ KOHN SHAM ENERGIES
 {0}
 RESTART COORDINATES WAVEFUNCTION LATEST
 LANCZOS PARAMETER
 {1}\n""".format(atoms.calc.cpmd_params['kohn_sham_energies'], str(atoms.calc.cpmd_params['lanczos_parameter']).strip("[]()").replace(",", ""))

    #FO-DFT COMBINE
    if atoms.calc.cpmd_params["lsd"] is True:
        nsups = atoms.calc.atoms_params["nsup"]
    else:
        nsups = ""
    if atoms.calc.atoms_params["transfer_orbitals"] is not None:
        fo_combine_key = """ OPTIMIZE WAVEFUNCTION
 RESTART WAVEFUNCTION COORDINATES
 ORTHOGONALIZE LOWDIN
 COMBINE SYSTEMS SAB
  {0} {1}
  {2}\n""".format(atoms.calc.atoms_params["valence_electrons"], nsups, str(atoms.calc.atoms_params["transfer_orbitals"]).strip("[]()").replace(",", ""))
    else:
        fo_combine_key = """ OPTIMIZE WAVEFUNCTION
 RESTART WAVEFUNCTION COORDINATES
 ORTHOGONALIZE LOWDIN
 COMBINE SYSTEMS
  {0} {1}\n""".format(atoms.calc.atoms_params["valence_electrons"], nsups)

    #FO-DFT FINAL KSHAM STEP
    if atoms.calc.atoms_params["transfer_orbitals"] is not None:
        fo_ksham_key = """ OPTIMIZE WAVEFUNCTION
 RESTART WAVEFUNCTION COORDINATES LATEST
 KSHAM ROUT STATE 
 {0}\n""".format(str(atoms.calc.atoms_params["transfer_orbitals"]).strip("[]()").replace(",", ""))
    else:
        fo_ksham_key = """ OPTIMIZE WAVEFUNCTION
 RESTART WAVEFUNCTION COORDINATES LATEST
 KSHAM ROUT\n"""

    calc_dict = {'singlepoint':singlepoint_key,
                 'geom_opt': geom_opt_key,
                 'fo_frag_ks': fo_frag_ks_key,
                 'fo_combine': fo_combine_key,
                 'fo_ksham': fo_ksham_key
                }

    control_str += calc_dict[atoms.calc.cpmd_params['calc_type']]
 
    # additional params
    if atoms.calc.cpmd_params['center_molecule'] == False:
        control_str += " CENTER MOLECULE OFF\n"
    else:
        control_str += " CENTER MOLECULE ON\n"

    if atoms.calc.cpmd_params['vdw_correction'] == True:
        control_str += " VDW CORRECTION ON\n"

    if atoms.calc.cpmd_params['lsd'] == True:
        control_str += " LSD\n"
    
    control_str += "&END"

    return control_str

def write_sysin(atoms):
    
    a, b, c, alpha, beta, gamma, symmetry, symmetry_trig = get_symmetry(atoms)

    if atoms.calc.system_params['poisson_solver'] is True:
        sym_str = "POISSON SOLVER {0}\n SYMMETRY".format(atoms.calc.system_params['poisson_method'])
        symmetry = "ISOLATED"
    else:
        sym_str = "SYMMETRY"

    try:
        cutoff = int(round(atoms.calc.system_params['cut_off']/Ry, 1))
    except:
        cutoff = "XXX"

    try:
        if atoms.calc.system_params['kpts'] is not None:
            kpts = str(atoms.calc.system_params['kpts']).strip("[]()").replace(",", "")
        else:
            raise Exception
    except:
        kpts = False
        pass

    sysin_str = """&SYSTEM
 ANGSTROM
 {8}
 {0}
 CELL ABSOLUTE DEGREE
 {1:.6f} {2:.6f} {3:.6f} {4:.3f} {5:.3f} {6:.3f}
 CUTOFF
 {7}\n""".format(symmetry, a, b, c, alpha, beta, gamma, cutoff, sym_str)

    #check for kpts
    if kpts is not False:
        sysin_str += " KPOINTS MONKHORST-PACK\n {0}\n".format(kpts)
    
    #check for charge
    if atoms.calc.atoms_params['charge'] is not 0:
        sysin_str += " CHARGE\n {0}\n".format(atoms.calc.atoms_params['charge'])
    if atoms.calc.atoms_params['multiplicity'] is not 1:
        sysin_str += " MULTIPLICITY\n {0}\n".format(atoms.calc.atoms_params['multiplicity'])

    sysin_str += "&END"

    return sysin_str


def write_dft(atoms):
    functional = atoms.calc.dft_params['xc']
    dft_str = """&DFT
 NEWCODE
 FUNCTIONAL {0}
&END""".format(functional)

    return dft_str

def write_vdw():
    vdw_str = """&VDW
 VDW PARAMETERS
 ALL GRIMME
&END"""
    return vdw_str

def write_atoms(atoms):

    #  we need to get the positions sorted by atom to be able to print the
    #  CPMD-type geometry-part
    position_dict = defaultdict(list)

    #  create a dictionary from atoms-object, type: {'atomsymbol':'array([x, y, z])'}
    for item in atoms:
        position_dict[item.symbol].append(item.position)

    #  get the atom types and lists of positions per type
    list_of_atoms = position_dict.keys()
    list_of_positions = position_dict.values()

    positions = atoms.get_positions()

    atoms_str = "&ATOMS\n"
    
    # now write the tricky part
    for a, p in izip(list_of_atoms, list_of_positions):
        #  get correct atomic number for pseudo potential
        atomic_number = pp_goed_pbe[a]
        atoms_str += "*PP_GOED/pbe/{0}-q{1}\n".format(a, atomic_number)
        atoms_str += " LMAX=D LOC=D\n"
        atoms_str += " {0}\n".format(len(p))
        for item in p:
            item = item.tolist()
            atoms_str += "{0:12.8f} {1:12.8f} {2:12.8f}\n".format(*item)
    atoms_str += "&END"

    return atoms_str

def get_symmetry(atoms):

    #dictionary for symmetries in CPMD
    symmetries = {
    "ISOLATED": 0,
    "CUBIC": 1,
    "FACE CENTERED CUBIC": 2,
    "BODY CENTERED CUBIC": 3,
    "HEXAGONAL": 4,
    "RHOMBOHEDRAL": 5,
    "TETRAGONAL": 6,
    "BODY CENTERED TETRAGONAL": 7,
    "ORTHORHOMBIC": 8,
    "MONOCLINIC": 12,
    "TRICLINIC": 14
    }

    vectors = atoms.get_cell()

    # cell parameters complete: a, b, c, alpha, beta, gamma
    a = float(np.linalg.norm(vectors[0]))
    b = float(np.linalg.norm(vectors[1]))
    c = float(np.linalg.norm(vectors[2]))
    alpha = float(rad2grad(angle_between(vectors[1], vectors[2])))
    beta = float(rad2grad(angle_between(vectors[0], vectors[2])))
    gamma = float(rad2grad(angle_between(vectors[0], vectors[1])))

    #round values to avoid floating point stuff (quick'n'dirty)
    a = round(a, 6)
    b = round(b, 6)
    c = round(c, 6)
    alpha = round(alpha, 3)
    beta = round(beta, 3)
    gamma = round(gamma, 3)

    #get symmetry for given cell parameters
    symmetry_trig = True  #Trigger for automagic symmetry. True, because we can do!

    #CUBIC, [1, 2, 3]
    if a==b==c and alpha==beta==gamma==90:
        symmetry = "CUBIC"
    #HEXAGONAL, [4]
    elif (a==b and alpha==120 or beta==120) or (b==c and beta==120 or gamma==120) or (a==c and alpha==120 or gamma==120):
        symmetry = "HEXAGONAL"
    #TRIGONAL or RHOMBOHEDRAL, [5]
    elif a==b==c and alpha==beta==gamma!=90:
        symmetry = "RHOMBOHEDRAL"

    #TETRAGONAL, [6]
    elif (a==b!=c or a==c!=b or b==c!=a) and alpha==beta==gamma==90:
        symmetry = "TETRAGONAL"

    #BODY CENTERED TETRAGONAL, [7]

    #ORTHORHOMBIC, [8]
    elif a!=b!=c and alpha==beta==gamma==90:
        symmetry = "ORTHORHOMBIC"

    #MONOCLINIC, [12]
    elif a!=b!=c and (alpha==beta==90!=gamma or alpha==gamma==90!=beta or beta==gamma==90!=alpha):
        symmetry = "MONOCLINIC"

    #TRICLINIC, [14]
    elif a!=b!=c and alpha!=beta!=gamma:
        symmetry = "TRICLINIC"

    else:
        symmetry = "Symmetry UNKNOWN! Algorithm not even finds TRICLINIC! CELL VECTOR is used. If the system should have a symmetry, check manually."
        symmetry_trig = False

    return a, b, c, alpha, beta, gamma, symmetry, symmetry_trig

def write_cpmd(fileobj, atoms):
    """Write cpmd *.inp-file with data from atoms object and cpmd-calculator"""
    if isinstance(fileobj, str):
        f = paropen(fileobj, 'w')
    
    # check for attached calculator. Only try to get all the CPMD-params when 
    # the cpmd.CPMD calculator is found
    if "ase.calculators.cpmd" in str(atoms.get_calculator()): 
        f.write(write_info(atoms))
        f.write("\n\n")
        f.write(write_control(atoms))
        f.write("\n\n")
        f.write(write_sysin(atoms))
        f.write("\n\n")
        f.write(write_dft(atoms))
        f.write("\n\n")
        if atoms.calc.cpmd_params['vdw_correction'] == True:
            f.write(write_vdw())
            f.write("\n\n")
        f.write(write_atoms(atoms))
        
    # if no or any other calculator is found, write only geometry-relevant parameters
    else:
        info_str = write_info(atoms)
        # Write some info about the partial input file generated
        info_str = info_str.replace("&END", " Input-File INCOMPLETE\n&END")    
        
        f.write(info_str)
        f.write("\n\n")
        f.write(write_sysin(atoms))
        f.write("\n\n")
        f.write(write_atoms(atoms))

    f.close()

def write_cpmd_fodft(fileobj, atoms):
    """Write cpmd *.inp-file with data from atoms object and cpmd-calculator"""
    if isinstance(fileobj, str):
        f = paropen(fileobj, 'w')

    # check for attached calculator. Only try to get all the CPMD-params when 
    # the cpmd.CPMD calculator is found
    #if "ase.calculators.cpmd" in str(atoms.get_calculator()):
    f.write(write_info(atoms))
    f.write("\n\n")
    f.write(write_control(atoms))
    f.write("\n\n")
    f.write(write_sysin(atoms))
    f.write("\n\n")
    f.write(write_dft(atoms))
    f.write("\n\n")
    if atoms.calc.cpmd_params['vdw_correction'] == True:
        f.write(write_vdw())
        f.write("\n\n")
    f.write(write_atoms(atoms))
    f.close


#  dictionary for pseudo_potentials (only pp_goed_pbe right now!)
#  used for cpmd-atoms-section and number of valence electrons
pp_goed_pbe = {
"Al": 3,
"Ar": 8,
"Au": 11,
"Be": 4,
"B": 3,
"Cl": 7,
"C": 4,
"F": 7,
"He": 2,
"H": 1,
"Li": 3,
"Mg": 10,
"Na": 9,
"Ne": 8,
"N": 5,
"O": 6,
"P": 5,
"Pt": 10,
"Si": 4,
"S": 6,
"Zr": 12
}
