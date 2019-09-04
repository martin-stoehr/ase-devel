#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Calculator for CPMD
"""

import numpy as np
import ase
import subprocess
from ase.calculators.calculator import Calculator, FileIOCalculator
from ase.io.cpmd import write_cpmd, read_cpmd, get_valence_electrons
from ase.atoms import Atom, Atoms
from ase.units import Ry
import os
from copy import deepcopy
import json


# keys by atoms object
atoms_keys = [
    'charge',
    'multiplicity',
    'valence_electrons',
    'transfer_orbitals',
    'nsup',
    ]

# keys for data from old CPMD calculation
data_keys = [
    'total_energy',
    'energy_zero',
    'forces',
    'symmetry',
    'geo_opt',
    'filename',
    ]
 
# keys for specific CPMD sections
cpmd_keys = [
    'calc_type',
    'conv_wf',
    'conv_geom',
    'center_molecule',
    'vdw_correction',
    'lanczos_parameter',
    'kohn_sham_energies',
    'lsd',
    'force_lsd',
    ]

system_keys = [
    'cut_off',
    'kpts',
    'poisson_solver',
    'poisson_method',
    ]

dft_keys = [
    'xc',
    ]

pbs_keys = [
    'nodes',
    'memory',
    'walltime']

# Names of parameter sections for functions like self.write_params(). For this reason
# only names of sections written to the file are in the variable
save_param_names = ["cpmd_params", 
                    "system_params",
                    "dft_params",
                    "pbs_params"]

# Names of ALL parameter sections
all_param_names = ["atoms_params",
                   "data_params", 
                   "cpmd_params",
                   "system_params",
                   "dft_params",
                   "pbs_params"]


##############################
# Defaults for cpmd-calc #
##############################
atoms_defaults = {}

data_defaults = {}

cpmd_defaults =    {"calc_type": "singlepoint",
                         "conv_wf": "1.0E-6",
                         "conv_geom": "1.0E-4",
                         "center_molecule": False,
                         "vdw_correction": False,
                         "lanczos_parameter": [100, 8, -1, 1.0e-8],
                         "kohn_sham_energies": 0,
                         "lsd": False,
                         "force_lsd": False
                        }

system_defaults =  {"cut_off": 1500,
                         "poisson_solver": False,
                         "poisson_method": "TUCKERMAN"}

dft_defaults =     {"xc": "PBE"}

pbs_defaults =     {"nodes": 4,
                    "memory": 2000,
                    "walltime": "24:00:00"
                   }

class CPMD(FileIOCalculator):
    """Calculator for CPMD

        """
    implemented_properties = []

    default_parameters = {"cpmd": cpmd_defaults,
                          "system": system_defaults,
                          "dft": dft_defaults,
                          "atoms": atoms_defaults,
                          "data": data_defaults,
                          "pbs": pbs_defaults
                         }

    def __init__(self, atoms, restart=None, **kwargs):
        self.atoms = atoms
        self.state = None
        self.label = None

        self.atoms_params = {}
        self.data_params = {}
        self.cpmd_params = {}
        self.system_params = {}
        self.dft_params = {}
        self.pbs_params = {}

        for key in atoms_keys:
            self.atoms_params[key] = None
        for key in data_keys:
            self.data_params[key] = None
        for key in cpmd_keys:
            self.cpmd_params[key] = None
        for key in system_keys:
            self.system_params[key] = None
        for key in dft_keys:
            self.dft_params[key] = None
        for key in pbs_keys:
            self.pbs_params[key] = None

        self.set_defaults()
        self.set(**kwargs)
        

    def set(self, **kwargs):
        for key in kwargs:
            if key in self.atoms_params:
                self.atoms_params[key] = kwargs[key]
            elif key in self.data_params:
                self.data_params[key] = kwargs[key]
            elif key in self.cpmd_params:
                self.cpmd_params[key] = kwargs[key]
            elif key in self.system_params:
                self.system_params[key] = kwargs[key]
            elif key in self.dft_params:
                self.dft_params[key] = kwargs[key]
            elif key in self.pbs_params:
                self.pbs_params[key] = kwargs[key]
            else:
                raise TypeError('Parameter not defined: ' + key)

        self.energy_zero = self.data_params["energy_zero"]
        self.forces = self.data_params["forces"]
        self.stress = None
        
        if self.label == None:
            self.label = self.data_params['filename']


        # calculate valence electrons
        if self.atoms_params['charge'] is None:
            self.atoms_params['charge'] = 0
        if self.atoms_params['valence_electrons'] is None:
            self.atoms_params['valence_electrons'] = int(get_valence_electrons(self.atoms)) \
                                                     - self.atoms_params['charge']
        if self.atoms_params['multiplicity'] is None:
            self.atoms_params['multiplicity'] = int((self.atoms_params['valence_electrons'] % 2)+1)
        if self.atoms_params['nsup'] is None:
            self.atoms_params['nsup'] = ((self.atoms_params['valence_electrons'] 
                                        - (self.atoms_params['multiplicity'] - 1))/2 
                                        + (self.atoms_params['multiplicity'] - 1))
        # nsup formula: number_of_alpha = 
        # [(val_electrons - number_of_unpaired_e)/2 + number_of_unpaired_e]
        # reason: CPMD takes alpha as majority spin
        
        # update values that change after another one was changed
        # e.g.: valence_electrons after charge
        #       multiplicity after valence_electrons
        #       nsup after multiplicity

        if 'charge' in kwargs:
            # update valence_electrons, multiplicity and nsup.
            self.atoms_params['valence_electrons'] = int(get_valence_electrons(self.atoms)) \
                                                     - self.atoms_params['charge']
            self.atoms_params['multiplicity'] = int((self.atoms_params['valence_electrons'] % 2)+1)
            self.atoms_params['nsup'] = ((self.atoms_params['valence_electrons'] 
                                        - (self.atoms_params['multiplicity'] - 1))/2 
                                        + (self.atoms_params['multiplicity'] - 1))
        elif 'valence_electrons' in kwargs:
            # update multiplicity and nsup. try/except for fo-dft multiplicity. 
            # Maybe avoid this with extra param for fo-dft (fodft_params) 
            # and re-assign the variable in io/cpmd.py as well
            try:
                self.atoms_params['multiplicity'] = int((self.atoms_params['valence_electrons'] % 2)+1)
                self.atoms_params['nsup'] = ((self.atoms_params['valence_electrons'] 
                                            - (self.atoms_params['multiplicity'] - 1))/2 
                                            + (self.atoms_params['multiplicity'] - 1))
            except:
                pass

        elif 'multiplicity' in kwargs:
            # update nsup
            self.atoms_params['nsup'] = ((self.atoms_params['valence_electrons'] 
                                        - (self.atoms_params['multiplicity'] - 1))/2 
                                        + (self.atoms_params['multiplicity'] - 1))


        # check if lsd is necessary OR forced
        try:
            if (self.atoms_params['valence_electrons'] % 2) or (self.cpmd_params['force_lsd'] == True):
                self.cpmd_params['lsd'] = True
            else:
                self.cpmd_params['lsd'] = False
        except:
            pass #everything ok, FO-DFT format for multiplicity found... 

        #IMPORTANT: Multiplicity has to have highest priority. If user changes multiplicity, it
        #           must not update again!
        #           For the user this means: Change charge first, then adjust multiplicity if it
        #           is more complicated than multiplicity = (valence_electrons / 2)+1 (e.g. Metals!?)
        self.params = zip(all_param_names, [self.atoms_params, 
                                            self.data_params, 
                                            self.cpmd_params, 
                                            self.system_params,
                                            self.dft_params,
                                            self.pbs_params])

    def set_defaults(self):
        self.atoms_params.update(atoms_defaults)
        self.data_params.update(data_defaults)
        self.cpmd_params.update(cpmd_defaults)
        self.system_params.update(system_defaults)
        self.dft_params.update(dft_defaults)
        self.pbs_params.update(pbs_defaults)

    def update(self, atoms):
        self.set(charge=atoms.calc.atoms_params['charge'])

    def write_params(self, filename):
        all_params = zip(save_param_names, [self.cpmd_params, 
                                            self.system_params, 
                                            self.dft_params, 
                                            self.pbs_params]) 

        with open(filename, "wb") as outfile:
            json.dump(all_params, outfile)

    def read_params(self, filename):
        """ Read in saved parameters for CPMD-calculator. Will ignore all data_params when reading in"""
        with open(filename, "rb") as infile:
            all_params_named = json.load(infile)
        nothing, all_params = zip(*all_params_named)
        self.cpmd_params, self.system_params, self.dft_params, self.pbs_params = all_params 
        self.set()

    def create_fragments(self, atoms, fo_files=None):
        """ Function to create fragments for FO-DFT from complete cell
            
            Usage: 
            
                frag1, frag2 = atoms.calc.create_fragments(atoms)
            or                        
                atoms.calc.create_fragments(atoms, fo_files=True)

            Latter will invoke atoms.calc.fodft and create the inputs
                   
                   """
        frag1 = deepcopy(atoms)
        frag2 = deepcopy(atoms)

        #if "cpmd.CPMD" not in str(frag1.get_calculator()):
        #    frag1.calc = CPMD(atoms)
        #    frag2.calc = CPMD(atoms)
        
        print("Edit fragment 1")
        frag1.edit()
        frag1.calc.update(frag1)
        print("Edit fragment 2")
        frag2.edit()
        frag2.calc.update(frag2)

        print("Fragments created.")

        if fo_files == True:
            fo_label = raw_input("Label for FO-DFT files: ")
            atoms.calc.fodft(frag1, frag2, label=fo_label, input_only=True)

        return frag1, frag2

    def write_input(self, atoms):
        FileIOCalculator.write_input(self, atoms)
        write_cpmd(os.path.join(self.directory, self.prefix+".inp"), atoms)


    def calculate(self, atoms, command=None):
#        FileIOCalculator.calculate
        self.results = {}
        self.write_input(atoms)
        command_str = command + " /data/schober/opt/CPMD/cpmd.x " + self.prefix+".inp" \
                              + " /data/schober/opt/PSEUDOPOT" + " > " + self.prefix+".out"
        olddir = os.getcwd() 
        os.chdir(self.directory)
        errorcode = subprocess.call(command_str, shell=True)
        if errorcode:
            raise RuntimeError('returned an error: {0} with command {1}'.format(errorcode, command_str))
        
        results = read_cpmd(self.prefix+".out")
        os.chdir(olddir)
        self.results['energy'] = results.calc.data_params['total_energy']
        self.results['forces'] = results.calc.data_params['forces']

    def fodft(self, frag1, frag2, label=None, input_only=False):
        """
        -----------------------------------------------------------------
        | Function to generate FO-DFT input files for CPMD calculations |
        -----------------------------------------------------------------
        
        Arguments...

        frag1 (=Fragment 1) and frag2 (=Fragment 2) have to be
        ASE atoms-objects with CPMD calculator attached.

        label will be used for main folder and filenames.

        input_only=True: Write only files and exit, no qsub @local machine

        Fragment1 is the "master" object, all necessary parameters for the 
        final step for FO-DFT are taken from Fragment1. The function will
        read out fragment specific values from Fragment2 and add them to
        the Fragment1-object. 

        Important parameters for Fragment1 and Fragment2 are:
        ------------------------------------------------------------------------
        charge                      : read in from old CPMD-calculation or set
        multiplicity                : = (valence_electrons % 2) + 1
        valence_electrons           : calculated for pseudo-potential and charge
        ------------------------------------------------------------------------
        These should be set and checked for both fragments!

        Some parameters are taken exclusively from Fragment1 and applied to both
        fragment calculations. These are:
        -----------------------------------------------------
        cell                        : The unit cell, frag1 == frag2
        transfer_orbitals           : Orbitals of interest for S_AB 
        kohn_sham_energies          : Number of empty orbtials to be calculated
        lanczos_parameter           : Parameters for Lanczos Diagonalisation
        label                       : String that will be added to folders/files

        The function will check if the unit cell for both fragments is identical
        and raise an exception when not.

        In the current folder a subfolder "fo_dft_$label" will be generated.

        fo_dft_$label
        ---frag1
        -----frag1.inp
        -----frag1-KS.inp
        ---frag2
        -----frag2.inp
        -----frag2-KS.inp
        ---combine
        -----combine_systems.inp
        -----ksham.inp
        

        """

        if np.array_equal(frag1.get_cell(), frag2.get_cell()) is False:
            raise Exception("Error: Unit cells for fragment 1 and fragment 2 not identical!") 

        #deepcopy now, changes for other fo-dft files should be temporary
        fo_frag1 = deepcopy(frag1)
        fo_frag2 = deepcopy(frag2)

        #LSD always true for FO-DFT, CPMD gives error when used without LSD!
        fo_frag1.calc.set(force_lsd=True)
        fo_frag2.calc.set(force_lsd=True)

        #label magic - priority: arg* label > frag1 label > generic label
        if label is not None:
            fo_frag1.calc.set_label(label)
        else:
            try:
                fo_frag1.calc.prefix
            except:
                fo_frag1.calc.set_label((frag1+frag2).get_chemical_formula())
        
        user_label = fo_frag1.calc.prefix

        fo_frag1.calc.set_label("fo_dft_"+user_label+"/frag1/frag1_"+user_label)
        fo_frag2.calc.set_label("fo_dft_"+user_label+"/frag2/frag2_"+user_label)

        val_frag1 = fo_frag1.calc.atoms_params["valence_electrons"]
        val_frag2 = fo_frag2.calc.atoms_params["valence_electrons"]
        
        charge_frag1 = fo_frag1.calc.atoms_params["charge"]
        charge_frag2 = fo_frag2.calc.atoms_params["charge"]

        multip_frag1 = fo_frag1.calc.atoms_params["multiplicity"]
        multip_frag2 = fo_frag2.calc.atoms_params["multiplicity"]

        nsup1 = fo_frag1.calc.atoms_params["nsup"]
        nsup2 = fo_frag2.calc.atoms_params["nsup"]

        # collect all file paths for submit scripts
        file_paths = []
        
        # write wf-calc for frag1
        if fo_frag1.calc.directory != os.curdir and not os.path.isdir(fo_frag1.calc.directory):
            os.makedirs(fo_frag1.calc.directory)

        write_cpmd(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+".inp"), fo_frag1)
        file_paths.append(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+".inp"))

        # write wf-calc for frag2
        if fo_frag2.calc.directory != os.curdir and not os.path.isdir(fo_frag2.calc.directory):
            os.makedirs(fo_frag2.calc.directory)

        write_cpmd(os.path.join(fo_frag2.calc.directory, fo_frag2.calc.prefix+".inp"), fo_frag2)
        file_paths.append(os.path.join(fo_frag2.calc.directory, fo_frag2.calc.prefix+".inp")) 

        fo_frag1.calc.set(calc_type="fo_frag_ks")
        fo_frag2.calc.set(calc_type="fo_frag_ks")

        write_cpmd(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+"-KS.inp"), fo_frag1)
        write_cpmd(os.path.join(fo_frag2.calc.directory, fo_frag2.calc.prefix+"-KS.inp"), fo_frag2)
        file_paths.append(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+"-KS.inp"))
        file_paths.append(os.path.join(fo_frag2.calc.directory, fo_frag2.calc.prefix+"-KS.inp"))    
        #combine, one atoms object with parameters from 2nd
        fo_frag1.calc.set(calc_type="fo_combine")

        # calculate parameters for combined cell (multiplicity is additive!?)
        # Account for:
        # m1 = s1 + 1 PLUS m2 = s2 + 1 IS NOT m1 + m2 = s1 + s2 + 1
        # correct: IS m1 + m2 = (s1 + s2) + 1 (since its the total number of 
        # unpaired electrons, no matter how many fragments/atoms/whatever

        total_multiplicity = multip_frag1 + multip_frag2 - 1
        total_charge = charge_frag1 + charge_frag2
        total_valence = val_frag1 + val_frag2

        #first, generate string for valence electrons
        val_string = str(val_frag1) + " " + str(val_frag2)
        nsup_string = str(nsup1) + " " + str(nsup2)
        
        #set new values (careful, when charge is changed, valence electrons are updated)
        fo_frag1.calc.set(charge=total_charge)
        fo_frag1.calc.set(multiplicity=total_multiplicity)
        fo_frag1.calc.set(valence_electrons=val_string)
        fo_frag1.calc.set(nsup=nsup_string)

        fo_frag1.extend(fo_frag2) # add atoms from frag2, but keep calculator from frag1!

        fo_frag1.calc.set_label("fo_dft_"+user_label+"/combine/combine_fragments_"+user_label)
        if fo_frag1.calc.directory != os.curdir and not os.path.isdir(fo_frag1.calc.directory):
                    os.makedirs(fo_frag1.calc.directory)
        
        write_cpmd(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+".inp"), fo_frag1)
        file_paths.append(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+".inp"))

        # final step for FO-DFT
        fo_frag1.calc.set_label("fo_dft_"+user_label+"/combine/ksham_final_"+user_label)
        fo_frag1.calc.set(calc_type="fo_ksham")
        write_cpmd(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+".inp"), fo_frag1)
        file_paths.append(os.path.join(fo_frag1.calc.directory, fo_frag1.calc.prefix+".inp"))

        # total generic step for just meeeeeeeeee alooooooooooone
        # write my bash-script for PBS

        nodes = fo_frag1.calc.pbs_params['nodes']
        memory = fo_frag1.calc.pbs_params['memory']
        walltime = fo_frag1.calc.pbs_params['walltime']

        main_folder = "fo_dft_"+user_label+"/"
        submit_file = os.path.join("fo_dft_"+user_label, 'submit_fo_dft_'+user_label)
        f = open(submit_file, "w")
        f.write("""#PBS -N {0} 
#PBS -S /bin/bash 
#PBS -j oe 
#PBS -l walltime={1},nodes=1:ppn={2}:squeeze,pmem={3}mb
#PBS -M christoph.schober@ch.tum.de 
#PBS -m abe     
NODEsc={2}
FROM=$PBS_O_WORKDIR
TMPDIR=${{TMPDIR}}/${{PBS_JOBID}}
JOBID=${{PBS_JOBID}}

export FROM TMPDIR JOBID NODEsc

echo "#--- Job started at `date`"  # write into the output file of the 
   # queuing system at what time the job
   # started 

mkdir -p $TMPDIR || exit 1   # create the temporary directory using  
  # the job ID 

cd $FROM || exit 2   # change into the directory that you 
  # submitted from 

cp -r * $TMPDIR      # copy all necessary files into the temporary directory 
echo $JOBID > $FROM/jobid
cd $TMPDIR        # change into the temporary directory 
""".format(main_folder, walltime, nodes, memory))

        # define !!alot!! of filenames and paths...

        # get relative path right
        # main folder
        main_folder = "fo_dft_"+user_label+"/"
        rel_file_paths = []

        for path in file_paths:
#            path = path.replace(main_folder, "")
            path = os.path.split(path)
            rel_file_paths.append(path[-1])

        frag1 = rel_file_paths[0]
        frag2 = rel_file_paths[1]

        frag1ks = rel_file_paths[2] 
        frag2ks = rel_file_paths[3]

        combine = rel_file_paths[4]
        ksham = rel_file_paths[5]

        f.write("""
cd frag1
mpirun.openmpi -np {0} /data/schober/opt/CPMD/cpmd.x {1} /data/schober/opt/PSEUDOPOT > {2}.out
mpirun.openmpi -np {0} /data/schober/opt/CPMD/cpmd.x {5} /data/schober/opt/PSEUDOPOT > {6}.out

cd ..
cd frag2 

mpirun.openmpi -np {0} /data/schober/opt/CPMD/cpmd.x {3} /data/schober/opt/PSEUDOPOT > {4}.out
mpirun.openmpi -np {0} /data/schober/opt/CPMD/cpmd.x {7} /data/schober/opt/PSEUDOPOT > {8}.out

cd ..

cp frag1/RESTART.1 combine/RESTART.R1
cp frag2/RESTART.1 combine/RESTART.R2

cd combine

mpirun.openmpi -np {0} /data/schober/opt/CPMD/cpmd.x {9} /data/schober/opt/PSEUDOPOT > {10}.out
mpirun.openmpi -np {0} /data/schober/opt/CPMD/cpmd.x {11} /data/schober/opt/PSEUDOPOT > {12}.out

cd ..

rm frag1/RESTART.1 frag2/RESTART.1 combine/RESTART.R1 combine/RESTART.R2 combine/RESTART.1

cp -r * $FROM  # after completion of the run, copy  
  # everything of need back to where you  
  # submitted from 

echo "#--- Job ended at `date`" # write into the output file of the     
   # queuing system at what time the job
   # finished""".format(nodes, frag1, frag1.split(".")[0], frag2, frag2.split(".")[0], frag1ks, 
                        frag1ks.split(".")[0], frag2ks, frag2ks.split(".")[0], combine, 
                        combine.split(".")[0], ksham, ksham.split(".")[0]))

        f.close()

        if input_only == False:
            olddir = os.getcwd()
            os.chdir("fo_dft_"+user_label)
            #command = "qsub -h "+ submit_file 
            code = subprocess.Popen(["qsub", "submit_fo_dft_"+user_label], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            out, err = code.communicate()
            if err:
                 raise RuntimeError('returned an error: {0}'.format(err))
            os.chdir(olddir)
            print(out.strip())

        #DEBUG option!!!
        #return fo_frag1
