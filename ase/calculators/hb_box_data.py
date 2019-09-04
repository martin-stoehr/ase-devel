###################################################
#
#  This module was taken from the Hotbit package 
#          (distributed under GNU GPL)
#
###################################################

# Copyright (C) 2008 NSC Jyvaskyla
# Please see the accompanying LICENSE file for further information.
#
# Experimental data * mass, R_cov (2008 data), R_vdw, EA from www.webelements.com (updated 21/May/2015)
#                   * IE from gElemental 1.2.0
#                   * EN according to Allred and Rochow (Wiley-VCH periodic table, 2007)
#
# UNITS:
#     * mass in amu
#     * all radii in Angstrom
#     * all energies in eV

from numpy import nan

data={}

data['H'] ={'Z':1, 'symbol':'H',  'name':'hydrogen',  'mass': 1.0079, 'R_cov':0.31, 'R_vdw':1.20, 'IE':0.0135, 'EA':72.27, 'add_orb':'1s', 'rem_orb':'1s', 'EN':2.2 }            
data['He']={'Z':2, 'symbol':'He', 'name':'helium',                                                                         'add_orb':'2s', 'rem_orb':'1s' }
data['Li']={'Z':3, 'symbol':'Li', 'name':'lithium',   'mass':6.941,   'R_cov':1.28, 'R_vdw':1.82,                          'add_orb':'2s', 'rem_orb':'2s', 'EN':1.  }
data['Be']={'Z':4, 'symbol':'Be', 'name':'beryllium', 'mass':9.0122,  'R_cov':0.96,                                        'add_orb':'2p', 'rem_orb':'2s', 'EN':1.5 }
data['B'] ={'Z':5, 'symbol':'B',  'name':'boron',     'mass':10.81,   'R_cov':0.84, 'R_vdw':2.08, 'IE':8.294,  'EA':0.277, 'add_orb':'2p', 'rem_orb':'2p', 'EN':2.  }
data['C'] ={'Z':6, 'symbol':'C',  'name':'carbon',    'mass':12.0107, 'R_cov':0.76, 'R_vdw':1.70, 'IE':11.256, 'EA':1.594, 'add_orb':'2p', 'rem_orb':'2p', 'EN':2.5 }            
data['N'] ={'Z':7, 'symbol':'N',  'name':'nitrogen',  'mass':14.0067, 'R_cov':0.71, 'R_vdw':1.55, 'IE':14.527, 'EA':0.072, 'add_orb':'2p', 'rem_orb':'2p', 'EN':3.1 }            
data['O'] ={'Z':8, 'symbol':'O',  'name':'oxygen',    'mass':15.9994, 'R_cov':0.66, 'R_vdw':1.52, 'IE':13.612, 'EA':1.460, 'add_orb':'2p', 'rem_orb':'2p', 'EN':3.5 }            
data['F'] ={'Z':9, 'symbol':'F',  'name':'fluorine',  'mass':18.9984, 'R_cov':0.57, 'R_vdw':1.47, 'IE':17.4228,'EA':3.4012,'add_orb':'2p', 'rem_orb':'2p', 'EN':4.1 }    
data['Ne']={'Z':10,'symbol':'Ne', 'name':'neon',                                                                           'add_orb':'3s', 'rem_orb':'2p' }
data['Na']={'Z':11,'symbol':'Na', 'name':'sodium',    'mass':22.9898, 'R_cov':1.66, 'R_vdw':2.27, 'IE':5.136,  'EA':0.547, 'add_orb':'3s', 'rem_orb':'3s', 'EN':1.  }           
data['Mg']={'Z':12,'symbol':'Mg', 'name':'magnesium', 'mass':24.3050, 'R_cov':1.41, 'R_vdw':1.73, 'IE':7.642,  'EA':0.000, 'add_orb':'3p', 'rem_orb':'3s', 'EN':1.2 }             
data['Al']={'Z':13,'symbol':'Al', 'name':'aluminium', 'mass':26.9815, 'R_cov':1.21, 'R_vdw':nan,  'IE':5.986,              'add_orb':'3p', 'rem_orb':'3p', 'EN':1.5 }
data['Si']={'Z':14,'symbol':'Si', 'name':'silicon',   'mass':28.0855, 'R_cov':1.11, 'R_vdw':2.10, 'IE':8.151,              'add_orb':'3p', 'rem_orb':'3p', 'EN':1.7 }
data['P'] ={'Z':15,'symbol':'P',  'name':'phosphorus','mass':30.9738, 'R_cov':1.07, 'R_vdw':1.80, 'IE':10.486,             'add_orb':'3p', 'rem_orb':'3p', 'EN':2.1 }
data['S'] ={'Z':16,'symbol':'S',  'name':'sulfur',    'mass':32.065,  'R_cov':1.05, 'R_vdw':1.80, 'IE':10.356, 'EA':2.072, 'add_orb':'3p', 'rem_orb':'3p', 'EN':2.4 }
data['Cl']={'Z':17,'symbol':'Cl', 'name':'chlorine',  'mass':35.4530, 'R_cov':1.02, 'R_vdw':1.75, 'IE':12.962, 'EA':3.615, 'add_orb':'3p', 'rem_orb':'3p', 'EN':2.8 }            
data['Ar']={'Z':18,'symbol':'Ar', 'name':'argon',     'mass':39.948,                                                       'add_orb':'3p', 'rem_orb':'4s' }
data['K'] ={'Z':19,'symbol':'K',  'name':'potassium', 'mass':39.0983, 'R_cov':2.03, 'R_vdw':2.75, 'IE':4.338,  'EA':0.501, 'add_orb':'4s', 'rem_orb':'4s', 'EN':0.9 }             
data['Ca']={'Z':20,'symbol':'Ca', 'name':'calcium',   'mass':40.078,  'R_cov':1.41,               'IE':6.113,              'add_orb':'3d', 'rem_orb':'4s', 'EN':1.  }
data['Sc']={'Z':21,'symbol':'Sc', 'name':'scandium',  'mass':44.9559, 'R_cov':1.44,               'IE':6.54,               'add_orb':'3d', 'rem_orb':'3d', 'EN':1.2 }
data['Ti']={'Z':22,'symbol':'Ti', 'name':'titanium',  'mass':47.8760, 'R_cov':1.60, 'R_vdw':2.15, 'IE':6.825,  'EA':0.078, 'add_orb':'3d', 'rem_orb':'3d', 'EN':1.3 }            
data['V'] ={'Z':23,'symbol':'V',  'name':'vanadium',  'mass':50.942,  'R_cov':1.22, 'add_orb':'3d', 'rem_orb':'3d'}
data['Cr']={'Z':24,'symbol':'Cr', 'name':'chromium',  'mass':51.9961, 'R_cov':1.39, 'R_vdw':nan,  'IE':6.766,              'add_orb':'3d', 'rem_orb':'3d', 'EN':1.6 }
data['Mn']={'Z':25,'symbol':'Mn', 'name':'manganese', 'mass':54.938,  'R_cov':1.17, 'add_orb':'3d', 'rem_orb':'3d'}
data['Fe']={'Z':26,'symbol':'Fe', 'name':'iron',      'mass':55.845,  'R_cov':1.32, 'R_cov_hs':1.52, 'IE':7.870,        'add_orb':'3d', 'rem_orb':'3d', 'EN':1.6 }
data['Co']={'Z':27,'symbol':'Co', 'name':'cobalt',    'mass':58.933,  'R_cov':1.16, 'add_orb':'3d', 'rem_orb':'3d'}
data['Ni']={'Z':28,'symbol':'Ni', 'name':'nickel',    'mass':58.6934, 'R_cov':1.24, 'R_vdw':1.63, 'IE':7.635,              'add_orb':'3d', 'rem_orb':'3d', 'EN':1.5 }
data['Cu']={'Z':29,'symbol':'Cu', 'name':'copper',    'mass':63.546,  'R_cov':1.38, 'R_vdw':2.00, 'IE':7.727,  'EA':1.227, 'add_orb':'4s', 'rem_orb':'4s', 'EN':1.8 }
data['Zn']={'Z':30,'symbol':'Zn', 'name':'zinc',      'mass':65.38,   'R_cov':1.25, 'add_orb':'4p', 'rem_orb':'4s'}
data['Ga']={'Z':31,'symbol':'Ga', 'name':'gallium',   'mass':69.723,  'R_cov':1.26, 'add_orb':'4p', 'rem_orb':'4p'}
data['Ge']={'Z':32,'symbol':'Ge', 'name':'germanium', 'mass':62.631,  'R_cov':1.22, 'add_orb':'4p', 'rem_orb':'4p'}
data['As']={'Z':33,'symbol':'As', 'name':'arsenic',   'mass':74.922,  'R_cov':1.20, 'add_orb':'4p', 'rem_orb':'4p'}
data['Se']={'Z':34,'symbol':'Se', 'name':'selenium',  'mass':78.971,  'R_cov':1.16, 'add_orb':'4p', 'rem_orb':'4p'}
data['Br']={'Z':35,'symbol':'Br', 'name':'bromine',   'mass':79.904,  'R_cov':1.20, 'R_vdw':1.85, 'IE':11.814,             'add_orb':'4p', 'rem_orb':'4p', 'EN':2.7 }
data['Kr']={'Z':36,'symbol':'Kr', 'name':'krypton',                                                                        'add_orb':'5s', 'rem_orb':'4p' }
data['Rb']={'Z':37,'symbol':'Rb', 'name':'rubidium',  'mass':84.468,  'R_cov':2.16, 'add_orb':'5s', 'rem_orb':'5s'}
data['Sr']={'Z':38,'symbol':'Sr', 'name':'strontium', 'mass':87.62,   'R_cov':1.95, 'R_vdw':2.49, 'IE':5.69,   'EA':0.052, 'add_orb':'4d', 'rem_orb':'5s', 'EN':1.  }
data['Y']= {'Z':39,'symbol':'Y',  'name':'yttrium',   'mass':88.906,  'R_cov':1.62, 'add_orb':'4d', 'rem_orb':'4d'}
data['Zr']={'Z':40,'symbol':'Zr', 'name':'zirconium', 'mass':91.224,  'R_cov':1.45, 'add_orb':'4d', 'rem_orb':'4d'}
data['Nb']={'Z':41,'symbol':'Nb', 'name':'niobium',   'mass':92.906,  'R_cov':1.34, 'add_orb':'4d', 'rem_orb':'4d'}
data['Mo']={'Z':42,'symbol':'Mo', 'name':'molybdenum','mass':95.94,   'R_cov':1.54, 'R_vdw':2.10, 'IE':7.08,   'EA':0.744, 'add_orb':'5s', 'rem_orb':'4d', 'EN':1.3 }
data['Tc']={'Z':43,'symbol':'Tc', 'name':'technetium','mass':98.907,  'R_cov':1.27, 'add_orb':'4d', 'rem_orb':'5s'}
data['Ru']={'Z':44,'symbol':'Ru', 'name':'ruthenium', 'mass':101.07,  'R_cov':1.46,               'IE':7.37,               'add_orb':'4d', 'rem_orb':'4d', 'EN':1.4 }
data['Rh']={'Z':45,'symbol':'Rh', 'name':'rhodium',   'mass':102.9055,'R_cov':1.42,               'IE':7.46,               'add_orb':'4d', 'rem_orb':'4d', 'EN':1.5 }
data['Pd']={'Z':46,'symbol':'Pd', 'name':'palladium', 'mass':106.42,  'R_cov':1.39, 'R_vdw':1.63, 'IE':8.337,              'add_orb':'5s', 'rem_orb':'4d', 'EN':1.4 }
data['Ag']={'Z':47,'symbol':'Ag', 'name':'silver',    'mass':107.868, 'R_cov':1.45, 'R_vdw':1.72, 'IE':7.576,  'EA':1.302, 'add_orb':'5s', 'rem_orb':'5s', 'EN':1.4 }
data['Cd']={'Z':48,'symbol':'Cd', 'name':'cadmium',   'mass':112.414, 'R_cov':1.48, 'add_orb':'5p', 'rem_orb':'5s'}
data['In']={'Z':49,'symbol':'In', 'name':'indium',    'mass':114.818, 'R_cov':1.44, 'add_orb':'5p', 'rem_orb':'5p'}
data['Sn']={'Z':50,'symbol':'Sn', 'name':'tin',       'mass':118.710, 'R_cov':1.39, 'R_vdw':2.17, 'IE':7.344,              'add_orb':'5p', 'rem_orb':'5p', 'EN':1.7 }
data['Sb']={'Z':51,'symbol':'Sb', 'name':'antimony',  'mass':121.760, 'R_cov':1.40, 'add_orb':'5p', 'rem_orb':'5p'}
data['Te']={'Z':52,'symbol':'Te', 'name':'tellurium', 'mass':127.6,   'R_cov':1.36, 'add_orb':'5p', 'rem_orb':'5p'}
data['I'] ={'Z':53,'symbol':'I',  'name':'iodine',    'mass':126.9045,'R_cov':1.33, 'R_vdw':2.20, 'IE':10.451,             'add_orb':'5p', 'rem_orb':'5p', 'EN':2.2 }
data['Xe']={'Z':54,'symbol':'Xe', 'name':'xenon',     'mass':131.293,               'R_vdw':2.10, 'IE':12.130,             'add_orb':'6s', 'rem_orb':'5p' }
data['Cs']={'Z':55,'symbol':'Cs', 'name':'cesium',    'mass':132.905, 'R_cov':2.35, 'add_orb':'6s', 'rem_orb':'6s'}
data['Ba']={'Z':56,'symbol':'Ba', 'name':'barium',    'mass':137.328, 'R_cov':1.98, 'add_orb':'5d', 'rem_orb':'6s'}
data['Lu']={'Z':71,'symbol':'Lu', 'name':'lutetium',  'mass':174.967, 'R_cov':1.56, 'add_orb':'5d', 'rem_orb':'5d'}
data['Hf']={'Z':72,'symbol':'Hf', 'name':'hafnium',   'mass':178.49,  'R_cov':1.44, 'add_orb':'5d', 'rem_orb':'5d'}
data['Ta']={'Z':73,'symbol':'Ta', 'name':'tantalum',  'mass':180.948, 'R_cov':1.34, 'add_orb':'5d', 'rem_orb':'5d'}
data['W'] ={'Z':74,'symbol':'W',  'name':'tungsten',  'mass':183.84,  'R_cov':1.30, 'add_orb':'5d', 'rem_orb':'5d'}
data['Re']={'Z':75,'symbol':'Re', 'name':'rhenium',   'mass':186.207, 'R_cov':1.28, 'add_orb':'5d', 'rem_orb':'5d'}
data['Os']={'Z':76,'symbol':'Os', 'name':'osmium',    'mass':190.23,  'R_cov':1.26, 'add_orb':'5d', 'rem_orb':'5d'}
data['Ir']={'Z':77,'symbol':'Ir', 'name':'iridium',   'mass':192.217, 'R_cov':1.27, 'add_orb':'5d', 'rem_orb':'5d'}
data['Pt']={'Z':78,'symbol':'Pt','name':'platinum',   'mass':195.084, 'R_cov':1.36,'R_vdw':1.75, 'IE':9.013,  'EA':2.127, 'add_orb':'5d', 'rem_orb':'6s', 'EN':1.4 }
data['Au']={'Z':79,'symbol':'Au','name':'gold',       'mass':196.9666,'R_cov':1.36,'R_vdw':1.66, 'IE':9.221,  'EA':2.308, 'add_orb':'6s', 'rem_orb':'6s', 'EN':1.4 }
data['Hg']={'Z':80,'symbol':'Hg', 'name':'mercury',   'mass':200.592, 'R_cov':1.49, 'add_orb':'6s', 'rem_orb':'6p'}
data['Tl']={'Z':81,'symbol':'Tl', 'name':'thallium',  'mass':204.383, 'R_cov':1.48, 'add_orb':'6p', 'rem_orb':'6p'}
data['Pb']={'Z':82,'symbol':'Pb', 'name':'lead',      'mass':207.2,   'R_cov':1.47, 'add_orb':'6p', 'rem_orb':'6p'}
data['Bi']={'Z':83,'symbol':'Bi', 'name':'bismuth',   'mass':208.980, 'R_cov':1.46, 'add_orb':'6p', 'rem_orb':'6p'}
data['Po']={'Z':84,'symbol':'Po', 'name':'polonium',  'mass':208.982, 'R_cov':1.46, 'add_orb':'6p', 'rem_orb':'6p'}
data['At']={'Z':85,'symbol':'At', 'name':'astatine',  'mass':209.987, 'R_cov':1.45, 'add_orb':'6p', 'rem_orb':'6p'}
data['Rn']={'Z':86,'symbol':'Rn', 'name':'radon',     'mass':222.018, 'R_cov':nan,  'add_orb':'7s', 'rem_orb':'6p'}
data['Fr']={'Z':87,'symbol':'Fr', 'name':'francium',  'mass':223.020, 'R_cov':nan,  'add_orb':'7s', 'rem_orb':'7s'}
data['Ra']={'Z':88,'symbol':'Ra', 'name':'radium',    'mass':226.025, 'R_cov':nan,  'add_orb':'7s', 'rem_orb':'6d'}
data['Ac']={'Z':89,'symbol':'Ac', 'name':'actinium',  'mass':227.0278,                            'IE':5.381,              'add_orb':'6d', 'rem_orb':'6d', 'EN':1.  }   
data['Th']={'Z':90,'symbol':'Th','name':'thorium',    'mass':232.0381, 'R_cov':1.65,              'IE':6.307,              'add_orb':'5f', 'rem_orb':'6d', 'EN':1.1 }     
data['U'] ={'Z':92,'symbol':'U', 'name':'uranium',    'mass':238.0289, 'R_cov':1.42,              'IE':6.194,              'add_orb':'5f', 'rem_orb':'5f', 'EN':1.2 }   
data['Np']={'Z':93,'symbol':'Np','name':'neptunium',  'mass':237.048,                             'IE':6.266,              'add_orb':'5f', 'rem_orb':'6d', 'EN':1.2 }
data['Pu']={'Z':94,'symbol':'Pu','name':'plutonium',  'mass':244.0642,                            'IE':6.026,              'add_orb':'5f', 'rem_orb':'5f', 'EN':1.2 }    
data['X'] ={'Z':99,'symbol':'X', 'name':'dummy'} 
## 'add_orb' and 'rem_orb' for calculations of IP and EA
#  (e- to 'add_orb' => Anion, e- from 'rem_orb' => cation) 

# update with valence orbital data                        
valence_orbitals={}
valence_orbitals['H'] =['1s']
valence_orbitals['He']=['1s']
valence_orbitals['Li']=['2s','2p']
valence_orbitals['Be']=['2s','2p']
valence_orbitals['B'] =['2s','2p']
valence_orbitals['C'] =['2s','2p']
valence_orbitals['N'] =['2s','2p']
valence_orbitals['O'] =['2s','2p']
valence_orbitals['F'] =['2s','2p']
valence_orbitals['Ne']=['2s','2p']
valence_orbitals['Na']=['3s','3p']
valence_orbitals['Mg']=['3s','3p']
valence_orbitals['Al']=['3s','3p']
valence_orbitals['Si']=['3s','3p']
valence_orbitals['P'] =['3s','3p']
valence_orbitals['S'] =['3s','3p']
valence_orbitals['Cl']=['3s','3p']
valence_orbitals['Ar']=['3s','3p']
valence_orbitals['K'] =['3d','4s','4p']
valence_orbitals['Ca']=['3d','4s','4p']
valence_orbitals['Sc']=['3d','4s','4p']
valence_orbitals['Ti']=['3d','4s','4p']
valence_orbitals['V'] =['3d','4s','4p']
valence_orbitals['Cr']=['3d','4s','4p']
valence_orbitals['Mn']=['3d','4s','4p']
valence_orbitals['Fe']=['3d','4s','4p']
valence_orbitals['Co']=['3d','4s','4p']
valence_orbitals['Ni']=['3d','4s','4p']
valence_orbitals['Cu']=['3d','4s','4p']
valence_orbitals['Zn']=['3d','4s','4p']
valence_orbitals['Ga']=['3d','4s','4p']
valence_orbitals['Ge']=['3d','4s','4p']
valence_orbitals['As']=['3d','4s','4p']
valence_orbitals['Se']=['3d','4s','4p']
valence_orbitals['Br']=['3d','4s','4p']
valence_orbitals['Kr']=['3d','4s','4p']
valence_orbitals['Rb']=['4d','5s','5p']
valence_orbitals['Sr']=['4d','5s','5p']
valence_orbitals['Y'] =['4d','5s','5p']
valence_orbitals['Zr']=['4d','5s','5p']
valence_orbitals['Nb']=['4d','5s','5p']
valence_orbitals['Mo']=['4d','5s','5p']
valence_orbitals['Tc']=['4d','5s','5p']
valence_orbitals['Ru']=['4d','5s','5p']
valence_orbitals['Rh']=['4d','5s','5p']
valence_orbitals['Pd']=['4d','5s','5p']
valence_orbitals['Ag']=['4d','5s','5p']
valence_orbitals['Cd']=['4d','5s','5p']
valence_orbitals['In']=['4d','5s','5p']
valence_orbitals['Sn']=['4d','5s','5p']
valence_orbitals['Sb']=['4d','5s','5p']
valence_orbitals['Te']=['4d','5s','5p']
valence_orbitals['I'] =['4d','5s','5p']
valence_orbitals['Xe']=['4d','5s','5p']
valence_orbitals['Cs']=['5d','6s','5p']
valence_orbitals['Ba']=['5d','6s','5p']
valence_orbitals['Lu']=['5d','6s','6p']
valence_orbitals['Hf']=['5d','6s','6p']
valence_orbitals['Ta']=['5d','6s','6p']
valence_orbitals['W'] =['5d','6s','6p']
valence_orbitals['Re']=['5d','6s','6p']
valence_orbitals['Os']=['5d','6s','6p']
valence_orbitals['Ir']=['5d','6s','6p']
valence_orbitals['Pt']=['5d','6s','6p']
valence_orbitals['Au']=['5d','6s','6p']
valence_orbitals['Hg']=['5d','6s','6p']
valence_orbitals['Tl']=['5d','6s','6p']
valence_orbitals['Pb']=['5d','6s','6p']
valence_orbitals['Bi']=['5d','6s','6p']
valence_orbitals['Po']=['5d','6s','6p']
valence_orbitals['At']=['5d','6s','6p']
valence_orbitals['Rn']=['5d','6s','6p']
valence_orbitals['Fr']=['6d','7s','7p']
valence_orbitals['Ra']=['6d','7s','7p']
valence_orbitals['Ac']=['5f','6d','7s','7p']
valence_orbitals['Th']=['5f','6d','7s','7p']
valence_orbitals['U'] =['5f','6d','7s','7p']
valence_orbitals['Np']=['5f','6d','7s','7p']
valence_orbitals['Pu']=['5f','6d','7s','7p']

for key in valence_orbitals:
    data[key]['valence_orbitals']=valence_orbitals[key]


# Set electronic configurations (orbital occupations)
aux=[ ['H', '',{'1s':1}],\
      ['He','',{'1s':2}],\
      # second row
      ['Li','He',{'2s':1,'2p':0}],\
      ['Be','He',{'2s':2,'2p':0}],\
      ['B', 'He',{'2s':2,'2p':1}],\
      ['C', 'He',{'2s':2,'2p':2}],\
      ['N', 'He',{'2s':2,'2p':3}],\
      ['O', 'He',{'2s':2,'2p':4}],\
      ['F', 'He',{'2s':2,'2p':5}],\
      ['Ne','He',{'2s':2,'2p':6}],\
      # third row
      ['Na','Ne',{'3s':1,'3p':0}],\
      ['Mg','Ne',{'3s':2,'3p':0}],\
      ['Al','Ne',{'3s':2,'3p':1}],\
      ['Si','Ne',{'3s':2,'3p':2}],\
      ['P', 'Ne',{'3s':2,'3p':3}],\
      ['S', 'Ne',{'3s':2,'3p':4}],\
      ['Cl','Ne',{'3s':2,'3p':5}],\
      ['Ar','Ne',{'3s':2,'3p':6}],\
      # fourth row
      ['K', 'Ar',{'3d':0,'4s':1,'4p':0}],\
      ['Ca','Ar',{'3d':0,'4s':2,'4p':0}],\
      ['Sc','Ar',{'3d':1,'4s':2,'4p':0}],\
      ['Ti','Ar',{'3d':2,'4s':2,'4p':0}],\
      ['V', 'Ar',{'3d':3,'4s':2,'4p':0}],\
      ['Cr','Ar',{'3d':5,'4s':1,'4p':0}],\
      ['Mn','Ar',{'3d':5,'4s':2,'4p':0}],\
      ['Fe','Ar',{'3d':6,'4s':2,'4p':0}],\
      ['Co','Ar',{'3d':7,'4s':2,'4p':0}],\
      ['Ni','Ar',{'3d':8,'4s':2,'4p':0}],\
      ['Cu','Ar',{'3d':10,'4s':1,'4p':0}],\
      ['Zn','Ar',{'3d':10,'4s':2,'4p':0}],\
      ['Ga','Ar',{'3d':10,'4s':2,'4p':1}],\
      ['Ge','Ar',{'3d':10,'4s':2,'4p':2}],\
      ['As','Ar',{'3d':10,'4s':2,'4p':3}],\
      ['Se','Ar',{'3d':10,'4s':2,'4p':4}],\
      ['Br','Ar',{'3d':10,'4s':2,'4p':5}],\
      ['Kr','Ar',{'3d':10,'4s':2,'4p':6}],\
      # fifth row
      ['Rb','Kr',{'4d':0,'5s':1,'5p':0}],
      ['Sr','Kr',{'4d':0,'5s':2,'5p':0}],
      ['Y', 'Kr',{'4d':1,'5s':2,'5p':0}],
      ['Zr','Kr',{'4d':2,'5s':2,'5p':0}],
      ['Nb','Kr',{'4d':4,'5s':1,'5p':0}],
      ['Mo','Kr',{'4d':5,'5s':1,'5p':0}],
      ['Tc','Kr',{'4d':5,'5s':2,'5p':0}],
      ['Ru','Kr',{'4d':7,'5s':1,'5p':0}],
      ['Rh','Kr',{'4d':8,'5s':1,'5p':0}],
      ['Pd','Kr',{'4d':10,'5s':0,'5p':0}],
      ['Ag','Kr',{'4d':10,'5s':1,'5p':0}],
      ['Cd','Kr',{'4d':10,'5s':2,'5p':0}],
      ['In','Kr',{'4d':10,'5s':2,'5p':1}],
      ['Sn','Kr',{'4d':10,'5s':2,'5p':2}],
      ['Sb','Kr',{'4d':10,'5s':2,'5p':3}],
      ['Te','Kr',{'4d':10,'5s':2,'5p':4}],
      ['I', 'Kr',{'4d':10,'5s':2,'5p':5}],
      ['Xe','Kr',{'4d':10,'5s':2,'5p':6}],
      # sixth row
      ['Cs','Xe',{'5d':0,'6s':1,'6p':0}],
      ['Ba','Xe',{'5d':0,'6s':2,'6p':0}],
      ['Lu','Xe',{'4f':14,'5d':1,'6s':2,'6p':0}],
      ['Hf','Xe',{'4f':14,'5d':2,'6s':2,'6p':0}],
      ['Ta','Xe',{'4f':14,'5d':3,'6s':2,'6p':0}],
      ['W', 'Xe',{'4f':14,'5d':4,'6s':2,'6p':0}],
      ['Re','Xe',{'4f':14,'5d':5,'6s':2,'6p':0}],
      ['Os','Xe',{'4f':14,'5d':6,'6s':2,'6p':0}],
      ['Ir','Xe',{'4f':14,'5d':7,'6s':2,'6p':0}],
      ['Pt','Xe',{'4f':14,'5d':9,'6s':1,'6p':0}],
      ['Au','Xe',{'4f':14,'5d':10,'6s':1,'6p':0}],
      ['Hg','Xe',{'4f':14,'5d':10,'6s':2,'6p':0}], 
      ['Tl','Xe',{'4f':14,'5d':10,'6s':2,'6p':1}], 
      ['Pb','Xe',{'4f':14,'5d':10,'6s':2,'6p':2}], 
      ['Bi','Xe',{'4f':14,'5d':10,'6s':2,'6p':3}], 
      ['Po','Xe',{'4f':14,'5d':10,'6s':2,'6p':4}], 
      ['At','Xe',{'4f':14,'5d':10,'6s':2,'6p':5}], 
      ['Rn','Xe',{'4f':14,'5d':10,'6s':2,'6p':6}], 
      # seventh row
      ['Fr','Rn',{'6d':0,'7s':1,'7p':0}],
      ['Ra','Rn',{'6d':0,'7s':2,'7p':0}],
      ['Ac','Rn',{'5f':0,'6d':1,'7s':2,'7p':0}],
      ['Th','Rn',{'5f':0,'6d':2,'7s':2,'7p':0}],
      ['U', 'Rn',{'5f':3,'6d':1,'7s':2,'7p':0}],
      ['Np','Rn',{'5f':4,'6d':1,'7s':2,'7p':0}],
      ['Pu','Rn',{'5f':6,'6d':0,'7s':2,'7p':0}] ]
      
configurations={}          
for item in aux:
    el, core, occu=item
    if core!='': 
        configurations[el]=configurations[core].copy()
    else:
        configurations[el]={}        
    configurations[el].update(occu)
for key in configurations:
    config=configurations[key]    
    data[key]['configuration']=config
    data[key]['valence_number']=sum( [config[orbital] for orbital in data[key]['valence_orbitals']] )
        
if __name__=='__main__':    
    for symbol in data:
        print('X'*40)
        for d in data[symbol]:
            print(d, data[symbol][d])
    
    
   
# SVWN5-LSDA/UGBS reference data for free atomic volumes in a_0^3
# taken from: Kannemann, F. O.; Becke, A. D. J. Chem. Phys. 136, 034109 (2012)
free_volumes = {}

free_volumes['H']  =   9.194 
free_volumes['He'] =   4.481 
free_volumes['Li'] =  91.96 
free_volumes['Be'] =  61.36 
free_volumes['B']  =  49.81
free_volumes['C']  =  36.73
free_volumes['N']  =  27.63
free_volumes['O']  =  23.52
free_volumes['F']  =  19.32 
free_volumes['Ne'] =  15.95
free_volumes['Na'] = 109.4
free_volumes['Mg'] = 103.1
free_volumes['Al'] = 120.4
free_volumes['Si'] = 104.2
free_volumes['P']  =  86.78
free_volumes['S']  =  77.13
free_volumes['Cl'] =  66.37
free_volumes['Ar'] =  57.34
free_volumes['K']  = 203.1
free_volumes['Ca'] = 212.2
free_volumes['Sc'] = 183.1
free_volumes['Ti'] = 162.3
free_volumes['V']  = 143.2
free_volumes['Cr'] = 108.2
free_volumes['Mn'] = 123.1
free_volumes['Fe'] = 105.7
free_volumes['Co'] =  92.94
free_volumes['Ni'] =  83.79
free_volumes['Cu'] =  75.75
free_volumes['Zn'] =  81.18
free_volumes['Ga'] = 118.4
free_volumes['Ge'] = 116.3
free_volumes['As'] = 107.5
free_volumes['Se'] = 103.2
free_volumes['Br'] =  95.11
free_volumes['Kr'] =  87.61
free_volumes['Rb'] = 248.8
free_volumes['Sr'] = 273.7
free_volumes['Y']  = 249.2
free_volumes['Zr'] = 223.8
free_volumes['Nb'] = 175.8
free_volumes['Mo'] = 156.8
free_volumes['Tc'] = 160.0
free_volumes['Ru'] = 136.7
free_volumes['Rh'] = 127.8
free_volumes['Pd'] =  97.02
free_volumes['Ag'] = 112.8
free_volumes['Cd'] = 121.6
free_volumes['In'] = 167.9
free_volumes['Sn'] = 172.0
free_volumes['Sb'] = 165.5
free_volumes['Te'] = 163.0
free_volumes['I']  = 154.0
free_volumes['Xe'] = 146.1
free_volumes['Cs'] = 342.0
free_volumes['Ba'] = 385.8
free_volumes['La'] = 343.4
free_volumes['Ce'] = 350.3
free_volumes['Pr'] = 334.9
free_volumes['Nd'] = 322.2
free_volumes['Pm'] = 310.3
free_volumes['Sm'] = 299.5
free_volumes['Eu'] = 289.6
free_volumes['Gd'] = 216.1
free_volumes['Tb'] = 268.9
free_volumes['Dy'] = 259.8
free_volumes['Ho'] = 251.3
free_volumes['Er'] = 243.2
free_volumes['Tm'] = 235.5 
free_volumes['Yb'] = 228.3
free_volumes['Lu'] = 229.6
free_volumes['Hf'] = 210.0
free_volumes['Ta'] = 197.5
free_volumes['W']  = 183.2
free_volumes['Re'] = 174.7
free_volumes['Os'] = 164.1
free_volumes['Ir'] = 150.4
free_volumes['Pt'] = 135.8
free_volumes['Au'] = 125.3
free_volumes['Hg'] = 131.3
free_volumes['Tl'] = 185.8
free_volumes['Pb'] = 195.7
free_volumes['Bi'] = 193.0
free_volumes['Po'] = 189.1
free_volumes['At'] = 185.9
free_volumes['Rn'] = 181.1
free_volumes['Fr'] = 357.8
free_volumes['Ra'] = 407.3
free_volumes['Ac'] = 383.1
free_volumes['Th'] = 362.1
free_volumes['Pa'] = 346.6
free_volumes['U']  = 332.5
free_volumes['Np'] = 319.6
free_volumes['Pu'] = 308.1
free_volumes['Am'] = 297.4
free_volumes['Cm'] = 300.6
free_volumes['Bk'] = 275.8
free_volumes['Cf'] = 266.3
free_volumes['Es'] = 257.4
free_volumes['Fm'] = 209.7
free_volumes['Md'] = 203.2
free_volumes['No'] = 230.2
free_volumes['Lr'] = 236.9


## Occupation of valence orbitals in free atoms (non-zero occupations only)
ValOccs_lm_free = {}
ValOccs_lm_free['H']  = {'s':1.}
ValOccs_lm_free['He'] = {'s':2.}
ValOccs_lm_free['Li'] = {'s':1., 'p':0.}
ValOccs_lm_free['Be'] = {'s':2., 'p':0.}
ValOccs_lm_free['B']  = {'s':2., 'p':0.33333333}
ValOccs_lm_free['C']  = {'s':2., 'p':0.66666666}
ValOccs_lm_free['N']  = {'s':2., 'p':1.}
ValOccs_lm_free['O']  = {'s':2., 'p':1.33333333}
ValOccs_lm_free['F']  = {'s':2., 'p':1.66666666}
ValOccs_lm_free['Ne'] = {'s':2., 'p':2.}
ValOccs_lm_free['Na'] = {'s':1., 'p':0.}
ValOccs_lm_free['Mg'] = {'s':2., 'p':0.}
ValOccs_lm_free['Al'] = {'s':2., 'p':0.33333333}
ValOccs_lm_free['Si'] = {'s':2., 'p':0.66666666}
ValOccs_lm_free['P']  = {'s':2., 'p':1.}
ValOccs_lm_free['S']  = {'s':2., 'p':1.33333333}
ValOccs_lm_free['Cl'] = {'s':2., 'p':1.66666666}
ValOccs_lm_free['Ar'] = {'s':2., 'p':2.}
ValOccs_lm_free['K']  = {'s':1., 'p':0.}
ValOccs_lm_free['Ca'] = {'s':2., 'p':0.}
ValOccs_lm_free['Sc'] = {'s':2., 'd':0.2}
ValOccs_lm_free['Ti'] = {'s':2., 'd':0.4}
ValOccs_lm_free['V']  = {'s':2., 'd':0.6}
ValOccs_lm_free['Cr'] = {'s':1., 'd':1.}
ValOccs_lm_free['Mn'] = {'s':2., 'd':1.}
ValOccs_lm_free['Fe'] = {'s':2., 'd':1.2}
ValOccs_lm_free['Co'] = {'s':2., 'd':1.4}
ValOccs_lm_free['Ni'] = {'s':2., 'd':1.6}
ValOccs_lm_free['Cu'] = {'s':1., 'd':2.}
ValOccs_lm_free['Zn'] = {'s':2., 'd':2.}
ValOccs_lm_free['Ga'] = {'s':2., 'p':0.33333333}
ValOccs_lm_free['Ge'] = {'s':2., 'p':0.66666666}
ValOccs_lm_free['As'] = {'s':2., 'p':1.}
ValOccs_lm_free['Se'] = {'s':2., 'p':1.33333333}
ValOccs_lm_free['Br'] = {'s':2., 'p':1.66666666}
ValOccs_lm_free['Kr'] = {'s':2., 'p':2.}
ValOccs_lm_free['Rb'] = {'s':1., 'p':0.}
ValOccs_lm_free['Sr'] = {'s':2., 'p':0.}
ValOccs_lm_free['Y']  = {'s':2., 'p':0., 'd':0.2}
ValOccs_lm_free['Zr'] = {'s':2., 'p':0., 'd':0.4}
ValOccs_lm_free['Nb'] = {'s':1., 'p':0., 'd':0.8}
ValOccs_lm_free['Mo'] = {'s':1., 'p':0., 'd':1.}
ValOccs_lm_free['Tc'] = {'s':1., 'p':0., 'd':1.2}
ValOccs_lm_free['Ru'] = {'s':1., 'p':0., 'd':1.4}
ValOccs_lm_free['Rh'] = {'s':1., 'p':0., 'd':1.6}
ValOccs_lm_free['Pd'] = {'s':0., 'p':0., 'd':2.}
ValOccs_lm_free['Ag'] = {'s':1., 'p':0., 'd':2.}
ValOccs_lm_free['Cd'] = {'s':2., 'p':0., 'd':2.}
ValOccs_lm_free['In'] = {'s':2., 'p':0.33333333}
ValOccs_lm_free['Sn'] = {'s':2., 'p':0.66666666}
ValOccs_lm_free['Sb'] = {'s':2., 'p':1.}
ValOccs_lm_free['Te'] = {'s':2., 'p':1.33333333}
ValOccs_lm_free['I']  = {'s':2., 'p':1.66666666}
ValOccs_lm_free['Xe'] = {'s':2., 'p':2.}
ValOccs_lm_free['Cs'] = {'s':1., 'p':0.}
ValOccs_lm_free['Ba'] = {'s':2., 'p':0.}
ValOccs_lm_free['Lu'] = {'s':2., 'p':0., 'd':0.2}
ValOccs_lm_free['Hf'] = {'s':2., 'p':0., 'd':0.4}
ValOccs_lm_free['Ta'] = {'s':2., 'p':0., 'd':0.6}
ValOccs_lm_free['W']  = {'s':2., 'p':0., 'd':0.8}
ValOccs_lm_free['Re'] = {'s':2., 'p':0., 'd':1.}
ValOccs_lm_free['Os'] = {'s':2., 'p':0., 'd':1.2}
ValOccs_lm_free['Ir'] = {'s':2., 'p':0., 'd':1.4}
ValOccs_lm_free['Pt'] = {'s':1., 'p':0., 'd':1.8}
ValOccs_lm_free['Au'] = {'s':1., 'p':0., 'd':2.}
ValOccs_lm_free['Hg'] = {'s':2., 'p':0., 'd':2.}
ValOccs_lm_free['Tl'] = {'s':2., 'p':0.33333333}
ValOccs_lm_free['Pb'] = {'s':2., 'p':0.66666666}
ValOccs_lm_free['Bi'] = {'s':2., 'p':1.}
ValOccs_lm_free['Po'] = {'s':2., 'p':1.33333333}
ValOccs_lm_free['As'] = {'s':2., 'p':1.66666666}
ValOccs_lm_free['Rn'] = {'s':2., 'p':2.}


## optimal confinement parameters (optimized for band structures of homonuclear crystals)
## from M. Wahiduzzaman, et al. J. Chem. Theory Comput. 9, 4006-4017 (2013)
## general form V_conf = (r/r0)**s, r0 in Bohr
conf_parameters = {}
conf_parameters['H'] = {'r0':1.6,  's':2.2}
conf_parameters['He']= {'r0':1.4,  's':11.4} 
conf_parameters['Li']= {'r0':5.0,  's':8.2}
conf_parameters['Be']= {'r0':3.4,  's':13.2} 
conf_parameters['B'] = {'r0':3.0,  's':10.4} 
conf_parameters['C'] = {'r0':3.2,  's':8.2}
conf_parameters['N'] = {'r0':3.4,  's':13.4} 
conf_parameters['O'] = {'r0':3.1,  's':12.4} 
conf_parameters['F'] = {'r0':2.7,  's':10.6} 
conf_parameters['Ne']= {'r0':3.2,  's':15.4} 
conf_parameters['Na']= {'r0':5.9,  's':12.6} 
conf_parameters['Mg']= {'r0':5.0,  's':6.2}
conf_parameters['Al']= {'r0':5.9,  's':12.4} 
conf_parameters['Si']= {'r0':4.4,  's':12.8} 
conf_parameters['P'] = {'r0':4.0,  's':9.6}
conf_parameters['S'] = {'r0':3.9,  's':4.6} 
conf_parameters['Cl']= {'r0':3.8,  's':9.0}
conf_parameters['Ar']= {'r0':4.5,  's':15.2} 
conf_parameters['K'] = {'r0':6.5,  's':15.8} 
conf_parameters['Ca']= {'r0':4.9,  's':13.6} 
conf_parameters['Sc']= {'r0':5.1,  's':13.6} 
conf_parameters['Ti']= {'r0':4.2,  's':12.0} 
conf_parameters['V'] = {'r0':4.3,  's':13.0} 
conf_parameters['Cr']= {'r0':4.7,  's':3.6}
conf_parameters['Mn']= {'r0':3.6,  's':11.6} 
conf_parameters['Fe']= {'r0':3.7,  's':11.2} 
conf_parameters['Co']= {'r0':3.3,  's':11.0} 
conf_parameters['Ni']= {'r0':3.7,  's':2.2}
conf_parameters['Cu']= {'r0':5.2,  's':2.2}
conf_parameters['Zn']= {'r0':4.6,  's':2.2}
conf_parameters['Ga']= {'r0':5.9,  's':8.8}
conf_parameters['Ge']= {'r0':4.5,  's':13.4} 
conf_parameters['As']= {'r0':4.4,  's':5.6}
conf_parameters['Se']= {'r0':4.5,  's':3.8}
conf_parameters['Br']= {'r0':4.3,  's':6.4}
conf_parameters['Kr']= {'r0':4.8,  's':15.6} 
conf_parameters['Rb']= {'r0':9.1,  's':16.8} 
conf_parameters['Sr']= {'r0':6.9,  's':14.8} 
conf_parameters['Y'] = {'r0':5.7,  's':13.6} 
conf_parameters['Zr']= {'r0':5.2,  's':14.0} 
conf_parameters['Nb']= {'r0':5.2,  's':15.0}  
conf_parameters['Mo']= {'r0':4.3,  's':11.6} 
conf_parameters['Tc']= {'r0':4.1,  's':12.0} 
conf_parameters['Ru']= {'r0':4.1,  's':3.8}
conf_parameters['Rh']= {'r0':4.0,  's':3.4}
conf_parameters['Pd']= {'r0':4.4,  's':2.8}
conf_parameters['Ag']= {'r0':6.5,  's':2.0}
conf_parameters['Cd']= {'r0':5.4,  's':2.0}
conf_parameters['In']= {'r0':4.8,  's':13.2} 
conf_parameters['Sn']= {'r0':4.7,  's':13.4} 
conf_parameters['Sb']= {'r0':5.2,  's':3.0}
conf_parameters['Te']= {'r0':5.2,  's':3.0} 
conf_parameters['I'] = {'r0':6.2,  's':2.0}
conf_parameters['Xe']= {'r0':5.2,  's':16.2} 
conf_parameters['Cs']= {'r0':10.6, 's':13.6} 
conf_parameters['Ba']= {'r0':7.7,  's':12.0} 
conf_parameters['La']= {'r0':7.4,  's':8.6}
conf_parameters['Lu']= {'r0':5.9,  's':16.4} 
conf_parameters['Hf']= {'r0':5.2,  's':14.8} 
conf_parameters['Ta']= {'r0':4.8,  's':13.8} 
conf_parameters['W'] = {'r0':4.2,  's':8.6}
conf_parameters['Re']= {'r0':4.2,  's':13.0}
conf_parameters['Os']= {'r0':4.0,  's':8.0}
conf_parameters['Ir']= {'r0':3.9,  's':12.6}
conf_parameters['Pt']= {'r0':3.8,  's':12.8}
conf_parameters['Au']= {'r0':4.8,  's':2.0}
conf_parameters['Hg']= {'r0':6.7,  's':2.0}
conf_parameters['Tl']= {'r0':7.3,  's':2.2}
conf_parameters['Pb']= {'r0':5.7,  's':3.0}
conf_parameters['Bi']= {'r0':5.8,  's':2.6}
conf_parameters['Po']= {'r0':5.5,  's':2.2}
conf_parameters['Ra']= {'r0':7.0,  's':14.0}
conf_parameters['Th']= {'r0':6.2,  's':4.4}

for key in conf_parameters.keys():
    conf_parameters[key]['mode'] = 'general'
    

## shell resolved U-parameters (as obtained from PBE-DFT calculations)
## from M. Wahiduzzaman, et al. J. Chem. Theory Comput. 9, 4006-4017 (2013)
## using U parameter of occupied shell with highest l for unoccupied shells
U_parameters = {}
U_parameters['H'] = {'d':0.419731, 'p':0.419731, 's':0.419731}
U_parameters['He']= {'d':0.742961, 'p':0.742961, 's':0.742961}
U_parameters['Li']= {'d':0.131681, 'p':0.131681, 's':0.174131}
U_parameters['Be']= {'d':0.224651, 'p':0.224651, 's':0.270796}
U_parameters['B'] = {'d':0.296157, 'p':0.296157, 's':0.333879}
U_parameters['C'] = {'d':0.364696, 'p':0.364696, 's':0.399218}
U_parameters['N'] = {'d':0.430903, 'p':0.430903, 's':0.464356}
U_parameters['O'] = {'d':0.495405, 'p':0.495405, 's':0.528922}
U_parameters['F'] = {'d':0.558631, 'p':0.558631, 's':0.592918}
U_parameters['Ne']= {'d':0.620878, 'p':0.620878, 's':0.656414}
U_parameters['Na']= {'d':0.087777, 'p':0.087777, 's':0.165505}
U_parameters['Mg']= {'d':0.150727, 'p':0.150727, 's':0.224983}
U_parameters['Al']= {'d':0.186573, 'p':0.203216, 's':0.261285} 
U_parameters['Si']= {'d':0.196667, 'p':0.247841, 's':0.300005}
U_parameters['P'] = {'d':0.206304, 'p':0.289262, 's':0.338175}
U_parameters['S'] = {'d':0.212922, 'p':0.328724, 's':0.375610}
U_parameters['Cl']= {'d':0.214242, 'p':0.366885, 's':0.412418}
U_parameters['Ar']= {'d':0.207908, 'p':0.404106, 's':0.448703}
U_parameters['K'] = {'d':0.171297, 'p':0.081938, 's':0.136368}
U_parameters['Ca']= {'d':0.299447, 'p':0.128252, 's':0.177196}
U_parameters['Sc']= {'d':0.322610, 'p':0.137969, 's':0.189558}
U_parameters['Ti']= {'d':0.351019, 'p':0.144515, 's':0.201341}
U_parameters['V'] = {'d':0.376535, 'p':0.149029, 's':0.211913}
U_parameters['Cr']= {'d':0.312190, 'p':0.123012, 's':0.200284}
U_parameters['Mn']= {'d':0.422038, 'p':0.155087, 's':0.230740}
U_parameters['Fe']= {'d':0.442914, 'p':0.156593, 's':0.239398}
U_parameters['Co']= {'d':0.462884, 'p':0.157219, 's':0.247710}
U_parameters['Ni']= {'d':0.401436, 'p':0.106180, 's':0.235429}
U_parameters['Cu']= {'d':0.420670, 'p':0.097312, 's':0.243169}
U_parameters['Zn']= {'d':0.518772, 'p':0.153852, 's':0.271212}
U_parameters['Ga']= {'d':0.051561, 'p':0.205025, 's':0.279898}
U_parameters['Ge']= {'d':0.101337, 'p':0.240251, 's':0.304342}
U_parameters['As']= {'d':0.127856, 'p':0.271613, 's':0.330013}
U_parameters['Se']= {'d':0.165858, 'p':0.300507, 's':0.355433}
U_parameters['Br']= {'d':0.189059, 'p':0.327745, 's':0.380376}
U_parameters['Kr']= {'d':0.200972, 'p':0.353804, 's':0.404852}
U_parameters['Rb']= {'d':0.180808, 'p':0.073660, 's':0.130512}
U_parameters['Sr']= {'d':0.234583, 'p':0.115222, 's':0.164724}
U_parameters['Y'] = {'d':0.239393, 'p':0.127903, 's':0.176814}
U_parameters['Zr']= {'d':0.269067, 'p':0.136205, 's':0.189428}
U_parameters['Nb']= {'d':0.294607, 'p':0.141661, 's':0.200280}
U_parameters['Mo']= {'d':0.317562, 'p':0.145599, 's':0.209759}
U_parameters['Tc']= {'d':0.338742, 'p':0.148561, 's':0.218221}
U_parameters['Ru']= {'d':0.329726, 'p':0.117901, 's':0.212289}
U_parameters['Rh']= {'d':0.350167, 'p':0.113146, 's':0.219321}
U_parameters['Pd']= {'d':0.369605, 'p':0.107666, 's':0.225725}
U_parameters['Ag']= {'d':0.388238, 'p':0.099994, 's':0.231628}
U_parameters['Cd']= {'d':0.430023, 'p':0.150506, 's':0.251776}
U_parameters['In']= {'d':0.156519, 'p':0.189913, 's':0.257192}
U_parameters['Sn']= {'d':0.171708, 'p':0.217398, 's':0.275163}
U_parameters['Sb']= {'d':0.184848, 'p':0.241589, 's':0.294185}
U_parameters['Te']= {'d':0.195946, 'p':0.263623, 's':0.313028}
U_parameters['I'] = {'d':0.206534, 'p':0.284168, 's':0.331460}
U_parameters['Xe']= {'d':0.211949, 'p':0.303641, 's':0.349484}
U_parameters['Cs']= {'d':0.159261, 'p':0.069450, 's':0.120590}
U_parameters['Ba']= {'d':0.199559, 'p':0.105176, 's':0.149382}
U_parameters['La']= {'d':0.220941, 'p':0.115479, 's':0.160718}
U_parameters['Lu']= {'d':0.220882, 'p':0.126480, 's':0.187365}
U_parameters['Hf']= {'d':0.249246, 'p':0.135605, 's':0.200526}
U_parameters['Ta']= {'d':0.273105, 'p':0.141193, 's':0.212539}
U_parameters['W'] = {'d':0.294154, 'p':0.144425, 's':0.223288}
U_parameters['Re']= {'d':0.313288, 'p':0.146247, 's':0.233028}
U_parameters['Os']= {'d':0.331031, 'p':0.146335, 's':0.241981}
U_parameters['Ir']= {'d':0.347715, 'p':0.145121, 's':0.250317}
U_parameters['Pt']= {'d':0.363569, 'p':0.143184, 's':0.258165}
U_parameters['Au']= {'d':0.361156, 'p':0.090767, 's':0.255962}
U_parameters['Hg']= {'d':0.393392, 'p':0.134398, 's':0.272767}
U_parameters['Tl']= {'d':0.119520, 'p':0.185496, 's':0.267448}
U_parameters['Pb']= {'d':0.128603, 'p':0.209811, 's':0.280804}
U_parameters['Bi']= {'d':0.142210, 'p':0.231243, 's':0.296301}
U_parameters['Po']= {'d':0.158136, 'p':0.250546, 's':0.311976}
U_parameters['Ra']= {'d':0.167752, 'p':0.093584, 's':0.151368}
U_parameters['Th']= {'d':0.211980, 'p':0.114896, 's':0.174221}


## reference values for C6_AA for free atoms in Ha*Bohr**6
C6_ref = { 'H':6.50,   'He':1.46, \
          'Li':1387.0, 'Be':214.0,   'B':99.5,    'C':46.6,    'N':24.2,   'O':15.6,   'F':9.52,   'Ne':6.38, \
          'Na':1556.0, 'Mg':627.0,  'Al':528.0,  'Si':305.0,   'P':185.0 , 'S':134.0, 'Cl':94.6,   'Ar':64.3, \
           'K':3897.0, 'Ca':2221.0, 'Sc':1383.0, 'Ti':1044.0,  'V':832.0, 'Cr':602.0, 'Mn':552.0,  'Fe':482.0, \
          'Co':408.0,  'Ni':373.0,  'Cu':253.0,  'Zn':284.0,  'Ga':498.0, 'Ge':354.0, 'As':246.0,  'Se':210.0, \
          'Br':162.0,  'Kr':129.6,  'Rb':4691.0, 'Sr':3170.0, 'Rh':469.0, 'Pd':157.5, 'Ag':339.0,  'Cd':452.0, \
          'In':779.0,  'Sn':659.0,  'Sb':492.0,  'Te':396.0,   'I':385.0, 'Xe':285.9, 'Ba':5727.0, 'Ir':359.1, \
          'Pt':347.1,  'Au':298.0,  'Hg':392.0,  'Pb':697.0,  'Bi':571.0 }

## reference values for static polarizabilities in Bohr**3
alpha0_ref = { 'H':4.50,  'He':1.38, \
              'Li':164.2, 'Be':38.0,   'B':21.0,    'C':12.0,    'N':7.4,   'O':5.4,   'F':3.8,   'Ne':2.67, \
              'Na':162.7, 'Mg':71.0,  'Al':60.0,  'Si':37.0,   'P':25.0 , 'S':19.6, 'Cl':15.0,   'Ar':11.1, \
               'K':292.9, 'Ca':160.0, 'Sc':120.0, 'Ti':98.0,  'V':84.0, 'Cr':78.0, 'Mn':63.0,  'Fe':56.0, \
              'Co':50.0,  'Ni':48.0,  'Cu':42.0,  'Zn':40.0,  'Ga':60.0, 'Ge':41.0, 'As':29.0,  'Se':25.0, \
              'Br':20.0,  'Kr':16.8,  'Rb':319.2, 'Sr':199.0, 'Rh':56.1, 'Pd':23.68, 'Ag':50.6,  'Cd':39.7, \
              'In':75.0,  'Sn':60.0,  'Sb':44.0,  'Te':37.65,   'I':35.0, 'Xe':27.3, 'Ba':275.0, 'Ir':42.51, \
              'Pt':39.68,  'Au':36.5,  'Hg':33.9,  'Pb':61.8,  'Bi':49.02 }
              
## this is currently only dummy dictionary
R_conf = {}

## additional bulk reference data (from http://www.webelements.com)
## lattice constants in Angstroms, extends ase.data.reference_states
additional_bulk = {}
additional_bulk['C']  = {'hcp':{'a':2.464, 'c':6.711}}
additional_bulk['Ge'] = {'fcc':{'a':5.6575}}
additional_bulk['Sn'] = {'diamond':{'a':6.48920}}



#--EOF--#
