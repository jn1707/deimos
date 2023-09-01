"""
Calculate neutrino oscillation probability using density matrices
Optionally can define:
    - Matter potential
    - Decoherent open quantum system term

Useful references:
  [1] https://ocw.mit.edu/courses/chemistry/5-74-introductory-quantum-mechanics-ii-spring-2009/lecture-notes/MIT5_74s09_lec12.pdf

Tom Stuttard
"""

#TODO Would a class be better here?

import numpy as np
from scipy.integrate import solve_ivp
import warnings
try :
    from odeintw import odeintw
except :
    raise Exception("ERROR : Could not import `odeintw`, ensure it is installed and visible in your `PATH`/`PYTHONPATH` ")


from deimos.utils.constants import *
from deimos.utils.matrix_algebra import *
from deimos.models.decoherence.decoherence_operators import get_complete_sun_matrix, get_decoherence_operator_nxn_basis, get_model_D_matrix
from deimos.utils.coordinates import *


#TODO Write some tests for these functions that can be run with an arg or whatever...



#
# Globals
#

# Unit conversions
#TODO Use my natural_units.py modules
km_to_eV = 5.06773093741e9 # [km] -> [1/eV]
GeV_to_eV = 1.e9 # [GeV] -> [eV]
hbar = 6.5821195691e-16 # [eV s]


#
# Basic mathematical operations for density matrices
#

'''
Here providing the basic mathematical functions for using density matrices.
Everything is treated as complex matrices.

Neutrino-specific stuff like the PMNS matrix and mass <-> flavor basis transformations are also provided.

Have kept these outside of the actual calculator class below since they are very general and can be imported 
easily and used elsewhere.
TODO Move to common.py?
'''

def get_rho(psi) :
    """Get denisty matrix, rho, given a state (ket), psi: rho = |psi><psi| (outer product) """
    return np.outer(psi,psi).astype(np.complex128)


def get_pmns_matrix(theta, dcp=0.) :
    """ Get the PMNS matrix (rotation from mass to flavor basis)"""
    #TODO Majorana phases
    if len(theta) == 1 :
        assert (dcp is None) or np.isclose(dcp, 0.)
        # This is just the standard unitary rotation matrix in 2D
        pmns = np.array( [ [np.cos(theta[0]),np.sin(theta[0])], [-np.sin(theta[0]),np.cos(theta[0])] ], dtype=np.complex128 )
    elif len(theta) == 3 :
        # Using definition from https://en.wikipedia.org/wiki/Neutrino_oscillation
        pmns = np.array( [
            [   np.cos(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[1])*np.exp(-1.j*dcp)  ], 
            [  -np.sin(theta[0])*np.cos(theta[2]) -np.cos(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[0])*np.cos(theta[2]) -np.sin(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.sin(theta[2])*np.cos(theta[1])  ], 
            [  np.sin(theta[0])*np.sin(theta[2]) -np.cos(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    -np.cos(theta[0])*np.sin(theta[2]) -np.sin(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[2])*np.cos(theta[1])  ],
        ], dtype=np.complex128 )
    else :
        raise Exception("Only 2x2 or 3x3 PMNS matrix supported")
    assert is_unitary(pmns), "PMNS matrix must be unitary"
    return pmns


def get_mass_splittings_squared_array(mass_splittings_squared) :
    """ Get the N-1 mass splittings and fill the NxN matrix (as seen in the Hamiltonian), where N is num neutrino states """
    num_states = len(mass_splittings_squared) + 1
    assert num_states > 1
    ms_array = np.zeros( (num_states,num_states) )
    for i in range(num_states-1) :
        ms_array[i+1,i+1] = mass_splittings_squared[i]
    return ms_array


def psi_flav_to_mass_basis(psi_flav,pmns) :
    """ Rotate wavefunction from flavor to mass basis: U^dagger psi"""
    return np.dot( dagger(pmns), psi_flav ).astype(np.complex128)


def psi_mass_to_flav_basis(psi_mass,pmns) :
    """ Rotate wavefunction from mass to flavor basis: U psi"""
    return np.dot( pmns, psi_mass ).astype(np.complex128)


def rho_flav_to_mass_basis(rho_flav,pmns) :
    """ Rotate density matrix from flavor to mass basis: U^dagger rho U """
    assert is_square(rho_flav)
    assert is_square(pmns)
    return np.dot( dagger(pmns), np.dot(rho_flav,pmns) ).astype(np.complex128)


def rho_mass_to_flav_basis(rho_mass,pmns) :
    """ Rotate density matrix from mass to flavor basis: U rho U^dagger """
    assert is_square(rho_mass)
    assert is_square(pmns)
    return np.dot( pmns, np.dot(rho_mass,dagger(pmns)) ).astype(np.complex128)


def rho_flavor_prob(rho_flav,flav) :
    """ 
    Get the projection of a flavor basis vector onto a density matrix (which is itself expressed in the flavor basis).
    This gives the probability of detecting the neutrino as this flavor
    Can use either: P = <psi|rho|psi>
    Or: P(a->b) = Tr[rho_a(t) * rho_b(0)]
    """
    psi_flav = np.zeros( rho_flav.shape[0], dtype=np.complex128 ) # Create empty flvor vector, using rho shape to get num flavors
    psi_flav[flav] = 1. + 0.j # Set the desired flavor to 1
    #flav_prob = np.dot( psi_flav.T, np.dot(rho_flav,psi_flav) )
    flav_prob = np.trace( np.dot(rho_flav,get_rho(psi_flav)) )
    #assert np.isreal(flav_prob), "Falvor probability must be real, something has gone wrong (P=%s)"%flav_prob
    assert np.isclose(flav_prob.imag,0.), "Flavor probability must be real, something has gone wrong (P=%s)"%flav_prob #np.isreal doesn't work here, as struggles with ~0 values (results from numerical integration precision)
    flav_prob = flav_prob.real
    return flav_prob


def psi_flavor_prob(psi_flav,flav) : 
    """ 
    Project flavor basis vector onto current state of wavefunction to get osc prob
    P(a->b) = |<psi_b(t)|psi_a(0)>|^2
    """
    target_psi_flav = np.zeros_like( psi_flav, dtype=np.complex128 ) # Create empty flvor vector, using rho shape to get num flavors
    target_psi_flav[flav] = 1. + 0.j # Set the desired flavor to 1
    inner_prod = np.inner(psi_flav,target_psi_flav)
    flav_prob = inner_prod * np.conj(inner_prod)
    assert np.isclose(flav_prob.imag,0.), "Flavor probability must be real, something has gone wrong (P=%s)"%flav_prob #np.isreal doesn't work here, as struggles with ~0 values (results from numerical integration precision)
    flav_prob = flav_prob.real
    return flav_prob


def get_mass_projection(rho_mass,mass_index) :
    """ 
    Get the projection of a mass basis vector onto a density matrix (which is itself expressed in the mass basis).
    This is analagous to the flavor probability, but gets the current mass composition of the density matrix rather than flavor composition
    """
    psi_mass = np.zeros( rho_mass.shape[0], dtype=np.complex128 ) # Create empty flvor vector, using rho shape to get num flavors
    psi_mass[mass_index] = 1. + 0.j # Set the desired mass eigenstate to 1
    mass_proj = np.dot( psi_mass.T, np.dot(rho_mass,psi_mass) )
    assert np.isclose(mass_proj.imag,0.), "Mass projection must be real, something has gone wrong" #np.isreal doesn't work here, as struggles with ~0 values (results from numerical integration precision)
    mass_proj = mass_proj.real
    return mass_proj


#
# Matter effects
#

def get_electron_density_per_m3(matter_density_g_per_cm3, electron_fraction) :
    '''
    Calculate an SI units electron number density from :
      - the overall matter density (in g/cm^2 as it is usually reported)
      - electron fraction 
    '''
    return matter_density_g_per_cm3 * np.power(1.e-2,-3) * electron_fraction


def get_matter_potential_flav(num_states, matter_density_g_per_cm3, electron_fraction, nsi_matrix=None) :
    '''
    Calculate the matter potential (in the flavor basis) corresponding to a given matter density and electron fraction
    This assumes constant density matter
    Includes [cm^-3] -> [m^-3] conversion
    Includes [eV^-2] -> [GeV^-2] conversion

    #TODO docs
    '''

    # Get electron potential
    #TODO Want to calculate this form first principles but soemthig is going wrong...
    #TODO Just using a conversion from nuSQuIDS for now, but fix this...
    if True :
        HI_constants = 7.63254e-14 #TODO
        electron_density_to_potential = HI_constants * matter_density_g_per_cm3 * electron_fraction
    else :
        fermi_constant = 1.16639e-23 # [eV^-2]
        avogadro_number = 6.0221415e+23 # [mol^-1]
        constants = np.sqrt(2) * fermi_constant * avogadro_number # [ 1 / mol eV^2 ]
        electron_density_per_m3 = get_electron_density_per_m3(matter_density_g_per_cm3,electron_fraction)
        electron_density_to_potential = electron_density_per_m3 * constants

    # Create overall potential matrix in the flavor basis
    V = np.zeros( (num_states,num_states), dtype=np.complex128 )
    V[0,0] = 1.

    # Add NSI if requested
    if nsi_matrix is not None :
        V = V + nsi_matrix

    # Convert to potential
    V = V * electron_density_to_potential

    return V


#
# Solver
#

# def get_rho_dot(L, E, H, rho, calc_basis, L_power=None, E_power=None, **kw) :
def get_rho_dot(H, rho, calc_basis, D_matrix=None, D_matrix_basis=None) :
    '''
    Compute the time derivative of the density matrix, rho, e.g. d(rho)/dt.

    This is what we will numerically solve to get the time dependence of the system, e.g. rho(t), 
    and compute oscillation probabilities.

    Standard (mass splitting + mixing) oscillations, as well as matter effects (including NSI), 
    are contained in the Hamiltonian (H). A (potentially) non-unitary Lindblad operator D[rho]
    can also be included.

    Expansion of the various Hermitian operators involved (rho, H, D[rho]) into the SU(N) basis
    is a common method to solve this. Can expand the whole term of just the D[rho] part (have checked
    that this is equvialent).

    There are two equivalent methods in the literature, depending on the basis used:

      1) d(rho)/dt = i[H,rho] - D[rho]   - NxN basis

      2) d(rho)/dt = M * rho             - SU(N) basis   (M = M_H + M_D)

    Note that for case 1, one is still able to define your decoherence free parameters in the 
    SU(N) basis, and thus compute D[rho] in the SU(N) basis, as long as you convert D[rho] back 
    to NxN basis to calculate D[rho].

    Ultimately, return d(rho)/dt in the NxN basis (as this solver solves in that basis)
    '''

    #
    # Case 1 - NxN basis
    #

    if calc_basis == "nxn" :

        # Checks on rho (basically just sanity checking for myself that these conditio s are real proeprties of a density matrix)
        if False :
            for i, j in [ [0, 1], [0, 2], [1, 2]] :
                assert np.isclose(rho[i,j].real, rho[j,i].real), "rho[%i,%i].real != rho[%i,%i].real (%0.3g != %0.3g)" % (i,j, j,i, rho[i,j].real, rho[j,i].real)
                assert np.isclose(rho[i,j].imag, -rho[j,i].imag), "rho[%i,%i].imag != -rho[%i,%i].imag (%0.3g != %0.3g)" % (i,j, j,i, rho[i,j].imag, rho[j,i].imag)

        # Calculate the Hamiltonian contribution to the time derivative
        rho_dot = -1j * commutator(H, rho)

        # Get the decoherence operator, and subtract it from the time derivative
        if D_matrix is not None :
            assert D_matrix_basis is not None
            D_rho = get_decoherence_operator_nxn_basis(rho=rho, D_matrix=D_matrix, D_matrix_basis=D_matrix_basis).astype(np.complex128)
            rho_dot -= D_rho


    #
    # Case 2 - SU(N) basis
    #

    #TODO Am I missing the rho0 coefficient somewhow? See 2001.09250 eq 2.4 and 1711.03680 page 2 text

    elif calc_basis == "sun" :

        # Check inputs
        assert "gamma_matrix" in kw
        gamma_matrix = kw["gamma_matrix"]

        # I user omitted the 0'th rw./col, add it (it is all zeros)
        gamma_matrix_sun = get_complete_sun_matrix(num_states=rho.shape[0], sun_matrix=gamma_matrix)

        # Decompose rho
        rho_sun = convert_matrix_to_sun_decomposition_coefficients(rho)

        # Get the decoherence matrix
        M_D = -1. * gamma_matrix_sun

        # Get expanded Hamiltonian matrix
        # Have two separate methods here, need to sort this out a bit, am a bit confused...
        M_H = np.zeros_like(M_D)
        if True :
            # See 2001.09250 eq 2.7
            assert rho.shape[0] == 3, "Only implemented for 3 neutrino flvors currently"
            Delta_21 = H[1,1]
            Delta_31 = H[2,2]
            Delta_32 = Delta_31 - Delta_21
            M_H[1,2] = Delta_21 * -1.
            M_H[2,1] = Delta_21
            M_H[4,5] = Delta_31 * -1.
            M_H[5,4] = Delta_31
            M_H[6,7] = Delta_32 * -1.
            M_H[7,6] = Delta_32
        else :
            # 1711.03680 eq 8 (see also 2001.07580 eq 9)
            H_sun = convert_matrix_to_sun_decomposition_coefficients(H)
            for j in range(len(H_sun)) : #TODO general
                M_H[k,j] += 2. * np.sum([ SU3_STRUCTURE_CONSTANTS[i,j,k] * H_sun[i] for i in range(9) ])

        # Get overall matrix
        M = M_H + M_D

        # Compute d(rho)/dt
        rho_dot_sun = np.array([
            np.sum([ ( M[i,j] * rho_sun[j] ) for j in range(rho_sun.size) ]) for i in range(rho_sun.size)
        ], dtype=np.complex128)

        # Convert back to NxN
        rho_dot = convert_sun_decomposition_coefficients_to_matrix(rho_dot_sun)


    #
    # Done
    #

    else :
        raise Exception("Unknown `calc_basis` '%s', choose from ['nxn','sun']" % calc_basis)

    return rho_dot



class DensityMatrixOscSolver(object) :
    '''
    Class for calculating the time-dependence of a neutrino oscillation system,
    expressed in the density matrix formalism.

    Solution is obtained numerically, using `odetintw`

    Includes support for matter effects, decoherence.
    '''

    def __init__(self,
        num_states=NUM_STATES,
        dm2_eV2=None, # Mass spittings
        theta_rad=None, # Mixing angles
        rtol=1.e-12,
        atol=1.e-12,
        mxstep=10000000,
        **kw# Crank this up if get "Solver failed : Excess work done on this call (perhaps wrong Dfun type)" errors
    ) :

        # Store args
        self.num_states = num_states
        self.rtol = rtol
        self.atol = atol
        self.mxstep = mxstep
        
        # Check whether custom solver and method were passed
        self.custom_solver = None
        if "solver" in kw:
            self.custom_solver = kw["solver"]
            assert self.custom_solver in ["solve_ivp", "odeintw"], "solver not implemented. Use either 'solve_ivp' or 'odeintw'. "
            
        self.custom_method = None
        if "method" in kw:
            self.custom_method = kw["method"]
            
        # Init some internal states
        self.matter_potential_flav_eV = None
        if theta_rad is None :
            theta_rad =  MIXING_ANGLES_rad[:(1 if self.num_states == 2 else 3)]
        self.set_mixing_angles(theta_rad) #TODO More general function to support arbitrary num states
        if dm2_eV2 is None :
            dm2_eV2 = MASS_SPLITTINGS_eV2[:self.num_states-1] 
        self.set_mass_splittings(dm2_eV2)

        # Checks
        assert self.num_states > 1


    def set_mixing_angles(self, theta_rad, deltacp=0.) :
        self.theta_rad = theta_rad
        self.PMNS = get_pmns_matrix(self.theta_rad, dcp=deltacp)
        # self.PMNS_inv = get_pmns_matrix(-1.*self.theta_rad, dcp=deltacp) #TODO check this
        assert self.PMNS.shape == (self.num_states,self.num_states,)


    def set_mass_splittings(self,mass_splittings_eV2) :
        assert mass_splittings_eV2.size == 1 if self.num_states == 2 else 3
        self.dm2_eV2 = get_mass_splittings_squared_array(mass_splittings_eV2)
        assert self.dm2_eV2.shape == (self.num_states,self.num_states,)


    def get_mass_splittings(self) :
        ret = [ self.dm2_eV2[1, 1] ]
        if self.num_states == 3 :
            ret.append( self.dm2_eV2[2, 2] )
        return tuple(ret)


    def set_matter_potential(self, matter_potential_flav_eV) :
        '''
        Set the matter potential, in the flavor basis
        '''
        self.matter_potential_flav_eV = matter_potential_flav_eV
    
    
    def time_evolution_operator(self,H, L):
        return np.exp(-1j*H*L)
    

    def calc_osc_probs(self,
        E_GeV,
        L_km,
        initial_state,
        initial_basis="flavor", # Optionally can set initial state in mass basis
        nubar=False,
        calc_basis="nxn", # Optionally can choose which basis to perform calculation in (nxn, sun)

        # Options to be passed to the decoherence calculator
        decoh_opts=None,
        lightcone_opts=None,

        # Options to be passed to the SME calculator
        sme_opts=None,
        detector_opts= None,
        neutrino_source_opts=None,
        
        # Misc
        verbose=False,
        plot_density_matrix_evolution=False, # Flag for plotting the solved density matrix vs distance/energy
    ) :
        '''

        Returns:
            osc_probs : array
                Oscillation probability from the chosen initial flavor to all possible final 
                flavors, for each distamce and energy value requested.
                Indexing : [ E value, L value, final state flavor ]
        '''

        #
        # Check inputs
        #

        # Check basis
        assert initial_basis in BASES, "Unknown `initial_basis` '%s', choose from %s" % (initial_basis,BASES)

        # Get states
        states = np.array(range(0,self.num_states))

        # Check initial state is valid
        assert initial_state in states, "`%i` is not a valid initial flavor, choose from %s" % (initial_state,flavors)

        # Check energies and distances are arrays
        # If found a single value, turn into an array
        if np.isscalar(E_GeV) : 
            E_GeV = [E_GeV]
        E_GeV = np.asarray(E_GeV)
        assert E_GeV.ndim == 1

        if np.isscalar(L_km) : 
            L_km = [L_km]
        L_km = np.asarray(L_km)
        assert L_km.ndim == 1

        # If L array is not ascending run odeintw on each node separately 
        #(get problems if it isn't, not sure why, perhaps odeintw is requires sorted arrays?)
        calc_L_nodes_separately = False
        if np.all(np.diff(L_km) >= 0) == False:
            calc_L_nodes_separately = True

        # Will need to pass some kwargs to `derive` later, populate as I go
        rhodot_kw = {}

        # Report
        if verbose :
            print("Num flavors : %i" % self.num_states)
            print("Initial state : %i (%s basis)" % (initial_state,initial_basis))
            print("L : %s km" % L_km)
            print("E : %s GeV" % E_GeV)


        #
        # Decoherence parameters
        #

        decoh_D_matrix = None
        decoh_D_matrix_basis = None
        include_decoherence = False

        # Handle the optional decoherence effects
        if decoh_opts is not None :

            include_decoherence = True

            # User provides a D matrix and energy-depenedence
            assert "D_matrix0_eV" in decoh_opts # D matrix for E0
            assert "D_matrix_basis" in decoh_opts # The basis the D matrix is defined in (NxN or SU(N))
            assert "E0_eV" in decoh_opts
            assert "n" in decoh_opts

            # Grab the vars, handling units
            decoh_opts = copy.deepcopy(decoh_opts)
            decoh_D_matrix0 = decoh_opts.pop("D_matrix0_eV")
            #decoh_gamma0 = decoh_opts.pop("gamma0_eV") # Removed this line because decoh_gamma0 and decoh_D_matrix0 are redundant
            decoh_n = decoh_opts.pop("n")
            decoh_E0 = decoh_opts.pop("E0_eV")
            decoh_D_matrix_basis = decoh_opts.pop("D_matrix_basis") # Added this line!
            assert len(decoh_opts) == 0, "Unused decoherence arguments!?!"


        #
        # Lightcone fluctuation parameters
        #

        include_lightcone_fluctuations = False

        # Handle the optional lightcone fluctuation effects
        if lightcone_opts is not None :

            include_lightcone_fluctuations = True

            # Check args
            assert "dL0_km" in lightcone_opts
            assert "L0_km" in lightcone_opts
            assert "E0_eV" in lightcone_opts
            assert "m" in lightcone_opts
            assert "n" in lightcone_opts

            # Grab vars, handling units
            lightcone_opts = copy.deepcopy(lightcone_opts)
            lightcone_dL0 = lightcone_opts.pop("dL0_km") * km_to_eV
            lightcone_L0 = lightcone_opts.pop("L0_km") * km_to_eV
            lightcone_E0 = lightcone_opts.pop("E0_eV")
            lightcone_n = lightcone_opts.pop("n")
            lightcone_m = lightcone_opts.pop("m")

        # Check not doubling up on decoherence
        assert not (include_lightcone_fluctuations and include_decoherence), "Currently only supporting a single type of decoherence at once"


        #
        # SME (LIV) parameters
        #

        sme_a = None
        sme_c = None
        ra = None
        dec = None
        include_sme = False

        if sme_opts is not None :
            
            # To include SME parameters in calculation of the hamiltonian
            include_sme = True

            # User provides a(3) and c(4) coefficients
            assert "a_eV" in sme_opts
            assert "c" in sme_opts
            
            # User provides location of the neutrino source in the sky
            assert "ra" in neutrino_source_opts
            assert "dec" in neutrino_source_opts
            

            # Grab the vars, handling units
            sme_opts = copy.deepcopy(sme_opts)
            sme_a = sme_opts.pop("a_eV")
            sme_c = sme_opts.pop("c") # dimensionless
            
            ra = neutrino_source_opts["ra"]
            dec = neutrino_source_opts["dec"]
            
            alt_rad = neutrino_source_opts["altitude"]
            az_rad = neutrino_source_opts["altitude"]
            
            # Colatitude of the detector
            if detector_opts is not None:
                colatitude = detector_opts.get_colatitude()
            
            # Convert from degree to radians 
            ra_rad = np.deg2rad(ra)
            dec_rad = np.deg2rad(dec)
            
            a_eV_x, a_eV_y, a_eV_z = sme_a
            c_tx, c_ty, c_tz = sme_c
            
        
           
           # TODO implement the full effective hamiltonian
           # # Check if the inputs are matrices
           #  if isinstance(sme_a, np.ndarray) and isinstance(sme_c, np.ndarray):
           #      assert sme_a.shape == (4, 3, 3), "a_L^mu should be a hermitian 3x3 complex matrix"
           #      assert sme_c.shape == (4, 4, 3, 3), "c_L^mu^nu should be a hermitian 3x3 complex matrix"
           #  else:
           #  	raise TypeError("a_eV and c should be 4x4 matrices with numeric values.")
          
            
            # Allow ra and dec to be arrays abd loop through elements
            if not np.all((ra_rad >= 0) & (ra_rad <= 2 * np.pi)):
                raise ValueError("ra should be within the range [0, 2*pi].")
            
            if not np.all((dec_rad >= -np.pi / 2) & (dec_rad <= np.pi / 2)):
                raise ValueError("altitutde should be within the range [-pi/2, pi/2].")
            
            #Assert ra and dec have the same length
            if isinstance(dec_rad, np.ndarray) and isinstance(ra_rad, np.ndarray):
                assert len(dec_rad) == len(ra_rad), "ra and dec arrays provided must have the same length."
            
            #Check for additional SME arguments
            assert len(sme_opts) == 0, "Unused SME arguments!?!"


        #
        # Unit conversions
        #

        # Convert units (everything into energy-equivalent units, e.g. eV, /eV, etc)
        L = L_km * km_to_eV
        E = E_GeV * GeV_to_eV
        dm2 = self.dm2_eV2 # No conversion required


        #
        # Handle nuances of solver
        #

        
        max_L = np.max(L)
        
        # The sidereal frequency is ~ 1e-20 eV. Therefore for L<1e19 the rotation of the earth during
        # the propagation of the neutrinos can be neglected
        include_earth_rotation_during_neutrino_prop = False
        if include_sme:
            if max_L >= 1e19:
                include_earth_rotation_during_neutrino_prop = True
        
        # `odeintw` needs a number of `t` values (in this case `L`) to solve
        # These also need not to be bunched up too closely
        # Add nodes either side of the given L value to ensure good solution
        #TODO Can this be done better? what about if user defines a list of closely bnched values????
        #TODO Could I just solve for energy rather than distance in this case?
        pad_L = L.size == 1
        if self.custom_solver is None or self.custom_solver == 'odeintw':
            if pad_L :
                L_nodes = np.array( [0.1*L[0]] + L.tolist() + [2.*L[-1]] )
            else :
                L_nodes = L
        else:
            if pad_L :
                L_range = np.array([0, L[0]])
            else :
                L_range = np.array([0, max_L])

        #
        # Define physics
        #

        # Get the PMNS matrix
        PMNS = self.PMNS
        if verbose :
            print("\nPMNS matrix %s :" % (PMNS.shape,))
            print(PMNS)

        # Conjugate the PMNS matrix for antineutrinos
        if nubar :
            PMNS = np.conj(PMNS)

        # Define the mass component of the vacuum Hamiltonian
        # Do not considder the energy depenednce yet
        M = dm2

        if verbose :
            print("\nDelta mass matrix :")
            print(M)

        # Add a matter potential
        # Remember to rotate it to the mass basis
        V = self.matter_potential_flav_eV # Use V for shorthand...
        if V is not None :
            assert V.shape == M.shape, "Matter potential V shape does not match Hamiltonian shape"
            if verbose :
                print("\nMatter potential (flavor basis) :")
                print(V)
            V = rho_flav_to_mass_basis(V, PMNS) # Believe this is the correct way to rotate but need to double check (cross ref with nuSQuIDS)
            if verbose :
                print("\nMatter potential (mass basis) :")
                print(V)

        #TODO report effective mixing angles and mass splittings in matter potential


        #
        # Set initial conditions
        #

        # Get the initial state in the mass basis (we will calculate in the mass basis)
        # If user specified a flavor basis initial state, perform this conversion first

        if initial_basis == "flavor" :

            # Flavor wave function
            initial_psi_flav = np.zeros((self.num_states))
            initial_psi_flav[initial_state] = 1
            if verbose :
                print("\nInitial flavor state:")
                print(initial_psi_flav)

            # Flavor density matrix
            initial_rho_flav = get_rho(initial_psi_flav)
            if verbose :
                print("\nInitial density matrix (flavor basis):")
                print(initial_rho_flav)

            # Convert to mass basis
            initial_rho_mass = rho_flav_to_mass_basis(initial_rho_flav, PMNS)

        else :

            # Mass wave function
            initial_psi_mass = np.zeros((self.num_states))
            initial_psi_mass[initial_state] = 1
            if verbose :
                print("\nInitial mass state:")
                print(initial_psi_mass)

            # Mass density matrix
            initial_rho_mass = get_rho(initial_psi_mass)

        if verbose :
            print("\nInitial density matrix (mass basis):")
            print(initial_rho_mass)
        
        
        #Calculate osc probabilities by looping over each energy value
        def calc_osc_probs_energy_evolution(
                osc_probs_fixed_E
                ) :
            # Loop over energy values
            for E_val in E :    
    
                #
                # Set energy-dependent properties
                #
    
                # Get the vacuum Hamiltonian at this energy
                self.H = M**2 / (2. * E_val)
                
                # Add/subtract the matter hamiltonian for nu/nubar
                if V is not None :
                    self.H -= V if nubar else - V 
                    
                def amplitudes_effecitve_hamiltonian_sme(
                        alt,
                        az,
                        a_eV_x,
                        a_eV_y,
                        a_eV_z,
                        E_val,
                        L
                        ):
                    """
                    Calculate the effective Hamiltonian for the vector model of the Standard Model Extension (SME).
                
                    The function calculates the effective Hamiltonian for the "vector model" of the SME. 
                    It takes into account observer-dependent angles and amplitudes of the Lorentz-invariance violating physics
                    specified by sme_a and sme_c. The result is the effective Hamiltonian at a given location for a neutrino source 
                    located at (altitude, azimuth) on the sky and for a neutrino with energy E_val and propagating over a distance L.
                    
                    SME parameters are conventionally reported in the sun-centered celestial equatorial frame
                    For simplicity, we approximate this frame to coincide with the earth-centered celestial equatorial frame
                    Notes:
                        Angles are to be provided in rad
                    Returns: 
                        3x3 matrix in flavor basis as ndarray
                    References:
                        - arXiv:hep-ph/0406255
                        - arXiv:1010.4096
                    """
                    
                    #celestial colatitude and longitude
                    theta = np.pi/2 - alt
                    phi = np.pi - az
                    
                    #unit propagation vectors
                    NX = np.cos(colatitude) * np.sin(theta) * np.cos(phi) + np.sin(colatitude) * np.cos(theta)
                    NY = np.sin(theta) * np.sin(phi)
                    NZ = -np.sin(colatitude) * np.sin(theta) * np.cos(phi) + np.cos(colatitude) * np.cos(theta)
                    
                    # amplitudes
                    # the right ascension and declination of a neutrino are measured when detected. 
                    # To account for the rotation of the earth during propagation we need to rotate
                    # by R_z(-omega_sid*T_sid)
                    # If ra and dec of source are known change sign of As. (As = -As)
                    As0 = -NY * a_eV_x + NX * a_eV_y
                    As1 = + 2 * NY * c_tx - 2 * NX * c_ty
                    As = As0 + E_val * As1
                    
                    Ac0 = - NX * a_eV_x - NY * a_eV_y
                    Ac1 = 2 * NX * c_tx + 2 * NY * c_ty
                    Ac = Ac0 + E_val * Ac1
       
                    return As, Ac, NZ*a_eV_z
                    
                # Handle decoherence gamma param (or D matrix) energy-depenedence
                # Using the `gamma` function, but actually applying to the whole matrix rather than the individual elements (which is equivalent)
                if include_decoherence :
                    from deimos.models.decoherence.nuVBH_model import get_gamma
                    nonlocal decoh_D_matrix
                    decoh_D_matrix = get_gamma(gamma0_eV=decoh_D_matrix0, E_eV=E_val, E0_eV=decoh_E0, n=decoh_n)

    
                if self.custom_solver is None or self.custom_solver == 'odeintw':
                    #
                    # Define function to solve
                    #
        
                    # Define the time derivative of the density matrix: d(rho)/dt = -i[H,rho] (-D[rho])
                    # The additional -D[rho] term if for the case of decoherence: D[rho]_ij = rho_ij * Gamma_ij
                    # We will use this to numerically solve for rho and integrate
                    # Anything depending on `rho` or `L` must be done in here
                    
                    # Add SME to hamiltoinan
                    if include_sme:
                        self.As, self.Ac, self.const = amplitudes_effecitve_hamiltonian_sme(alt, az, a_eV_x, a_eV_y, a_eV_z, E_val, L)
                        h_eff = self.Ac + self.const
                        #rotate to mass basis
                        self.h_eff = rho_flav_to_mass_basis(h_eff, PMNS)

                        # Set the full Hamiltonian
                        self.H = self.H + self.h_eff
                        # Handle long baselines  
                        if include_earth_rotation_during_neutrino_prop:
                            self.As = rho_flav_to_mass_basis(self.As, PMNS)
                            self.Ac = rho_flav_to_mass_basis(self.Ac, PMNS)
                            self.const = rho_flav_to_mass_basis(self.const, PMNS)
                        
                    else:
                        self.H = self.H
                        
                    def derive(rho, L, decoh_D_matrix_basis=None, decoh_D_matrix=None):
        
                        # Handle lightcone fluctuations here (since depends on L)
                        if include_lightcone_fluctuations :
                            from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix
                            decoh_D_matrix_basis, decoh_D_matrix = get_lightcone_decoherence_D_matrix(num_states=self.num_states, H=self.H, E=E_val, L=L, m=lightcone_m, n=lightcone_n, dL0=lightcone_dL0, L0=lightcone_L0, E0=lightcone_E0) 
                        
                        # Handle long baselines
                        if include_earth_rotation_during_neutrino_prop:
                        
                            # Sidereal frequency in [eV]
                            sid_day = 86160 #[s]
                            omega_sid = 2*np.pi/sid_day*hbar # [eV]
                            
                            # Add SME terms to Hamiltonian                         
                            h_eff = self.As * np.sin(omega_sid*L) + self.Ac * np.cos(omega_sid*L) + self.const

                            self.H = self.H + h_eff
                            
                            # Compute the time derivative of the system
                            result = get_rho_dot(H=self.H, rho=rho, calc_basis=calc_basis, D_matrix=decoh_D_matrix, D_matrix_basis=decoh_D_matrix_basis)
                            return result
                        
                        else:
                            return get_rho_dot(H=self.H, rho=rho, calc_basis=calc_basis, D_matrix=decoh_D_matrix, D_matrix_basis=decoh_D_matrix_basis)
        
        
                    #
                    # Calculate oscillation probabilities
                    #
        
                    #TODO Can I solve directly for E if have range of E values but only a single L value, rathe than the hacks in place now?
        
                    # Solve matrix ODE to get rho (mass basis) for each L
                    #print("\nSolving...")
                    solved_rho_mass, infodict = odeintw(
                        derive, # d(rho)/dL
                        initial_rho_mass, # rho(0)
                        L_nodes, # L
                        args=(decoh_D_matrix_basis, decoh_D_matrix),
                        full_output=True,
                        rtol=self.rtol,
                        atol=self.atol,
                        #mxstep=self.mxstep,
                    )
                    assert infodict["message"] == "Integration successful.", "Solver failed : %s" % infodict["message"]
                    #print("Solver completed successfully")
        
                    # Track the entropy of the system
                    #TODO
        
                    # Remove the extra nodes added for solver stability
                    if pad_L :
                        mask = np.array( [False] + ([True]*L.size) + [False], dtype=bool )
                        solved_rho_mass = solved_rho_mass[mask]
        
                    # Convert solutions to flavor basis
                    solved_rho_flav = np.array([ rho_mass_to_flav_basis(rm, PMNS) for rm in solved_rho_mass ])
                    
                    # Calculate time evolution operator
                    # print(self.H.shape)
                    # print("L: ",L)
                    # time_evol_operator = self.time_evolution_operator(H = self.H, L = L)
                    # print("Time evol operator: ", time_evol_operator)
                    # initial_psi_mass = psi_flav_to_mass_basis(initial_psi_flav, PMNS)
                    # initial_psi_mass_2 = np.dot( dagger(PMNS), initial_psi_flav )
                    # print("Initial Flavor vector: ",initial_psi_flav)
                    # print("PMNS matrix: ",PMNS)
                    # print("Initial Mass vector: ", initial_rho_mass, get_rho(initial_psi_mass))
                    # final_psi_mass = np.dot(time_evol_operator, initial_psi_mass)
                    # print("Final Mass vector: ", final_psi_mass)
                    # final_psi_flav = psi_mass_to_flav_basis(final_psi_mass, PMNS)
                    # print("Final Flavor vector: ", final_psi_flav)
                
                # Solve-ivp for more flexible calcuations 
                else:
                    #
                    # Define function to solve
                    #
        
                    
                    # Add SME to hamiltoinan
                    if include_sme:
                        # Caculate the LIV and CPT violating contribution to the Hamiltonian
                        self.As, self.Ac, self.const = amplitudes_effecitve_hamiltonian_sme(alt, az, a_eV_x, a_eV_y, a_eV_z, E_val, L)
                        h_eff = self.Ac #+ self.const
                        # Rotate to mass basis
                        self.h_eff = rho_flav_to_mass_basis(h_eff, PMNS)
                        
                        # Set the full Hamiltonian
                        self.H = self.H + self.h_eff
                        
                        # Handle long baselines  
                        if include_earth_rotation_during_neutrino_prop:
                            self.As = rho_flav_to_mass_basis(self.As, PMNS)
                            self.Ac = rho_flav_to_mass_basis(self.Ac, PMNS)
                            self.const = rho_flav_to_mass_basis(self.const, PMNS)
                        
                    else:
                        self.H = self.H
                        
                    def derive(L,rho, decoh_D_matrix_basis=None, decoh_D_matrix=None):
                        
                        # Handle lightcone fluctuations here (since depends on L)
                        if include_lightcone_fluctuations :
                            from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix
                            decoh_D_matrix_basis, decoh_D_matrix = get_lightcone_decoherence_D_matrix(num_states=self.num_states, H=self.H, E=E_val, L=L, m=lightcone_m, n=lightcone_n, dL0=lightcone_dL0, L0=lightcone_L0, E0=lightcone_E0) 
        
                        # Handle long baselines                
                        if include_earth_rotation_during_neutrino_prop:
                        
                            # Sidereal frequency in [eV]
                            sid_day = 86160 #[s]
                            omega_sid = 2*np.pi/sid_day*hbar # [eV]
                            
                            # Add SME terms to Hamiltonian                         
                            h_eff = self.As * np.sin(omega_sid*L) + self.Ac * np.cos(omega_sid*L) + self.const
                            self.H = self.H + h_eff
                            
                            result = get_rho_dot(H=self.H, rho=rho.reshape(3,3), calc_basis=calc_basis, D_matrix=decoh_D_matrix, D_matrix_basis=decoh_D_matrix_basis).flatten()
                            return result
                        else:
                             return get_rho_dot(H=self.H, rho=rho.reshape(3,3), calc_basis=calc_basis, D_matrix=decoh_D_matrix, D_matrix_basis=decoh_D_matrix_basis).flatten()
        
        
                    #
                    # Calculate oscillation probabilities
                    #
        
                    #TODO Can I solve directly for E if have range of E values but only a single L value, rathe than the hacks in place now?
                    if self.custom_method is not None:
                        use_method = self.custom_method
                    else:
                        use_method = 'RK45'
                    
                    solved_rho_mass = solve_ivp(
                        derive, # d(rho)/dL
                        L_range, # range for which to solve ODE
                        initial_rho_mass.flatten(), # rho(0)
                        t_eval=L, # return solution for specific point
                        method=use_method,
                        args=(decoh_D_matrix_basis, decoh_D_matrix),
                        rtol=self.rtol,
                        atol=self.atol,  # atol + rtol * abs(y)
                        #mxstep=self.mxstep,
                    )
                    
                    assert solved_rho_mass.success == True, print(solved_rho_mass.message)
                    solved_rho_mass = np.transpose(solved_rho_mass.y).reshape(len(L),3, 3)
        
                    # Track the entropy of the system
                    #TODO
        
                    # Convert solutions to flavor basis
                    solved_rho_flav = np.array([ rho_mass_to_flav_basis(rm, PMNS) for rm in solved_rho_mass ])
                    
                    #TODO Remove, clean up, or add to `test`
                    
                    
                if False :
                    if verbose :
                        print("\nSolved rho (flavor basis):")
                        for r in solved_rho_flav : 
                            print(r)
                
                #
                # Get oscillation probabilties for each final state flavor
                #
               
                osc_probs = np.full( (len(L),len(states),), np.NaN ) # Indexing [L value, final state flavor]
                for i_L in range(0,len(L)) :
                    for i_f,final_flav in enumerate(states):
                        osc_probs[i_L,i_f] = rho_flavor_prob(solved_rho_flav[i_L,...],final_flav)
                        
                # Store the results for each energy
                osc_probs_fixed_E.append(osc_probs)
    
    
                #
                # Plot density matrix evolution
                #
    
                if plot_density_matrix_evolution :
    
                    assert np.isscalar(E) or (len(E) == 1)
                    assert len(L_km) > 1
    
                    rho_mass_fig = Figure(nx=self.num_states, ny=self.num_states, title=r"$\rho_{\rm{mass}}$", sharex=True, sharey=True, figsize=(9,7), row_headings=True, col_headings=True) 
                    rho_flav_fig = Figure(nx=self.num_states, ny=self.num_states, title=r"$\rho_{\rm{flavor}}$", sharex=True, sharey=True, figsize=(9,7), row_headings=True, col_headings=True) 
                    figs = [rho_mass_fig, rho_flav_fig]
                    for i in range(self.num_states) :
    
                        for fig in figs :
                            fig.set_row_heading(i, str(i) )
    
                        for j in range(self.num_states) :
    
                            if i == 0 :
                                for fig in figs :
                                    fig.set_col_heading(j, str(j) )
    
                            num_L = solved_rho_mass.shape[0]
    
                            rho_mass_ij = np.array([ solved_rho_mass[k,i,j] for k in range(num_L) ], dtype=np.complex128)
                            rho_flav_ij = np.array([ solved_rho_flav[k,i,j] for k in range(num_L) ], dtype=np.complex128)
    
                            rho_mass_ax = rho_mass_fig.get_ax(x=i, y=j)
                            rho_flav_ax = rho_flav_fig.get_ax(x=i, y=j)
    
                            rho_mass_ax.plot(L_km, rho_mass_ij.real, color="red", linestyle="-", zorder=5)
                            rho_mass_ax.plot(L_km, rho_mass_ij.imag, color="orange", linestyle="--", zorder=6)
    
                            rho_flav_ax.plot(L_km, rho_flav_ij.real, color="blue", linestyle="-", zorder=5)
                            rho_flav_ax.plot(L_km, rho_flav_ij.imag, color="lightblue", linestyle="--", zorder=6)
    
                            for ax in [rho_mass_ax, rho_flav_ax] :
                                ax.axhline(0., color="black", linestyle="--", linewidth=1, zorder=4)
                                ax.axhline(1./float(self.num_states), color="black", linestyle="-.", linewidth=1, zorder=4)
    
                    for fig in figs :
                        fig.quick_format( ylim=(-1., +1.), xlim=(L_km[0],L_km[-1]) )
                    rho_mass_fig.save("rho_mass.png")
                    rho_flav_fig.save("rho_flav.png")


            #
            # Done
            #
          
        
        #
        # Loop over energies
        #

        osc_prob_results = []
        
        # Loop through all pairs of ra and dec values (if provided) because h_eff depends on ra and dec 
        #and gets the same shape as them (n,). Therefore it cannot be broadcast together with H which has shape (3,3)
        if include_sme and isinstance(ra, np.ndarray):
            
            for i, (alt, az) in enumerate(zip(alt_rad, az_rad)):
                # Adjust input depending on solver 
                if self.custom_solver == 'solve_ivp':
                    L_val = L
                    L_range = np.array([0,L[i]])
                    L = np.asarray([L[i]])
                else:
                    pad_L = True
                    L_val = L
                    L_nodes = np.array( [0] + [L[i]] + [2.*L[i]] )
                    L = np.asarray([L[i]])
                    
                energies = []
                calc_osc_probs_energy_evolution(osc_probs_fixed_E = energies)
                
                osc_prob_results.append(energies)
                L = L_val
            
            # Bring array into the same shape and form as without SME options
            osc_prob_results = np.asarray(osc_prob_results)
            osc_prob_results = np.squeeze(osc_prob_results, axis =2)
            osc_prob_results = np. transpose(osc_prob_results, (1,0,2) )
        
        # If L_km is not ascending solve for each L_node separately
        elif calc_L_nodes_separately == True:
            for i in range(len(L)):
                # Adjust input depending on solver 
                if self.custom_solver == 'solve_ivp':
                    L_val = L
                    L_range = np.array([0,L[i]])
                    L = np.asarray([L[i]])
                else:
                    pad_L = True
                    L_val = L
                    L_nodes = np.array( [0] + [L[i]] + [2.*L[i]] )
                    L = np.asarray([L[i]])
                
                energies = []
                calc_osc_probs_energy_evolution(osc_probs_fixed_E = energies)
                
                osc_prob_results.append(energies)
                L = L_val
            
            # Bring array into the same shape and form as without SME options
            osc_prob_results = np.asarray(osc_prob_results)
            
            osc_prob_results = np.squeeze(osc_prob_results, axis =2)
            osc_prob_results = np. transpose(osc_prob_results, (1,0,2) )
        
        elif include_sme:
            alt = alt_rad
            az = az_rad
            calc_osc_probs_energy_evolution(osc_probs_fixed_E = osc_prob_results)
            
        else:
            calc_osc_probs_energy_evolution(osc_probs_fixed_E = osc_prob_results)
        
        # Numpy-ify
        osc_prob_results = np.asarray(osc_prob_results) # Indexing [E value, L value, final state flavor]

  
        if verbose :
            print("\nOscillation probabilities calculated")
            
        return osc_prob_results
        


#
# Test the code
#

def tests() :

    # This tests the how the solver works for small numbers of L values

    fig = Figure()
    E_GeV = np.linspace(1.,5.,num=100)
    initial_state = 1
    initial_basis = "flavor"
    for i,L_km in enumerate([ [1299.,1300.,1301.], [100.,1300.,2000.], [100.,1299.,1300.,1301.,2000.] ]) :
        osc_probs = calc_osc_probs( E_GeV=E_GeV,
                                    initial_state=initial_state,
                                    initial_basis=initial_basis,
                                    L_km=L_km)
        center = ( len(L_km) - 1 ) / 2 
        print(center)
        osc_probs = osc_probs[:,center,:]
        fig.get_ax().plot( E_GeV, osc_probs[:,1], linestyle="--", label=str(i), alpha=0.2 )
    fig.get_ax().grid(True)
    fig.get_ax().legend()

    dumpFiguresToPDF("osc_solver_tests.pdf")


run_tests = False
if run_tests : 
    test()

    # TODO trace
    #TODO rho^2 = rho



#
# Main function
#

if __name__ == "__main__" :


    '''
    Show some basic oscillation probabilities using these tools
    '''

    # Control numpy printing
    #np.set_printoptions(suppress=True,precision=2)

    # Create solver (using default settings)
    solver = DensityMatrixOscSolver( num_states=3 )


    #
    # DUNE
    #
    
    # Define physics
    E_GeV = np.linspace(0.6,10.,num=1000)
    L_km = np.array([1300.])
    initial_state = 1

    # Calculate oscillation probability
    osc_probs = solver.calc_osc_probs( 
        E_GeV=E_GeV,
        initial_state=initial_state,
        L_km=L_km,
    )

    # Squeeze out the single-valued distance dimension
    osc_probs = np.squeeze(osc_probs)

    # Plot
    fig = Figure( ny=2, title="DUNE")
    fig.get_ax(y=0).plot( E_GeV,osc_probs[:,0], color="red", linestyle="-", label=r"$%s \rightarrow %s$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[0]) )
    fig.get_ax(y=1).plot( E_GeV,osc_probs[:,1], color="blue", linestyle="-", label=r"$%s \rightarrow %s$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[1]) )
    fig.get_ax(y=-1).set_xlabel(r'$E$ [GeV]')
    for ax in fig.get_all_ax() :
        ax.set_ylabel('Probability')
        ax.grid(True)
        ax.legend()
    fig.tight_layout()


    #
    # Daya Bay
    #

    # See http://www1.phys.vt.edu/~dayabay/about.html

    L_km = np.logspace(-1,2,num=1000)
    E_GeV = np.array([4.e-3])
    initial_state = 0
    
    # Calculate oscillation probability
    osc_probs = solver.calc_osc_probs(
        E_GeV=E_GeV,
        initial_state=initial_state,
        L_km=L_km,
    )

    # Squeeze out the single-valued energy dimension
    osc_probs = np.squeeze(osc_probs)

    # Plot
    fig = Figure( ny=1, title="Daya Bay")
    fig.get_ax(y=0).plot( L_km,osc_probs[:,0], color="red", linestyle="-", label=r"$%s \rightarrow %s$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[0]) )
    fig.get_ax(y=-1).set_xlabel(r'$L$ [km]')
    for ax in fig.get_all_ax() :
        ax.set_xscale('log')
        ax.set_ylim(0.,1.1)
        ax.set_ylabel('Probability')
        ax.grid(True)
        ax.legend()
    fig.tight_layout()


    #
    # Atmospheric neutrinos
    #

    # Define neutrinos
    L_km = np.linspace(0.,EARTH_DIAMETER_km,num=100)
#        E_GeV = np.linspace(1.,50.,num=100)
    E_GeV = np.array([10.,25.,100.])
    initial_state = 1
    
    # Calculate oscillation probability
    osc_probs = solver.calc_osc_probs(
        E_GeV=E_GeV,
        initial_state=initial_state,
        L_km=L_km,
    )

    # Plot
    fig = Figure( ny=1, title="Atmospheric")
    for i_E in range(0,E_GeV.size) :
        fig.get_ax(y=0).plot( L_km,osc_probs[i_E,:,1], linestyle="-", label=r"$E = %0.3g$ GeV"%E_GeV[i_E] )
    fig.get_ax(y=-1).set_xlabel(r'$L$ [km]')
    for ax in fig.get_all_ax() :
        ax.set_ylim(0.,1.1)
        ax.set_ylabel (r"$P( %s \rightarrow %s)$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[1]) )
        ax.grid(True)
        ax.legend()
    fig.tight_layout()


    #
    # Done
    #

    #Dump figures to PDF
    print("\nGenerating PDF...")
    dump_figures_to_pdf( replace_file_ext(__file__,".pdf") )

