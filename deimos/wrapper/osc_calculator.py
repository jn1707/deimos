'''
Wrapper class providing a common interface to a range of oscillation solvers,
both from this project and external.

Tom Stuttard
'''

import sys, os, collections, numbers, re
import numpy as np
import matplotlib.pyplot as plt
import warnings
import healpy as hp

try:
    import nuSQUIDSpy as nsq
    from nuSQUIDSDecohPy import nuSQUIDSDecoh, nuSQUIDSDecohAtm
    NUSQUIDS_AVAIL = True
except ImportError as e:
    NUSQUIDS_AVAIL = False

from deimos.utils.constants import *
from deimos.models.decoherence.decoherence_operators import get_model_D_matrix
from deimos.density_matrix_osc_solver.density_matrix_osc_solver import DensityMatrixOscSolver, get_pmns_matrix, get_matter_potential_flav
from deimos.utils.oscillations import calc_path_length_from_coszen
from deimos.utils.coordinates import *


#
# Globals
#

DEFAULT_CALC_BASIS = "nxn"
DEFAULT_DECOHERENCE_GAMMA_BASIS = "sun"


#
# Calculator
#

class OscCalculator(object) :
    '''
    A unified interface to a range of oscillation + decoherence calculation tools.
    Allows easy comparison of methods.
    ''' 

    def __init__(self,
        tool, # Name of the underlying calculion tool
        atmospheric, # Bool indicating calculating in atmospheric parameter space (e.g. zenith instead of baseline)
        num_neutrinos=3,
        **kw
    ) :

        # Store args
        self.tool = tool
        self.atmospheric = atmospheric
        self.num_neutrinos = num_neutrinos

        # Checks
        assert self.num_neutrinos in [2,3]

        # Useful derived values
        self.num_sun_basis_vectors = self.num_neutrinos ** 2

        # Init
        if self.tool == "nusquids" :
            self._init_nusquids(**kw)
        elif self.tool == "deimos" :
            self._init_deimos(**kw)
        else :
            raise Exception("Unrecognised tool : %s" % self.tool)

        # Set some default values for parameters
        self.set_matter("vacuum")
        mass_splitting_eV2, mixing_angles_rad, deltacp, mass_tex, _, flavors_tex, _, nu_colors = get_default_neutrino_definitions(self.num_neutrinos)
        self.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp)
        self.set_mass_splittings(*mass_splitting_eV2)
        self.set_tex_labels(flavor_tex=flavors_tex, mass_tex=mass_tex)
        self.set_colors(nu_colors)
        self.set_calc_basis(DEFAULT_CALC_BASIS)
        # self.set_decoherence_D_matrix_basis(DEFAULT_DECOHERENCE_GAMMA_BASIS)


    def parse_pisa_config(self,config) :
        '''
        Parse settings from a PISA config file and apply them
        '''

        pass #TODO


    def _init_nusquids(self,
        energy_nodes_GeV=None,
        coszen_nodes=None,
        interactions=False,
        error=1.e-6,
    ) :

        assert NUSQUIDS_AVAIL, "Cannot use nuSQuIDS, not installed"

        #
        # Calculation nodes
        #

        # Set some default nodes...

        if energy_nodes_GeV is None :
            energy_nodes_GeV = np.logspace(0.,3.,num=100)
        self.energy_nodes_GeV = energy_nodes_GeV

        if self.atmospheric :
            if coszen_nodes is None :
                coszen_nodes = np.linspace(-1.,1.,num=100)
            self.coszen_nodes = coszen_nodes
        else :
            assert coszen_nodes is None, "`coszen_nodes` argument only valid in `atmospheric` mode"


        #
        # Instantiate nuSQuIDS
        #

        # Get nuSQuiDS units
        self.units = nsq.Const()

        # Get neutrino type
        # Alwys do both, not the most efficient but simplifies things
        nu_type = nsq.NeutrinoType.both 

        if self.atmospheric :

            # Instantiate nuSQuIDS atmospheric calculator
            self.nusquids = nuSQUIDSDecohAtm(
                self.coszen_nodes,
                self.energy_nodes_GeV * self.units.GeV,
                self.num_neutrinos,
                nu_type,
                interactions,
            )

            # Add tau regeneration
            # if interactions :
            #     self.nusquids.Set_TauRegeneration(True) #TODO results look wrong, disable for now and investigate

        else :

            # Instantiate nuSQuIDS regular calculator
            self.nusquids = nuSQUIDSDecoh(
                self.energy_nodes_GeV * self.units.GeV,
                self.num_neutrinos,
                nu_type,
                interactions,
            )


        #
        # Various settings
        #

        self.nusquids.Set_rel_error(error)
        self.nusquids.Set_abs_error(error)

        self.nusquids.Set_ProgressBar(False)


    def _init_deimos(self,
        **kw
    ) :

        # Init persistent state variables required
        # These are things that are passed to `calc_osc_probs` basically
        # self._decoh_D_matrix_eV = None
        self._decoh_n = None
        self._decoh_E0_eV = None
        self._calc_basis = None
        # self._decoherence_D_matrix_basis = None
        self._decoh_model_kw = None
        self._lightcone_model_kw = None
        self._sme_model_kw = None

        # Instantiate solver
        self.solver = DensityMatrixOscSolver(
            num_states=self.num_neutrinos,
            **kw
        )


    def set_matter(self, matter, **kw) :

        #
        # Vacuum
        #

        if (matter == "vacuum") or (matter is None) :

            if self.tool == "nusquids" :
                self.nusquids.Set_Body(nsq.Vacuum())

            elif self.tool == "deimos" :
                self.solver.set_matter_potential(None)


        #
        # Earth
        #

        elif matter == "earth" :

            if self.tool == "nusquids" :
                if self.atmospheric :
                    self.nusquids.Set_EarthModel(nsq.EarthAtm())
                else :
                    raise Exception("`earth` is only an option in atmospheric mode")

            elif self.tool == "deimos" :
                raise Exception("`%s` does have an Earth model" % self.tool)


        #
        # Uniform matter density
        #

        elif matter == "constant" :

            # Check required kwargs present
            assert "matter_density_g_per_cm3" in kw
            assert "electron_fraction" in kw

            if self.tool == "nusquids" :
                self.nusquids.Set_Body(nsq.ConstantDensity(kw["matter_density_g_per_cm3"], kw["electron_fraction"]))

            elif self.tool == "deimos" :
                V = get_matter_potential_flav(num_states=self.num_neutrinos, matter_density_g_per_cm3=kw["matter_density_g_per_cm3"], electron_fraction=kw["electron_fraction"], nsi_matrix=None)
                self.solver.set_matter_potential(V)



        #
        # Error handling
        #

        else :
            raise Exception("Unrecognised `matter` : %s" % matter)


        self._matter = matter


    def set_mixing_angles(self,theta12, theta13=None, theta23=None, deltacp=0.) :
        '''
        Units: radians
        '''

        if self.num_neutrinos == 2 :
            assert theta13 is None
            assert theta23 is None
        else :
            assert theta13 is not None
            assert theta23 is not None

        if self.tool == "nusquids" :
            self.nusquids.Set_CPPhase( 0, 2, deltacp ) #TODO check indices
            self.nusquids.Set_MixingAngle( 0, 1, theta12 )
            if self.num_neutrinos > 2 :
                self.nusquids.Set_MixingAngle( 0, 2, theta13 )
                self.nusquids.Set_MixingAngle( 1, 2, theta23 )

        elif self.tool == "deimos" :
            self.solver.set_mixing_angles( np.array([ t for t in [theta12,theta13,theta23] if t is not None ]), deltacp=deltacp )
            # self.solver.set_mixing_angles( -1. * np.array([ t for t in [theta12,theta13,theta23] if t is not None ]) ) #TODO


    def get_mixing_angles(self) :
        raise Exception("TODO")


    def set_deltacp(self, deltacp) :
        if self.tool == "nusquids" :
            self.nusquids.Set_CPPhase( 0, 2, deltacp )
        elif self.tool == "deimos" :
            raise Exception("Cannot set delta CP on its own for `deimos`, use `set_mixing_angles`")


    def set_mass_splittings(self, deltam21, deltam31=None) :
        '''
        # Note: deltam31 is +ve for normal ordering and -ve for inverted ordering

        Units: eV**2
        '''

        if self.num_neutrinos == 2 :
            assert deltam31 is None
        else :
            assert deltam31 is not None

        if self.tool == "nusquids" :
            self.nusquids.Set_SquareMassDifference( 1, deltam21*self.units.eV*self.units.eV )
            if deltam31 is not None :
                self.nusquids.Set_SquareMassDifference( 2, deltam31*self.units.eV*self.units.eV )

        elif self.tool == "deimos" :
            self.solver.set_mass_splittings( np.array([ dm2 for dm2 in [deltam21, deltam31] if dm2 is not None ]) )


    def get_mass_splittings(self) :
        '''
        Units: eV**2
        '''

        if self.tool == "nusquids" :
            mass_splittings_eV2 = [ self.nusquids.Get_SquareMassDifference(1)/(self.units.eV*self.units.eV) ]
            if self.num_neutrinos > 2 :
                mass_splittings_eV2.append( self.nusquids.Set_GquareMassDifference(2)/(self.units.eV*self.units.eV) )
            return tuple(mass_splittings_eV2)

        elif self.tool == "deimos" :
            return self.solver.get_mass_splittings()



    def set_std_osc(self) :
        '''
        Use standard oscillations (e.g. disable decoherence)
        '''

        if self.tool == "nusquids" :
            self.set_calc_basis(DEFAULT_CALC_BASIS)
            # self.set_decoherence_D_matrix_basis(DEFAULT_CALC_BASIS)
            self.set_decoherence_D_matrix(D_matrix_eV=np.zeros((self.num_sun_basis_vectors,self.num_sun_basis_vectors)), n=0, E0_eV=1.)
        else :
            self._decoh_model_kw = None
            self._lightcone_model_kw = None
            self._sme_model_kw = None
            self._neutrino_source_kw = None
            self.detector_coordinates = None

    def set_calc_basis(self, basis) :

        if self.tool == "nusquids" :
            assert basis == "nxn" #TOO is this correct?

        elif self.tool == "deimos" :
            self._calc_basis = basis # Store for use later

        else :
            raise Exception("`%s` does not support setting calculation basis" % self.tool)


    # def set_decoherence_D_matrix_basis(self, basis) :

    #     if self.tool == "nusquids" :
    #         assert basis == "sun"

    #     elif self.tool == "deimos" :
    #         self._decoherence_D_matrix_basis = basis # Store for use later

    #     else :
    #         raise Exception("`%s` does not support setting decoherence gamma matrix basis" % self.tool)




    #
    # Decoherence member functions
    #

    def set_decoherence_D_matrix(self,
        D_matrix_eV,
        n, # energy-dependence
        E0_eV,
    ) :
        '''
        Set the decoherence D matrix, plus energy dependence

        Definitions in arXiv:2007.00068, e.g.:
          - D matrix -> eqn 10
          - energy-dependence (steered by n and E9) -> eqn 18
        '''

        #
        # Check inputs
        #

        # If user specified the full matrix, check dimensions
        assert isinstance(D_matrix_eV, np.ndarray)


        #
        # Set values
        #

        if self.tool == "nusquids" :
            assert np.allclose(D_matrix_eV.imag, 0.), "nuSQuIDS decoherence implementation currently does not support imaginary gamma matrix"
            self.nusquids.Set_DecoherenceGammaMatrix(D_matrix_eV.real * self.units.eV)
            self.nusquids.Set_DecoherenceGammaEnergyDependence(n)
            self.nusquids.Set_DecoherenceGammaEnergyScale(E0_eV)

        elif self.tool == "deimos" :
            self._decoh_model_kw = {
                "D_matrix0_eV" : D_matrix_eV, # Put 0 in this line!
                "n" : n,
                "E0_eV" : E0_eV,
                "D_matrix_basis" : "sun" # Added this line!
            }


    def set_decoherence_model(self, model_name, **kw) :
        '''
        Set the decoherence model to be one of the pre-defined models
        '''

        from deimos.utils.model.nuVBH_interactions.nuVBH_model import get_randomize_phase_decoherence_D_matrix, get_randomize_state_decoherence_D_matrix, get_neutrino_loss_decoherence_D_matrix

        get_D_matrix_func = None

        # Check if model is one of the nuVBH models, and get the D matrix definition function if so
        if model_name == "randomize_phase" :
            get_D_matrix_func = get_randomize_phase_decoherence_D_matrix

        elif model_name == "randomize_state" :
            get_D_matrix_func = get_randomize_state_decoherence_D_matrix

        elif model_name == "neutrino_loss" :
            get_D_matrix_func = get_neutrino_loss_decoherence_D_matrix

        else :
            raise Exception("Unknown decoherence model : %s" % model_name) 

        # Check kwarks
        kw = copy.deepcopy(kw)
        assert "gamma0_eV" in kw
        gamma0_eV = kw.pop("gamma0_eV")
        assert "n" in kw
        n = kw.pop("n")
        assert "E0_eV" in kw
        E0_eV = kw.pop("E0_eV")
        assert len(kw) == 0

        # Get the correct D matrix and pass to the solver
        D_matrix_basis, D_matrix0_eV =  get_D_matrix_func(num_states=self.num_neutrinos, gamma=gamma0_eV)
        self.set_decoherence_D_matrix( D_matrix_eV=D_matrix0_eV, n=n, E0_eV=E0_eV ) #TODO what about the basis?


    #
    # Lightcone flucutation member functions
    #

    def set_lightcone_fluctuations(
        self,
        dL0_km,
        L0_km,
        E0_eV,
        n,
        m,
    ) :
        '''
        Set lightcone fluctuation model parameters
        '''

        from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix

        if self.tool == "nusquids" :
            print("NotImplemented. This is placeholder code:")
            damping_power = 2
            D_matrix = np.diag([0,1,1,0,1,1,1,1,0])
            self.nusquids.Set_DecoherenceGammaMatrix(D_matrix.real)
            self.nusquids.Set_DecoherenceGammaEnergyDependence(n)
            self.nusquids.Set_DecoherenceGammaEnergyScale(E0_eV * self.units.eV)
            self.nusquids.Set_UseLightconeFluctuations(True)
            self.nusquids.Set_mLengthDependenceIndex(m)
            self.nusquids.Set_dL0(dL0_km * self.units.km)
            self.nusquids.Set_L0LengthScale(L0_km * self.units.km)
            self.nusquids.Set_DampingPower(damping_power)


        elif self.tool == "deimos" :
            self._lightcone_model_kw = {
                "dL0_km" : dL0_km,
                "L0_km" : L0_km,
                "E0_eV" : E0_eV,
                "n" : n,
                "m" : m,
            }

    #
    # SME member functions
    #

    def set_sme(self,
        a_eV,
        c,
    ) :
        '''
        TODO
        '''

        #
        # Check inputs
        #

        assert isinstance(a_eV, np.ndarray), "a_eV should be an array"
        assert isinstance(c, np.ndarray), "c should be an array"


        #
        # Set values
        #

        if self.tool == "nusquids" :
            raise NotImplementedError()

        elif self.tool == "deimos" :
            self._sme_model_kw = {
                "a_eV" : a_eV,
                "c" : c
            }

    def set_detector_location(self,
                              # Detector location in deg
                              lat, 
                              long, 
                              height_m,
                              ) :
        # Set detector location
        self.detector_coordinates = CoordTransform(
            detector_lat = lat, 
            detector_long = long, 
            detector_height_m = height_m
            )
        
    
    def set_neutrino_source(self,
                            # Date, Time and Timezone
                            date_str,
                            # Location on the sky
                            ra_deg=None, 
                            dec_deg=None,
                            ):
        
        #
        # Set values
        #

        if self.tool == "nusquids" :
            raise NotImplementedError()

        elif self.tool == "deimos" :
            #Set date, time and location of neutrino source
            coszen_neutrino_source, altitude_neutrino_source, azimuth_neutrino_source = self.detector_coordinates.get_coszen_altitude_and_azimuth(
                ra_deg = ra_deg, 
                dec_deg = dec_deg,
                date_str = date_str
                )
            
            self._neutrino_source_kw = {
                "time_stamp" : self.detector_coordinates.parse_date_string(
                    date_str = date_str
                    ),
                # Horizontal Coordinate System
                "coszen" : coszen_neutrino_source,
                "altitude" : altitude_neutrino_source,
                "azimuth" : azimuth_neutrino_source,
                # Equatorial Coordinate System
                "ra" : ra_deg,
                "dec" : dec_deg,
                # Store date_str for skymap
                "date_str" : date_str,
                "sidereal_time" : self.detector_coordinates.get_local_sidereal_time(date_str),
                "angle_to_aries": self.detector_coordinates.get_angle_to_aries(date_str)
            }
         
            
    def calc_osc_prob(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        distance_km=None,
        coszen=None,
        nubar=False,
    ) :

        #TODO caching
        #TODO Option for different final rho to allow nu->nubar transitions

        #
        # Check inputs
        # 
        
        # Check whether SME parameters were set
        if self._sme_model_kw:
            # If SME parameters are set, assert that neutrino source was defined
            assert ( (self._neutrino_source_kw["ra"] is not None) 
                    and (self._neutrino_source_kw["dec"] is not None) 
                    and (self._neutrino_source_kw["time_stamp"] is not None)), ValueError("Right ascension, declination and time stamp of neutrino must be provided when SME parameters are set.")
            
            # Check whether both coszen and ra, dec were set
            if self.atmospheric :
                assert distance_km is None, "Must not provide `distance_km` in atmospheric mode"
                if coszen is not None:
                    # Issue a warning about ignoring the coszen argument
                    warnings.warn("The coszen argument was ignored. Zenith angle was calculated from RA and declination.")
                    coszen = self._neutrino_source_kw["coszen"]
                    
            if initial_flavor is not None :
                initial_flavor = self._get_flavor_index(initial_flavor)
        
        else:
            if self.atmospheric :
                assert ( (coszen is not None) and (distance_km is None) ), "Must provide `coszen` (and not `distance_km`) in atmospheric mode"
            else :
                assert ( (distance_km is not None) and (coszen is None) ), "Must provide `distance_km` (and not `coszen`) in non-atmospheric mode" 
    
            if initial_flavor is not None :
                initial_flavor = self._get_flavor_index(initial_flavor)


        #
        # Calculate
        #

        if self.tool == "nusquids" :
            return self._calc_osc_prob_nusquids( initial_flavor=initial_flavor, initial_state=initial_state, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar )

        if self.tool == "deimos" :
            assert initial_flavor is not None, "must provide `initial_flavor` (`initial_state` not currently supported for %s" % self.tool
            return self._calc_osc_prob_deimos( initial_flavor=initial_flavor, nubar=nubar, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen)


    def _calc_osc_prob_nusquids(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        nubar=False,
        distance_km=None,
        coszen=None,
    ) :
        '''
        Calculate oscillation probability for the model

        Returned result has following structure: [ energy, coszen, final flavor ]
        '''


        #
        # Prepare
        #

        assert not ( (initial_flavor is None) and (initial_state is None) ), "Must provide `initial_flavor` or `initial_state`"
        assert not ( (initial_flavor is not None) and (initial_state is not None) ), "Must provide `initial_flavor` or `initial_state`, not both"

        # Calculate all final state flavors
        final_flavors = self.states

        # Handle scalars vs arrays
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        if distance_km is not None :
            distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
        if coszen is not None :
            coszen = np.asarray( [coszen] if np.isscalar(coszen) else coszen )

        # Arrays must be 1D
        assert energy_GeV.ndim == 1
        if distance_km is not None :
            assert distance_km.ndim == 1
        if coszen is not None :
            assert coszen.ndim == 1

        # Handle nubar
        rho = 1 if nubar else 0

        # # Handle nubar
        # if initial_flavor is not None :
        #     if initial_flavor < 0 :
        #         assert include_nubar
        #         initial_flavor = -1* initial_flavor
        #         rho = 1
        #     else :
        #         rho = 0


        #
        # Atmospheric case
        #

        if self.atmospheric :

            randomize_atmo_prod_height = False #TODO support

            # Init results container
            results = np.full( (energy_GeV.size, coszen.size, final_flavors.size, 2 ), np.NaN )

            # Determine shape of initial state vector
            state_shape = [ self.nusquids.GetNumCos(), self.nusquids.GetNumE() ]
            state_shape.append( 2 )
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, :, rho, initial_flavor ] = 1. # dims = [ cz node, E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Set the intial state
            self.nusquids.Set_initial_state(initial_state, nsq.Basis.flavor)

            # Evolve the state
            self.nusquids.EvolveState()

            # Evaluate the flavor at each grid point to get oscillation probabilities
            for i_E,E in enumerate(energy_GeV) :
                for i_cz,cz in enumerate(coszen) :
                    for i_f,final_flavor in enumerate(final_flavors) :
                        # results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( final_flavor, cz, E*self.units.GeV )#, rho ) #TODO Add randomize prod height arg
                        results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( int(final_flavor), cz, E*self.units.GeV, rho, randomize_atmo_prod_height) #TODO add nubar

            return results


        #
        # Distance case
        #

        else :

            # Init results container
            results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size), np.NaN )
            # results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size, 2), np.NaN )

            # Determine shape of initial state vector
            state_shape = [ self.nusquids.GetNumE() ]
            state_shape.append(2)
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, rho, initial_flavor ] = 1. # dims = [ E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Loop over distance nodes
            for i_L,L in enumerate(distance_km) :

                # Set then track, taking medium into account
                if self._matter == "vacuum" :
                    self.nusquids.Set_Track(nsq.Vacuum.Track(L*self.units.km))
                elif self._matter == "constant" :
                    self.nusquids.Set_Track(nsq.ConstantDensity.Track(L*self.units.km))
                else :
                    raise Exception("Unknown body : %s" % body) 

                # Set initial flavor
                self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor ) 

                # Evolve for the track distance
                self.nusquids.EvolveState()

                # Loop over energies
                for i_e,E in enumerate(energy_GeV) :

                    # Evaluate final state flavor composition
                    for i_f, final_flavor in enumerate(final_flavors) :
                        # for rho in [0, 1] :
                        #     results[i_e,i_L,i_f,rho] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )
                        results[i_e,i_L,i_f] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )

            #TODO squeeze unused dimensions?

            return results



    def _calc_osc_prob_deimos(self,
        initial_flavor,
        energy_GeV,
        distance_km=None,
        coszen=None,
        nubar=False,
    ) :

        #
        # Prepare
        #

        # Calculate all final state flavors
        final_flavors = self.states

        # Handle scalars vs arrays
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        if distance_km is not None :
            distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
        if coszen is not None :
            coszen = np.asarray( [coszen] if np.isscalar(coszen) else coszen )

        # Arrays must be 1D
        assert energy_GeV.ndim == 1
        if distance_km is not None :
            assert distance_km.ndim == 1
        if coszen is not None :
            assert coszen.ndim == 1


        #
        # Calculate
        #

        # coszen -> L conversion (for atmospheric case)
        if self.atmospheric :
            production_height_km = 22. # common with nuSQuIDS (Although gives differing results???). TODO steerable, and defined in constants.py
            detector_depth_km = 0. # common with nuSQuIDS (Although gives differing results???). TODO steerable, and defined in constants.py
            distance_km = calc_path_length_from_coszen(cz=coszen, h=production_height_km, d=detector_depth_km)

        # DensityMatrixOscSolver doesn't like decending distance values in the input arrays,
        # and this is what you get from coszen arrays often
        flip = False
        if self._sme_model_kw:
            pass
        else:
            if distance_km[-1] < distance_km[0] : 
                flip = True
                distance_km = np.flip(distance_km)

        # Run solver
        # 'results' has shape [N energy, N distance, N flavor]
        results = self.solver.calc_osc_probs(
            E_GeV=energy_GeV,
            L_km=distance_km,
            initial_state=initial_flavor,
            initial_basis="flavor",
            nubar=nubar,
            calc_basis=self._calc_basis,
            # D_matrix_basis=self._decoherence_D_matrix_basis,
            decoh_opts=self._decoh_model_kw,
            lightcone_opts=self._lightcone_model_kw,
            sme_opts=self._sme_model_kw,
            detector_opts=self.detector_coordinates,
            neutrino_source_opts=self._neutrino_source_kw,
            verbose=False
        )

        # Handle flip in results (L dimension)
        if flip :
            results = np.flip(results, axis=1)

        return results


    def _get_flavor_index(self,flavor) :

        index = None

        if isinstance(flavor, str) :
            if flavor in ["e","nue"] :
                index = 0
            elif flavor in ["mu","numu"] :
                index = 1
            elif flavor in ["tau","nutau"] :
                index = 2

        else :
            assert flavor in [0,1,2]
            index = flavor

        assert flavor < self.num_neutrinos

        return index 

    
    def set_tex_labels(self, flavor_tex, mass_tex) :
        self.flavor_tex = flavor_tex
        self.mass_tex = mass_tex


    def set_colors(self, nu_colors) :
        self.nu_colors = nu_colors


    @property
    def states(self) :
        return np.array(range(self.num_neutrinos))

    @property
    def flavors_tex(self) :
        return [ self.get_flavor_tex(i) for i in self.states ]


    @property
    def masses_tex(self) :
        return [ self.get_mass_tex(i) for i in self.states ]


    @property
    def flavors_color(self) :
        return [ self.get_flavor_color(i) for i in self.states ]


    def get_mass_tex(self, state) :
        return self.mass_tex[state]


    def get_flavor_tex(self, flavor, nubar=False) :
        nu_tex = r"\nu"
        if nubar :
            nu_tex = r"\bar{" + nu_tex + r"}"
        if flavor is None :
            nu_tex += r"_{\rm{all}}"
        else :
            nu_tex += r"_{" + self.flavor_tex[flavor] + r"}"
        return nu_tex


    def get_flavor_color(self, flavor) :
        return self.nu_colors[flavor]


    def get_transition_prob_tex(self,initial_flavor, final_flavor, nubar=False) :
        return r"P(%s \rightarrow %s)" % ( self.get_flavor_tex(initial_flavor, nubar), self.get_flavor_tex(final_flavor, nubar) )


    @property
    def PMNS(self) :
        '''
        Return the PMNS matrix
        '''

        if self.tool == "nusquids" :
            if self.num_neutrinos == 2 :
                theta = [ self.nusquids.Get_MixingAngle(0,1) ]
            elif self.num_neutrinos == 3 :
                theta = [ self.nusquids.Get_MixingAngle(0,1), self.nusquids.Get_MixingAngle(0,2), self.nusquids.Get_MixingAngle(1,2) ]
            else :
                raise Exception("`PMNS` function only supports 2/3 flavors")
            deltacp = self.nusquids.Get_CPPhase(0,2) #TODO check indices
            return get_pmns_matrix( theta=theta, dcp=deltacp )


        elif self.tool == "deimos" :
            return self.solver.PMNS


    def plot_osc_prob_vs_distance(self, 
        # Steer physics
        initial_flavor, 
        energy_GeV, 
        distance_km=None, coszen=None, 
        nubar=False, 
        # Plotting
        fig=None, ax=None, 
        label=None, 
        title=None,
        **plot_kw
    ) :
        '''
        Compute and plot the oscillation probability, vs propagation distance
        '''

        import matplotlib.pyplot as plt

        # Handle distance vs coszen
        if self.atmospheric :
            assert coszen is not None
            dist_kw = {"coszen" : coszen}
            x = coszen
            xlabel = COSZEN_LABEL
        else :
            assert distance_km is not None
            dist_kw = {"distance_km" : distance_km}
            x = distance_km
            xlabel = DISTANCE_LABEL

        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(x, np.ndarray)
        assert np.isscalar(energy_GeV)
        assert isinstance(nubar, bool)

        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1
        if fig is None : 
            fig, ax = plt.subplots( nrows=ny, sharex=True, figsize=( 6, 7 if self.num_neutrinos == 3 else 5) )
            if title is not None :
                fig.suptitle(title) 
        else :
            assert ax is not None
            assert len(ax) == ny
            assert title is None

        # Calc osc probs
        osc_probs = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Remove energy dimension, since this is single energy
        osc_probs = osc_probs[0,...]

        # Plot oscillations to all possible final states
        for final_flavor, tex in zip(self.states, self.flavors_tex) :
            ax[final_flavor].plot( x, osc_probs[:,final_flavor], label=label, **plot_kw )
            ax[final_flavor].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )

        # Plot total oscillations to any final state
        osc_probs_flavor_sum = np.sum(osc_probs,axis=1)
        ax[-1].plot( x, osc_probs_flavor_sum, label=label, **plot_kw ) # Dimension 2 is flavor
        ax[-1].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, None, nubar) )

        # Formatting
        ax[-1].set_xlabel(xlabel)
        if label is not None :
            ax[0].legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), 
        for this_ax in ax :
            this_ax.set_xlim(x[0], x[-1])
            this_ax.set_ylim(-0.05, 1.05)
            this_ax.grid(True)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_osc_prob_vs_cozen(self, coszen, *args, **kwargs) : # Alias
        return self.plot_osc_prob_vs_distance(coszen=coszen, *args, **kwargs)



    def plot_osc_prob_vs_energy(self, 
        # Steer physics
        initial_flavor, 
        energy_GeV, 
        distance_km=None, coszen=None, 
        nubar=False, 
        final_flavor=None,
        # Plotting
        fig=None, ax=None, 
        label=None, 
        title=None,
        xscale="linear",
        ylim=None,
        **plot_kw
    ) :
        '''
        Compute and plot the oscillation probability, vs neutrino energy
        '''

        import matplotlib.pyplot as plt

        # Handle distance vs coszen
        if self.atmospheric :
            assert coszen is not None
            dist_kw = {"coszen" : coszen}
            x = coszen
        else :
            assert distance_km is not None
            dist_kw = {"distance_km" : distance_km}
            x = distance_km

        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert np.isscalar(x)
        assert isinstance(nubar, bool)
        if final_flavor is not None :
            assert isinstance(final_flavor, int)

        # User may provide a figure, otherwise make one
        ny = ( self.num_neutrinos + 1 ) if final_flavor is None else 1
        if fig is None : 
            fig, ax = plt.subplots( nrows=ny, sharex=True, figsize=( 6, 4*ny) )
            if ny == 1 :
                ax = [ax]
            if title is not None :
                fig.suptitle(title) 
        else :
            assert ax is not None
            assert len(ax) == ny
            assert title is None

        # Calc osc probs
        osc_probs = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Remove distance dimension, since this is single distance
        osc_probs = osc_probs[:,0,...]

        # Plot oscillations to all possible final states
        final_flavor_values = self.states if final_flavor is None else [final_flavor]
        for i, final_flavor in enumerate(final_flavor_values) :
            ax[i].plot( energy_GeV, osc_probs[:,final_flavor], label=label, **plot_kw )
            ax[i].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )

        # Plot total oscillations to any final state
        if len(final_flavor_values) > 1 :
            osc_probs_flavor_sum = np.sum(osc_probs,axis=1)
            ax[-1].plot( energy_GeV, osc_probs_flavor_sum, label=label, **plot_kw ) # Dimension 2 is flavor
            ax[-1].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, None, nubar) )

        # Formatting
        if ylim is None :
            ylim = (-0.05, 1.05)
        ax[-1].set_xlabel(ENERGY_LABEL)
        if label is not None :
            ax[0].legend(fontsize=10) # loc='center left', bbox_to_anchor=(1, 0.5), 
        for this_ax in ax :
            this_ax.set_xlim(energy_GeV[0], energy_GeV[-1])
            this_ax.set_ylim(ylim)
            this_ax.grid(True)
            this_ax.set_xscale(xscale)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_cp_asymmetry() :
        '''
        Plot the CP(T) asymmetry
        '''

        import matplotlib.pyplot as plt

        raise NotImplemented("TODO")


    def plot_oscillogram(
        self,
        initial_flavor,
        final_flavor,
        energy_GeV,
        coszen,
        nubar=False,
        title=None,
    ) :
        '''
        Helper function for plotting an atmospheric neutrino oscillogram
        '''

        import matplotlib.pyplot as plt
        from deimos.utils.plotting import plot_colormap, value_spacing_is_linear

        assert self.atmospheric, "`plot_oscillogram` can only be called in atmopsheric mode"

        #
        # Steerig
        #

        # Plot steering
        transition_prob_tex = self.get_transition_prob_tex(initial_flavor, final_flavor, nubar)
        continuous_map = "jet" # plasma jet
        # diverging_cmap = "seismic" # PuOr_r RdYlGn Spectral


        #
        # Compute osc probs
        #

        # Define osc prob calc settings
        calc_osc_prob_kw = dict(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=energy_GeV,
            coszen=coszen, 
        )

        # Calc osc probs 
        osc_probs = self.calc_osc_prob( **calc_osc_prob_kw )

        # Get chose flavor/rho
        osc_probs = osc_probs[:, :, final_flavor]

        #
        # Plot
        #

        # Create fig
        fig, ax = plt.subplots( figsize=(7, 6) )
        if title is not None :
            fig.suptitle(title) 

        # Plot oscillogram
        plot_colormap( ax=ax, x=energy_GeV, y=coszen, z=osc_probs, vmin=0., vmax=1., cmap=continuous_map, zlabel=r"$%s$"%transition_prob_tex )

        # Format
        xscale = "linear" if value_spacing_is_linear(energy_GeV) else "log"
        yscale = "linear" if value_spacing_is_linear(coszen) else "log"
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(energy_GeV[0], energy_GeV[-1])
        ax.set_ylim(coszen[0], coszen[-1])
        ax.set_xlabel(ENERGY_LABEL)
        ax.set_ylabel(COSZEN_LABEL)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_right_ascension_vs_energy_2D(
            self,
            # Steer physics
            initial_flavor,
            energy_GeV,
            distance_km=None, coszen=None,
            nubar=False,
            final_flavor=None,
            # Plotting
            fig=None, ax=None,
            label=None,
            title=None,
            xscale="linear",
            ylim=None,
            **plot_kw
    ):
        '''
        Make a 2D plot of oscillation probabilities vs neutrino energy (x-axis) and right ascension (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
            
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            dec_0 = np.deg2rad(self._neutrino_source_kw["dec"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\delta \sim {:.2f}$".format(dec_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
    
        # Calculate probabilities
        probabilities2d = self.calc_osc_prob(
            initial_flavor = initial_flavor,
            energy_GeV = energy_GeV,
            **dist_kw
        )
        
        # Transpose array to plot right ascension vs. energy
        probabilities2d = np. transpose(probabilities2d, (1,0,2) )
    
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(probabilities2d[:, :, i] < -0.1) or np.any(probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of oscillation probabilities outside the range [0, 1].", UserWarning)
                
            # Plot the results
            im = ax[i].imshow(
                probabilities2d[:, :, i][::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), 0, 2 * np.pi],
                cmap="RdPu",
                vmin=0,
                vmax=1.0,
            )
            ax[i].set_ylabel("Right Ascension (rad)")
            ax[i].set_xscale(xscale)
    
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < 0.9) or np.any(osc_probs_flavor_sum > 1.1):
                warnings.warn("Values outside the range [0.9, 1.1] in osc_probs_flavor_sum.", UserWarning)
            
            ax[-1].imshow(
                osc_probs_flavor_sum[::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), 0, 2 * np.pi],
                cmap="RdPu",
                # Set the minimum values for the colorbar
                vmin=0.9,  
                vmax=1.1,
            )
            ax[-1].set_ylabel("Right Ascension (rad)")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(ax[-1].get_images()[0], ax=ax[-1], label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$")
    
        plt.tight_layout()
        plt.show()

        return fig, ax, probabilities2d
    
    
    def plot_declination_vs_energy_2D(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        # Plotting
        fig=None, ax=None,
        label=None,
        title=None,
        xscale="linear",
        ylim=None,
        **plot_kw
    ):
        '''
        Make a 2D plot of oscillation probabilities vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            ra_0 = np.deg2rad(self._neutrino_source_kw["ra"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\alpha \sim {:.2f}$".format(ra_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
    
        # Calculate probabilities
        probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
    
        # Transpose array to plot declination vs. energy
        probabilities2d = np.transpose(probabilities2d, (1, 0, 2))
    
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
            
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(probabilities2d[:, :, i] < -0.1) or np.any(probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of oscillation probabilities outside the range [0, 1].", UserWarning)
              
            # Plot the results
            im = ax[i].imshow(
                probabilities2d[:, :, i][::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), -np.pi / 2, np.pi / 2],  # Assuming declination range is [-1, 1]
                cmap="RdPu",
                vmin=0,
                vmax=1.0,
            )
            ax[i].set_ylabel("Declination (rad)")
            ax[i].set_xscale(xscale)
    
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < 0.9) or np.any(osc_probs_flavor_sum > 1.1):
                warnings.warn("Values outside the range [0.9, 1.1] in osc_probs_flavor_sum.", UserWarning)
            
            ax[-1].imshow(
                osc_probs_flavor_sum[::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), -np.pi/2, np.pi/2],  # Assuming declination range is [-1, 1]
                cmap="RdPu",
                vmin=0.9,
                vmax=1.1,
            )
            ax[-1].set_ylabel("Declination (rad)")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(ax[-1].get_images()[0], ax=ax[-1], label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$")
    
        plt.tight_layout()
        plt.show()
    
        return fig, ax, probabilities2d
    
    
    def plot_declination_vs_energy_2D_diff(
    self,
    # Steer physics
    initial_flavor,
    energy_GeV,
    distance_km=None, coszen=None,
    nubar=False,
    final_flavor=None,
    # Plotting
    fig=None, ax=None,
    label=None,
    title=None,
    xscale="linear",
    ylim=None,
    **plot_kw
    ):
        '''
        Make a 2D plot of the difference of oscillation probabilities between standard osc and with SME
        vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            ra_0 = np.deg2rad(self._neutrino_source_kw["ra"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\alpha \sim {:.2f}$".format(ra_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
            
        # Calculate probabilities for the non-standard case
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities2d = sme_probabilities2d - standard_probabilities2d

        # Select a colormap
        cmap_diff = 'bwr'

        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(diff_probabilities2d[:, :, i] < -1.1) or np.any(diff_probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)
            
            # Use the custom colormap to plot the difference of probabilities
            im = ax[i].imshow(
                diff_probabilities2d[:, :, i][::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), -np.pi / 2, np.pi / 2],
                cmap=cmap_diff,
                vmin=-1.0,
                vmax=1.0,
            )
            ax[i].set_ylabel("Declination (rad)")
            ax[i].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
            
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = diff_probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < -0.1) or np.any(osc_probs_flavor_sum > 0.1):
                warnings.warn("Values outside the range [-0.1, 0.1] in osc_probs_flavor_sum.", UserWarning)
            
            ax[-1].imshow(
                osc_probs_flavor_sum[::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), -np.pi/2, np.pi/2],  # Assuming declination range is [-1, 1]
                cmap=cmap_diff,
                vmin=-0.1,
                vmax=0.1,
            )
            ax[-1].set_ylabel("Declination (rad)")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(ax[-1].get_images()[0], ax=ax[-1], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$")
    

        return fig, ax, diff_probabilities2d
    
    
    def plot_right_ascension_vs_energy_2D_diff(
    self,
    # Steer physics
    initial_flavor,
    energy_GeV,
    distance_km=None, coszen=None,
    nubar=False,
    final_flavor=None,
    # Plotting
    fig=None, ax=None,
    label=None,
    title=None,
    xscale="linear",
    ylim=None,
    **plot_kw
    ):
        '''
        Make a 2D plot of the difference of oscillation probabilities between standard osc and with SME
        vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            dec_0 = np.deg2rad(self._neutrino_source_kw["dec"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\delta \sim {:.2f}$".format(dec_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
            
        # Calculate probabilities for the non-standard case
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities2d = sme_probabilities2d - standard_probabilities2d

        # Create a custom colormap
        cmap_diff = 'bwr'
        
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(diff_probabilities2d[:, :, i] < -1.1) or np.any(diff_probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)
               
            # Use the custom colormap to plot the difference of probabilities
            im = ax[i].imshow(
                diff_probabilities2d[:, :, i][::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), 0, 2 * np.pi ],
                cmap=cmap_diff,
                vmin=-0.1,
                vmax=0.1,
            )
            ax[i].set_ylabel("Right Ascension (rad)")
            ax[i].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
            
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = diff_probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < -0.1) or np.any(osc_probs_flavor_sum > 0.1):
                warnings.warn("Values outside the range [-0.1, 0.1] in osc_probs_flavor_sum.", UserWarning)
            
            ax[-1].imshow(
                osc_probs_flavor_sum[::-1],
                aspect="auto",
                extent=[min(energy_GeV), max(energy_GeV), 0, 2 * np.pi],  # Assuming declination range is [-1, 1]
                cmap=cmap_diff,
                vmin=-0.1,
                vmax=0.1,
            )
            ax[-1].set_ylabel("Right Asencion (rad)")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(ax[-1].get_images()[0], ax=ax[-1], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$")
    

        return fig, ax, diff_probabilities2d
    

    def plot_healpix_map(
            self,
            healpix_map, 
            nside,
            title,
            cbar_label,
            cmap='viridis',
            ):
        
        #Plot in the mollview projection
        projected_map = hp.mollview(
                            map = healpix_map, 
                            title=title + "\n", 
                            cmap=cmap,
                            xsize=2000, 
                            nest=False,
                            rot=(180, 0, 0),
                            coord='C',
                            flip = 'astro',
                            min=-1,
                            max=1,
                            cbar=False,
                            return_projected_map=True,
                            )
        
        # Add meridians and parallels
        hp.graticule()
        
        # Add declination labels
        for dec in np.arange(-75, 0, 15):
            #lonlat If True, theta and phi are interpreted as longitude and latitude in degree, otherwise, as colatitude and longitude in radian
            hp.projtext(359.9, dec, "\n" + f"{dec}°   ", lonlat=True, color="black", ha="right", va="center")  
        for dec in np.arange(0, 76, 15):
            if dec == 0:
                hp.projtext(359.9, dec, r"Declination $\delta$" + "\n\n", lonlat=True, color="black", ha="right", va="center", rotation ='vertical')
                continue
            hp.projtext(359.9, dec, f"{dec}°   " + "\n", lonlat=True, color="black", ha="right", va="center")
            
        # Add the right ascension labels
        hp.projtext(359.9, 0, "24h ", lonlat=True, color="black", ha="right", va="center")
        hp.projtext(0, 0, " 0h", lonlat=True, color="black", ha="left", va="center")
        hp.projtext(180, -90, "\n\n\n12h" + "\nRight Ascension " +  r"$\alpha$", lonlat=True, color="black", ha="center", va="center")
    
        # Create an empty image plot as a mappable for the colorbar
        img = plt.imshow(projected_map, cmap=cmap, vmin=-1, vmax=1)
        cb = plt.colorbar(img, shrink=0.7)  # You can adjust the size of the colorbar using 'shrink' parameter
        cb.set_label(label=cbar_label)  
        
        # Save the plot to a file
        # Replace Greek symbols with English letters
        cbar_label = cbar_label.replace(r"$\Delta", "Delta")
        cbar_label = cbar_label.replace(r"\nu", "nu")
        cbar_label = cbar_label.replace(r"\mu", "mu")
        cbar_label = cbar_label.replace(r"\rightarrow", "to")
        cbar_label = cbar_label.replace(r"_", " ")

        # Remove any remaining LaTeX commands (e.g., "{", "}", "$")
        cbar_label = re.sub(r"{|}|\$", "", cbar_label)

        # Remove spaces and add underscores between words
        cbar_label = cbar_label.replace(" ", "_")
        title = title.replace(" ", "_")
        plt.savefig(title + cbar_label + ".png", bbox_inches='tight')
        plt.show()
        
        # Close the current plot to free memory
        plt.close()  
        
        
    def plot_osc_prob_skymap_2D(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        #Plotting
        resolution= 8,
        cmap='bwr',
        ) :
        
        '''
        Make a 2D plot of neutrino oscillation probabilities vs right ascension and declination
        for a fixed energy.
        '''
        
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
        assert resolution > 0 and (resolution & (resolution - 1)) == 0, "resolution needs to be a power of 2."
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
        
        # Generate minimal ra and dec to cover all pixels of healpy map
        # Number of pixels of healpy map
        npix = hp.nside2npix(nside=resolution)
        # Convert pixel to polar coordinates (in deg)
        right_ascension_flat, declination_flat = hp.pix2ang(nside=resolution, ipix=np.arange(npix), lonlat=True)
     
        # Store in dictionary for density_matrix_osc_solver
        self._neutrino_source_kw["ra"] = right_ascension_flat
        self._neutrino_source_kw["dec"] = declination_flat
        
        # Calculate corresponding coszen and azimuth values
        coszen_neutrino_source, azimuth_neutrino_source = self.detector_coordinates.get_coszen_and_azimuth(
            ra_deg = right_ascension_flat, 
            dec_deg = declination_flat,
            date_str = self._neutrino_source_kw["date_str"],
            )
        
        # Store in dictionary for density_matrix_osc_solver
        self._neutrino_source_kw["coszen"] = coszen_neutrino_source
        self._neutrino_source_kw["azimuth"] = azimuth_neutrino_source
        
        #Store dictionaries for later use
        neutrinos_dict = self._neutrino_source_kw
        sme_dict = self._sme_model_kw

        # Calculate probabilities with SME model
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        neutrinos_dict["ra"] = right_ascension_flat
        neutrinos_dict["dec"] = declination_flat
        
        # Set _neutrino_source_kw values again and _sme_model_kw to zero to ensure 
        # that standard_probabilities2d has the same shape as sme_probabilities2d
        self._neutrino_source_kw = neutrinos_dict
        self._sme_model_kw = {key: np.zeros_like(value) if isinstance(value, np.ndarray) else 0 for key, value in sme_dict.items()}
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities = sme_probabilities2d - standard_probabilities2d
        
        # Define the possible final states
        if final_flavor is None:
            final_flavor = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        else:
            final_flavor = [final_flavor] 
        
        # Plot each healpix map
        k = len(energy_GeV)
        ny = self.num_neutrinos if final_flavor is None else 1
        for i in range(k):
            healpix_maps_flavours = diff_probabilities[i,:,:]
            
            # Round to two significant digits
            rounded_energy = round(energy_GeV[i], -int(np.floor(np.log10(abs(energy_GeV[i]))) - 1))
            
            # Display in scientific notation
            formatted_energy = f"{rounded_energy:.2e} GeV"
        
            for j in range(ny):
                single_healpix_map = healpix_maps_flavours[:,j]
                # Check if any values are outside the range [-1, 1]
                if np.any(single_healpix_map < -1.1) or np.any(single_healpix_map > 1.1):
                    warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)

                #Plot the difference in oscillation probabilities for all flavours
                self.plot_healpix_map(
                    healpix_map=single_healpix_map, 
                    nside=resolution, 
                    title=formatted_energy,
                    cbar_label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor[j] + r"})$", 
                    cmap=cmap
                )
               
            healpix_maps_sum_flavours = np.sum(healpix_maps_flavours, axis=1)
            healpix_maps_sum_flavours = np.squeeze(healpix_maps_sum_flavours)
            
            # Check if any values are outside the range [-0.1, 0.1]
            if np.any(single_healpix_map < -0.1) or np.any(single_healpix_map > 0.1):
                warnings.warn("Values of the sum of the difference of the oscillation probabilities outside the range [-0.1, 0.1].", UserWarning)
            
            # Plot sum of flavours 
            self.plot_healpix_map(
                healpix_map=healpix_maps_sum_flavours, 
                nside=resolution, 
                title=formatted_energy,
                cbar_label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$", 
                cmap=cmap
            )
            
            
    def compare_models(
        self,
        model_defs,
        initial_flavors,
        energy_GeV,
        distance_km, 
        include_std_osc=True,
    ) :
        '''
        Compare the different models/cases specified by `model_defs`

        model_defs : list of dicts
            Each dict must/can have the following entries:
                "calc_basis" (required)
                "D_matrix_basis" (required)
                "D_matrix" (required)
                "n" (required)
                "label" (required)
                "color" (optional)
                "linestyle" (optional)
        '''

        #TODO add comparison w.r.t. energy

        # Check inputs
        #TODO

        # Plot steering
        color_scale = ColorScale("hsv", len(model_defs))

        # Output containers
        figures = []

        # Loop over initial flavors
        for initial_flavor in initial_flavors :

            fig, ax = plt.subplots( nrows=self.num_neutrinos+1, sharex=True, figsize=(6,7) )
            figures.append(fig)

            # Plot std osc
            if include_std_osc :
                self.set_std_osc()
                self.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, color="lightgrey", label="Std osc")

            # Loop over models/cases
            for i_model, model_dict in enumerate(model_defs) :

                # Plot steering
                label = model_dict["label"]
                color = model_dict["color"] if "color" in model_dict else color_scale.get(i_model)
                linestyle = model_dict["linestyle"] if "linestyle" in model_dict else "-"

                # Set physics params
                self.set_calc_basis(model_dict["calc_basis"])
                self.set_decoherence_D_matrix_basis(model_dict["D_matrix_basis"])
                self.set_decoherence_D_matrix( D_matrix_eV=model_dict["D_matrix"], n=model_dict["n"] )

                # Plot
                self.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, color=color, linestyle=linestyle, label=label ) 

            # Add long range behaviour lines
            #TODO

            # Format
            ax[-1].set_xlabel(DISTANCE_LABEL)
            ax[0].legend( loc="upper right", fontsize=10 ) #TODO put to the right of the ax
            fig.quick_format( ylim=(-0.01,1.01), legend=False )

        return figures




def define_matching_perturbation_and_lindblad_calculators(num_neutrinos=3) :
    '''
    Create DecoherenceToyModel and OscCalculator instances
    with common osc parmas, etc.

    Allows for easy comparison between the models.
    '''

    #TODO Make this more flexible by making this a member function of OscCalculator
    # which returns a compatible DecoherenceToyModel instance.


    from deimos.utils.toy_model.decoherence_toy_model import DecoherenceToyModel, get_neutrino_masses

    #
    # Define system
    #

    # Get the system definition
    mass_splittings_eV2, mixing_angles_rad, deltacp, _, _, _, _, _ = get_default_neutrino_definitions(num_neutrinos)

    #TODO store flavor labels in class, or integrate with OscCalculator

    # Assuming some (arbitrary, at least w.r.t. oscillations) lightest neutrino mass, get masses from mass splittings
    lowest_neutrino_mass_eV = 0. #1.e-3
    masses_eV = get_neutrino_masses(lowest_neutrino_mass_eV, mass_splittings_eV2)

    # Get PMNS
    PMNS = get_pmns_matrix(mixing_angles_rad, dcp=deltacp)


    #
    # Create toy model
    #

    perturbation_toy_model = DecoherenceToyModel(
        num_states=num_neutrinos,
        mass_state_masses_eV=masses_eV,
        PMNS=PMNS,
        seed=12345,
    )


    #
    # Create Lindblad lindblad_calculator
    #

    lindblad_calculator = OscCalculator(
        tool="dmos", #TODO nusquids
        atmospheric=False,
        num_neutrinos=num_neutrinos,
    )

    lindblad_calculator.set_matter("vacuum")

    lindblad_calculator.set_mass_splittings(*mass_splittings_eV2)
    lindblad_calculator.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp)

    # Handle basis choice
    lindblad_calculator.set_calc_basis("nxn")


    #
    # Checks
    #

    assert np.array_equal(perturbation_toy_model.PMNS, lindblad_calculator.PMNS)
    assert np.isclose(perturbation_toy_model.get_mass_splittings()[0], mass_splittings_eV2[0]) # 21
    if num_neutrinos > 2 :
        assert np.isclose(perturbation_toy_model.get_mass_splittings()[1], mass_splittings_eV2[1]) # 31

    return perturbation_toy_model, lindblad_calculator

