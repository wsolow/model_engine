"""Implementations of the WOFOST waterbalance modules for simulation
of NPK limited production

Written by: Will Solow, 2025
"""
from datetime import date

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class NPK_Soil(TensorModel):
    """A simple module for soil N/P/K dynamics.
    """

    _NSOILI = Tensor(-99.) 
    _PSOILI = Tensor(-99.) 
    _KSOILI = Tensor(-99.) 

    
    _FERT_N_SUPPLY = Tensor(0.)
    _FERT_P_SUPPLY = Tensor(0.)
    _FERT_K_SUPPLY = Tensor(0.)

    class Parameters(ParamTemplate):
        NSOILBASE    = Tensor(-99.)  
        NSOILBASE_FR = Tensor(-99.)  

        PSOILBASE    = Tensor(-99.)  
        PSOILBASE_FR = Tensor(-99.)  
        
        KSOILBASE    = Tensor(-99.)  
        KSOILBASE_FR = Tensor(-99.)  

        
        NAVAILI = Tensor(-99.)
        PAVAILI = Tensor(-99.)
        KAVAILI = Tensor(-99.)

        
        NMAX = Tensor(-99.)
        PMAX = Tensor(-99.)
        KMAX = Tensor(-99.)

        
        BG_N_SUPPLY = Tensor(-99.)
        BG_P_SUPPLY = Tensor(-99.)
        BG_K_SUPPLY = Tensor(-99.)

        
        RNSOILMAX = Tensor(-99.)
        RPSOILMAX = Tensor(-99.)
        RKSOILMAX = Tensor(-99.)

        
        RNABSORPTION = Tensor(-99.)
        RPABSORPTION = Tensor(-99.)
        RKABSORPTION = Tensor(-99.)

        
        RNPKRUNOFF = TensorAfgenTrait()

    class StateVariables(StatesTemplate):
        SURFACE_N = Tensor(-99.) 
        SURFACE_P = Tensor(-99.) 
        SURFACE_K = Tensor(-99.) 

        TOTN_RUNOFF = Tensor(-99.) 
        TOTP_RUNOFF = Tensor(-99.) 
        TOTK_RUNOFF = Tensor(-99.) 
        
        NSOIL = Tensor(-99.)  
        PSOIL = Tensor(-99.)  
        KSOIL = Tensor(-99.)  

        NAVAIL = Tensor(-99.)  
        PAVAIL = Tensor(-99.)  
        KAVAIL = Tensor(-99.)  

        TOTN = Tensor(-99.) 
        TOTP = Tensor(-99.) 
        TOTK = Tensor(-99.) 
      
    class RateVariables(RatesTemplate):
        RNSOIL = Tensor(-99.)
        RPSOIL = Tensor(-99.)
        RKSOIL = Tensor(-99.)
        
        RNAVAIL = Tensor(-99.)
        RPAVAIL = Tensor(-99.)
        RKAVAIL = Tensor(-99.)

        
        FERT_N_SUPPLY = Tensor()
        FERT_P_SUPPLY = Tensor()
        FERT_K_SUPPLY = Tensor()

        
        RRUNOFF_N = Tensor(-99.)
        RRUNOFF_P = Tensor(-99.)
        RRUNOFF_K = Tensor(-99.)

        
        RNSUBSOIL = Tensor(-99.)
        RPSUBSOIL = Tensor(-99.)
        RKSUBSOIL = Tensor(-99.)

    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param cropdata: dictionary with WOFOST cropdata key/value pairs
        """

        self.params = self.Parameters(parvalues)
        
        
        p = self.params
        self._NSOILI = p.NSOILBASE
        self._PSOILI = p.PSOILBASE
        self._KSOILI = p.KSOILBASE
        
        self.states = self.StateVariables(
            NSOIL=p.NSOILBASE, PSOIL=p.PSOILBASE, KSOIL=p.KSOILBASE,
            NAVAIL=p.NAVAILI, PAVAIL=p.PAVAILI, KAVAIL=p.KAVAILI, 
            TOTN=0., TOTP=0., TOTK=0., SURFACE_N=0, SURFACE_P=0, SURFACE_K=0, 
            TOTN_RUNOFF=0, TOTP_RUNOFF=0, TOTK_RUNOFF=0)
        
        self.rates = self.RateVariables()
        
    
    def calc_rates(self, day:date, drv):
        """Compute Rates for model"""
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        
        r.FERT_N_SUPPLY = self._FERT_N_SUPPLY
        r.FERT_P_SUPPLY = self._FERT_P_SUPPLY
        r.FERT_K_SUPPLY = self._FERT_K_SUPPLY

        self._FERT_N_SUPPLY = 0.
        self._FERT_P_SUPPLY = 0.
        self._FERT_K_SUPPLY = 0.

        
        r.RRUNOFF_N = s.SURFACE_N * p.RNPKRUNOFF(k.DTSR)
        r.RRUNOFF_P = s.SURFACE_P * p.RNPKRUNOFF(k.DTSR)
        r.RRUNOFF_K = s.SURFACE_K * p.RNPKRUNOFF(k.DTSR)

        
        r.RNSUBSOIL = min(p.RNSOILMAX, s.SURFACE_N * p.RNABSORPTION)
        r.RPSUBSOIL = min(p.RPSOILMAX, s.SURFACE_P * p.RPABSORPTION)
        r.RKSUBSOIL = min(p.RKSOILMAX, s.SURFACE_K * p.RKABSORPTION)

        r.RNSOIL = -max(0., min(p.NSOILBASE_FR * self.NSOILI, s.NSOIL))
        r.RPSOIL = -max(0., min(p.PSOILBASE_FR * self.PSOILI, s.PSOIL))
        r.RKSOIL = -max(0., min(p.KSOILBASE_FR * self.KSOILI, s.KSOIL))

        
        RNUPTAKE = k.RNUPTAKE if "RNUPTAKE" in self.kiosk else 0.
        RPUPTAKE = k.RPUPTAKE if "RPUPTAKE" in self.kiosk else 0.
        RKUPTAKE = k.RKUPTAKE if "RKUPTAKE" in self.kiosk else 0.

        r.RNAVAIL = r.RNSUBSOIL + p.BG_N_SUPPLY - RNUPTAKE - r.RNSOIL
        r.RPAVAIL = r.RPSUBSOIL + p.BG_P_SUPPLY - RPUPTAKE - r.RPSOIL
        r.RKAVAIL = r.RKSUBSOIL + p.BG_K_SUPPLY - RKUPTAKE - r.RKSOIL
        
    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate states with rates
        """
        rates = self.rates
        states = self.states
        params = self.params

        
        states.SURFACE_N += (rates.FERT_N_SUPPLY - rates.RNSUBSOIL - rates.RRUNOFF_N)
        states.SURFACE_P += (rates.FERT_P_SUPPLY - rates.RPSUBSOIL - rates.RRUNOFF_P)
        states.SURFACE_K += (rates.FERT_K_SUPPLY - rates.RKSUBSOIL - rates.RRUNOFF_K)

        
        states.TOTN_RUNOFF += rates.RRUNOFF_N
        states.TOTP_RUNOFF += rates.RRUNOFF_P
        states.TOTK_RUNOFF += rates.RRUNOFF_K

        
        states.NSOIL += rates.RNSOIL * delt
        states.PSOIL += rates.RPSOIL * delt
        states.KSOIL += rates.RKSOIL * delt
        
        
        states.NAVAIL += rates.RNAVAIL * delt
        states.PAVAIL += rates.RPAVAIL * delt
        states.KAVAIL += rates.RKAVAIL * delt

        
        states.NAVAIL = min(states.NAVAIL, params.NMAX)
        states.PAVAIL = min(states.PAVAIL, params.PMAX)
        states.KAVAIL = min(states.KAVAIL, params.KMAX)


    def _on_APPLY_NPK(self, N_amount:float=None, P_amount:float=None, K_amount:float=None, 
                      N_recovery:float=None, P_recovery:float=None, K_recovery:float=None):
        """Apply NPK based on amounts and update relevant parameters
        """
        if N_amount is not None:
            self._FERT_N_SUPPLY = N_amount * N_recovery
            self.states.TOTN += N_amount
        if P_amount is not None:
            self._FERT_P_SUPPLY = P_amount * P_recovery
            self.states.TOTP += P_amount
        if K_amount is not None:
            self._FERT_K_SUPPLY = K_amount * K_recovery
            self.states.TOTK += K_amount
