"""Implementations of the WOFOST waterbalance modules for simulation
of NPK limited production

Written by: Will Solow, 2025
"""
from datetime import date
import torch

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

    def __init__(self, day:date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, dict)
        
        p = self.params
        self._NSOILI = p.NSOILBASE
        self._PSOILI = p.PSOILBASE
        self._KSOILI = p.KSOILBASE
        
        self.states = self.StateVariables(kiosk=self.kiosk, publish=[],
            NSOIL=p.NSOILBASE, PSOIL=p.PSOILBASE, KSOIL=p.KSOILBASE,
            NAVAIL=p.NAVAILI, PAVAIL=p.PAVAILI, KAVAIL=p.KAVAILI, 
            TOTN=0., TOTP=0., TOTK=0., SURFACE_N=0, SURFACE_P=0, SURFACE_K=0, 
            TOTN_RUNOFF=0, TOTP_RUNOFF=0, TOTK_RUNOFF=0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=[])

        self.zero_tensor = torch.Tensor([0.]).to(self.device)
    
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

        r.RNSUBSOIL = torch.min(p.RNSOILMAX, s.SURFACE_N * p.RNABSORPTION)
        r.RPSUBSOIL = torch.min(p.RPSOILMAX, s.SURFACE_P * p.RPABSORPTION)
        r.RKSUBSOIL = torch.min(p.RKSOILMAX, s.SURFACE_K * p.RKABSORPTION)

        r.RNSOIL = -torch.max(self.zero_tensor, torch.min(p.NSOILBASE_FR * self.NSOILI, s.NSOIL))
        r.RPSOIL = -torch.max(self.zero_tensor, torch.min(p.PSOILBASE_FR * self.PSOILI, s.PSOIL))
        r.RKSOIL = -torch.max(self.zero_tensor, torch.min(p.KSOILBASE_FR * self.KSOILI, s.KSOIL))
        
        RNUPTAKE = k.RNUPTAKE if "RNUPTAKE" in self.kiosk else 0.
        RPUPTAKE = k.RPUPTAKE if "RPUPTAKE" in self.kiosk else 0.
        RKUPTAKE = k.RKUPTAKE if "RKUPTAKE" in self.kiosk else 0.

        r.RNAVAIL = r.RNSUBSOIL + p.BG_N_SUPPLY - RNUPTAKE - r.RNSOIL
        r.RPAVAIL = r.RPSUBSOIL + p.BG_P_SUPPLY - RPUPTAKE - r.RPSOIL
        r.RKAVAIL = r.RKSUBSOIL + p.BG_K_SUPPLY - RKUPTAKE - r.RKSOIL
    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate states with rates
        """
        r = self.rates
        s = self.states
        p = self.params

        s.SURFACE_N = s.SURFACE_N + (r.FERT_N_SUPPLY - r.RNSUBSOIL - r.RRUNOFF_N)
        s.SURFACE_P = s.SURFACE_P + (r.FERT_P_SUPPLY - r.RPSUBSOIL - r.RRUNOFF_P)
        s.SURFACE_K = s.SURFACE_K + (r.FERT_K_SUPPLY - r.RKSUBSOIL - r.RRUNOFF_K)

        s.TOTN_RUNOFF = s.TOTN_RUNOFF + r.RRUNOFF_N
        s.TOTP_RUNOFF = s.TOTP_RUNOFF + r.RRUNOFF_P
        s.TOTK_RUNOFF = s.TOTK_RUNOFF + r.RRUNOFF_K

        s.NSOIL = s.NSOIL + r.RNSOIL * delt
        s.PSOIL = s.PSOIL + r.RPSOIL * delt
        s.KSOIL = s.KSOIL + r.RKSOIL * delt
        
        s.NAVAIL = s.NAVAIL + r.RNAVAIL * delt
        s.PAVAIL = s.PAVAIL + r.RPAVAIL * delt
        s.KAVAIL = s.KAVAIL + r.RKAVAIL * delt

        s.NAVAIL = torch.min(s.NAVAIL, p.NMAX)
        s.PAVAIL = torch.min(s.PAVAIL, p.PMAX)
        s.KAVAIL = torch.min(s.KAVAIL, p.KMAX)

    def _on_APPLY_NPK(self, N_amount:float=None, P_amount:float=None, K_amount:float=None, 
                      N_recovery:float=None, P_recovery:float=None, K_recovery:float=None):
        """Apply NPK based on amounts and update relevant parameters
        """
        s = self.states
        if N_amount is not None:
            self._FERT_N_SUPPLY = N_amount * N_recovery
            s.TOTN = s.TOTN + N_amount
        if P_amount is not None:
            self._FERT_P_SUPPLY = P_amount * P_recovery
            s.TOTP = s.TOTP + P_amount
        if K_amount is not None:
            self._FERT_K_SUPPLY = K_amount * K_recovery
            s.TOTK = s.TOTK + K_amount

    def reset(self, day:date):

        p = self.params
        self._NSOILI = p.NSOILBASE
        self._PSOILI = p.PSOILBASE
        self._KSOILI = p.KSOILBASE
        
        self.states = self.StateVariables(kiosk=self.kiosk, publish=[],
            NSOIL=p.NSOILBASE, PSOIL=p.PSOILBASE, KSOIL=p.KSOILBASE,
            NAVAIL=p.NAVAILI, PAVAIL=p.PAVAILI, KAVAIL=p.KAVAILI, 
            TOTN=0., TOTP=0., TOTK=0., SURFACE_N=0, SURFACE_P=0, SURFACE_K=0, 
            TOTN_RUNOFF=0, TOTP_RUNOFF=0, TOTK_RUNOFF=0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, publish=[])