"""Implementation of the Richard's Irrigation Model

Written by Will Solow, 2025
"""
import datetime
import torch
import numpy as np
from scipy.integrate import ode

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
EPS = 1e-12
class RE_Irrigation_Numpy(TensorModel):
    """Implements Richard's Equation Irrigation model
    """

    _INIT_COND = {"HS": 0, "FP":1}
    _T = NDArray(-99.)

    class Parameters(ParamTemplate):
        PSI_MIN    = Tensor(-99.)  # Water balance minimum
        PSI_MAX    = Tensor(-99.)  # Water balance maximum
        THETA_R    = Tensor(-99.)  # theta r parameter
        THETA_S    = Tensor(-99.)  # theta s parameter
        ALPHA      = Tensor(-99.)  # alpha parameter
        N          = Tensor(-99.)  # n parameter
        M          = Tensor(-99.)  # m parameter
        K_S        = Tensor(-99.)  # k s parameter
        SS         = Tensor(-99.)  # SS parameter
        NETA       = Tensor(-99.)  # Neta parameter
        MATRIC_POT = Tensor(-99.)  # Initial PSI

        LBCOND     = Tensor(-99.)  # Lower bound condition (0,1,2)
        INITCOND   = Tensor(-99.)  # initial bound condition (0,1)

        SPACE_S    = Tensor(-99.)  # Space step
        SD         = Tensor(-99.)  # Soil depth 
        TS         = Tensor(-99.)  # Time step
        RT         = Tensor(-99.)  # Run Time

    class RateVariables(RatesTemplate):
        BCT = NDArray(-99.) # Irrigation pulses
        BCB = NDArray(-99.) # BC_T

    class StateVariables(StatesTemplate):
        PSI_0   = NDArray(-99.) # Initial Water balance
        PSI     = NDArray(-99.) # Current water balance
        THETA   = NDArray(-99.) # Theta values
        WBS     = NDArray(-99.) # Water balance S
        DV      = NDArray(-99.) # Soil water levels
      
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device):
        """
        :param day: start date of the simulation
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        super().__init__(day, kiosk, parvalues, device)

        # Define initial states
        p = self.params
        p.M = 1 - 1 / p.N

        # NOTE: This won't work for multi dim tensors
        z = np.arange((p.SPACE_S / 2).cpu().numpy().item(), p.SD.cpu().numpy().item(), 
                         p.SPACE_S.cpu().numpy().item())
        
        self._N = len(z)

        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD.item()
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = np.zeros(len(z)) + p.MATRIC_POT.item()

        # Initial value for DV
        DV = np.zeros((self._N+2))
        DV[1:-1] = INIT_PSI  # Matric potential

        THETA = np.reshape(self.theta_func(INIT_PSI.reshape(-1)), INIT_PSI.shape)
        WBS = np.sum(THETA * p.SPACE_S.item())

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=THETA, PSI=INIT_PSI, WBS=WBS,DV=DV)
        self.rates = self.RateVariables()

        self.solver = ode(self.ode_func_blockcentered)
        #self.solver.set_integrator('vode', method='BDF', uband=1,lband=1)
        self.solver.set_integrator("dopri5", method="Adams")

    def calc_rates(self, day, drv):
        """Calculates the rates for irrigation
        """
        r = self.rates
        p = self.params

        r.BCB = p.MATRIC_POT.item()
        r.BCT = drv.IRRIG

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """
        p = self.params
        s = self.states
        r = self.rates

        self.solver.set_initial_value(s.DV.copy(), 0)

        params=(r.BCT[0], r.BCB[0], s.DV)
        self.solver.set_f_params(*params)
            
        self.solver.integrate(p.TS.cpu().numpy())
        s.DV = self.solver.y
        
        s.PSI = s.DV[1:-1]
        s.THETA = np.reshape(self.theta_func(s.PSI.reshape(-1)), s.PSI.shape)
        s.WBS = np.sum(s.THETA * p.SPACE_S.item())

    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return torch.unsqueeze(self.states.WBS, -1)
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = torch.tensor(getattr(self.states, v))
                elif v in self.rates.trait_names():
                    output_vars[i,:] = torch.tensor( getattr(self.rates,v))
            return output_vars
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        p = self.params
        p.M = 1 - 1 / p.N

        # NOTE: This won't work for multi dim tensors
        z = np.arange((p.SPACE_S / 2).cpu().numpy().item(), p.SD.cpu().numpy().item(), 
                         p.SPACE_S.cpu().numpy().item())
        self._N = len(z)
        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD.item()
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = np.zeros(len(z)) + p.MATRIC_POT.item()

        # Initial value for DV
        DV = np.zeros((self._N+2))
        DV[1:-1] = INIT_PSI  # Matric potential

        THETA = np.reshape(self.theta_func(INIT_PSI.reshape(-1)), INIT_PSI.shape)
        WBS = np.sum(THETA * p.SPACE_S.item())

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=THETA, PSI=INIT_PSI, WBS=WBS,DV=DV)
        self.rates = self.RateVariables()

        self.solver = ode(self.ode_func_blockcentered)
        self.solver.set_integrator('vode',method='BDF',uband=1,lband=1)
        
    def theta_func(self, psi):
        """
        Compute the theta function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA.item()) ** p.N.item()) ** (-p.M.item())

        Se[ psi>0. ] = 1.0

        return p.THETA_R.item() + (p.THETA_S.item() - p.THETA_R.item()) * Se
    
    def c_func(self, psi):
        """
        Compute the C Function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA.item()) ** p.N.item()) ** (-p.M.item())

        Se[ psi>0. ] = 1.0

        dSedh = p.ALPHA.item() * p.M.item() / ( 1 - p.M.item() ) * Se ** ( 1 / p.M.item() ) * ( 1 - Se ** (1/p.M.item()) ) ** p.M.item()

        return Se * p.SS.item()+( p.THETA_S.item() - p.THETA_R.item()) * dSedh

    def k_func(self, psi):
        """
        Compute the K function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA.item()) ** p.N.item()) ** (-p.M.item())
        Se[ psi>0. ] = 1.0

        return p.K_S.item() * Se ** p.NETA.item() * ( 1 - ( 1 - Se ** (1 / p.M.item() ) ) ** p.M.item())**2

    def c_inv_func(self, psi, psi_n):
        """Inverse of C Function"""
        Cinv = 1 / self.c_func(psi)
        return Cinv

    def boundary_fluxes(self, BC_T, BC_B, psiTn, psiBn):
        """Compute boundary fluxes"""
        p = self.params
        # Upper BC: Type 2 specified infiltration flux
        qT = BC_T

        # Lower BC: Type 1 specified pressure head
        psiB = BC_B
        Kout = (self.k_func(np.array([psiBn])) + self.k_func(np.array([psiB]))) / 2.

        qB = -Kout[0] * ( ( psiB - psiBn ) / ( p.SPACE_S.item() / 2 ) - 1. )

        return qT, qB
    
    def ode_func_blockcentered(self, t, DV, BC_T, BC_B, psi_n):
        """Function to be called by ODE solver"""
        return self.ode_func_call(t, DV, BC_T, BC_B, psi_n)

    def ode_func_call(self, t, DV, BC_T, BC_B, psi_n):
        """Ode function call"""

        p = self.params
        # Unpack the dependent variable
        psi = DV[1:-1]
        psi_n = psi_n[1:-1]

        q = np.zeros(self._N+1)
        K = np.zeros(self._N+1)
        K = self.k_func(psi)
        Kmid = (K[1:]+K[:-1])/2.
        
        # Boundary fluxes
        qT,qB = self.boundary_fluxes(BC_T, BC_B, psi[0], psi[-1])
        q[0] = qT
        q[-1] = qB

        # Internal nodes
        q[1:-1] = -Kmid * ( (psi[1:] - psi[:-1] ) / p.SPACE_S.item() - 1)

        # Continuity
        Cinv = self.c_inv_func(psi, psi_n)
        dpsidt = -Cinv * ( q[1:] - q[:-1] ) / p.SPACE_S.item()

        # Pack up dependent variable:
        dDVdt = np.hstack((np.array([qT]),dpsidt,np.array([qB])))
        return dDVdt
    
