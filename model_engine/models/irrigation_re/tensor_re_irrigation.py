"""Implementation of the Richard's Irrigation Model

Written by Will Solow, 2025
"""
import datetime
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import ode

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
EPS = 1e-12
class RE_Irrigation_Tensor(TensorModel):
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
        BCT = Tensor(-99.) # Irrigation pulses
        BCB = Tensor(-99.) # BC_T

    class StateVariables(StatesTemplate):
        PSI_0   = Tensor(-99.) # Initial Water balance
        PSI     = Tensor(-99.) # Current water balance
        THETA   = Tensor(-99.) # Theta values
        WBS     = Tensor(-99.) # Water balance S
        DV      = Tensor(-99.) # Soil water levels
      
    def __init__(self, day:datetime.date, kiosk:dict, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)

        # Define initial states
        p = self.params
        p.M = 1 - 1 / p.N

        # NOTE: This won't work for multi dim tensors
        z = torch.arange((p.SPACE_S / 2).item(), p.SD.item(), 
                         p.SPACE_S.item()).to(self.device)
        
        self._N = len(z)

        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = np.zeros(len(z)) + p.MATRIC_POT

        # Initial value for DV
        DV = torch.zeros((self._N+2)).to(self.device)
        DV[1:-1] = INIT_PSI  # Matric potential

        THETA = self.theta_func(INIT_PSI.view(-1)).view(INIT_PSI.shape)
        WBS = torch.sum(THETA * p.SPACE_S)

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=THETA, PSI=INIT_PSI, WBS=WBS,DV=DV)
        self.rates = self.RateVariables()

        self.solver = ode(self.ode_func_blockcentered)
        #self.solver.set_integrator('vode', method='BDF', uband=1,lband=1)
        self.solver.set_integrator('dopri5', method="Adams")

    def calc_rates(self, day, drv):
        """Calculates the rates for irrigation
        """
        r = self.rates
        p = self.params

        r.BCB = p.MATRIC_POT
        r.BCT = drv.IRRIG

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """
        p = self.params
        s = self.states
        r = self.rates

        #self.solver.set_initial_value(s.DV.clone().cpu().numpy(), 0)

        #params=(r.BCT[0], r.BCB[0], s.DV)
        #self.solver.set_f_params(*params)

        #self.solver.integrate(p.TS.cpu().numpy())
        # s.DV = self.solver.y

        y0 = s.DV.clone().unsqueeze(0)
        t_eval = torch.concatenate((torch.tensor([0.]).to(self.device), p.TS, p.TS+p.TS)).unsqueeze(0)

        term = to.ODETerm(self.tensor_ode_func_call)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        jit_solver = torch.compile(solver)

        sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
        
        s.PSI = s.DV[1:-1]
        s.THETA = self.theta_func(s.PSI.view(-1)).view(s.PSI.shape)
        s.WBS = torch.sum(s.THETA * p.SPACE_S)

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
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.rates,v)
            return output_vars
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        p = self.params
        p.M = 1 - 1 / p.N

        # NOTE: This won't work for multi dim tensors
        z = torch.arange((p.SPACE_S / 2).item(), p.SD.item(), 
                         p.SPACE_S.item()).to(self.device)
        
        self._N = len(z)

        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = np.zeros(len(z)) + p.MATRIC_POT

        # Initial value for DV
        DV = torch.zeros((self._N+2)).to(self.device)
        DV[1:-1] = INIT_PSI  # Matric potential

        THETA = self.theta_func(INIT_PSI.view(-1)).view(INIT_PSI.shape)
        WBS = torch.sum(THETA * p.SPACE_S)

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=THETA, PSI=INIT_PSI, WBS=WBS,DV=DV)
        self.rates = self.RateVariables()

        self.solver = ode(self.ode_func_blockcentered)
        #self.solver.set_integrator('vode', method='BDF', uband=1,lband=1)
        self.solver.set_integrator('dopri5', method="Adams")
        
    def theta_func(self, psi):
        """
        Compute the theta function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA) ** p.N) ** (-p.M)

        Se[ psi>0. ] = 1.0

        return p.THETA_R + (p.THETA_S - p.THETA_R) * Se
    
    def c_func(self, psi):
        """
        Compute the C Function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA) ** p.N) ** (-p.M)

        Se[ psi>0. ] = 1.0

        dSedh = p.ALPHA * p.M / ( 1 - p.M ) * Se ** ( 1 / p.M ) * ( 1 - Se ** (1/p.M) ) ** p.M

        return Se * p.SS+( p.THETA_S - p.THETA_R) * dSedh

    def k_func(self, psi):
        """
        Compute the K function
        """
        p = self.params
        Se = ( 1 + (psi * -p.ALPHA) ** p.N) ** (-p.M)
        Se[ psi>0. ] = 1.0

        return p.K_S * Se ** p.NETA * ( 1 - ( 1 - Se ** (1 / p.M ) ) ** p.M)**2

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
        Kout = (self.k_func(psiBn) + self.k_func(psiB)) / 2.

        qB = -Kout[0] * ( ( psiB - psiBn ) / ( p.SPACE_S / 2 ) - 1. )

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

        q = torch.zeros(self._N+1).to(self.device)
        K = torch.zeros(self._N+1).to(self.device)
        K = self.k_func(psi)
        Kmid = (K[1:]+K[:-1])/2.
        
        # Boundary fluxes
        qT,qB = self.boundary_fluxes(BC_T, BC_B, psi[0], psi[-1])
        q[0] = qT
        q[-1] = qB

        # Internal nodes
        q[1:-1] = -Kmid * ( (psi[1:] - psi[:-1] ) / p.SPACE_S - 1)

        # Continuity
        Cinv = self.c_inv_func(psi, psi_n)
        dpsidt = -Cinv * ( q[1:] - q[:-1] ) / p.SPACE_S

        # Pack up dependent variable:
        dDVdt = np.hstack((np.array([qT]),dpsidt,np.array([qB])))

        return dDVdt
        
    def tensor_ode_func_call(self, t, DV):
        """Ode function call"""

        r = self.rates
        s = self.states
        BC_T = r.BCT
        BC_B = r.BCT
        psi_n = s.DV

        p = self.params
        # Unpack the dependent variable
        psi = DV[0][1:-1]
        psi_n = psi_n[1:-1]

        q = torch.zeros(self._N+1).to(self.device)
        K = torch.zeros(self._N+1).to(self.device)
        K = self.k_func(psi)
        Kmid = (K[1:]+K[:-1])/2.
        
        # Boundary fluxes
        qT,qB = self.boundary_fluxes(BC_T, BC_B, psi[0], psi[-1])
        q[0] = qT
        q[-1] = qB

        # Internal nodes
        q[1:-1] = -Kmid * ( (psi[1:] - psi[:-1] ) / p.SPACE_S - 1)

        # Continuity
        Cinv = self.c_inv_func(psi, psi_n)
        dpsidt = -Cinv * ( q[1:] - q[:-1] ) / p.SPACE_S

        # Pack up dependent variable:
        dDVdt = torch.hstack((qT,dpsidt,qB))

        return dDVdt.unsqueeze(0)

