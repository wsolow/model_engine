"""Implementation of the Richard's Irrigation Model

Written by Will Solow, 2025
"""
import datetime
import torch
import numpy as np
from scipy.integrate import ode
import pandas as pd
from functools import partial

from model_engine.inputs.util import daylength
from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate
       
EPS = 1e-12
class RE_Irrigation_Tensor(TensorModel):
    """Implements Richard's Equation Irrigation model
    """

    _INIT_COND = {"HS": 0, "FP":1}

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
        pass
    class StateVariables(StatesTemplate):
        PSI_0 = Tensor(-99.) # Initial Water balance
        THETA = Tensor(-99.) # Theta values
        C = Tensor(-99.) # C values
        K = Tensor(-99.) # K values
        WB = Tensor(-99.) # Water Balance
        QIN = Tensor(-99.) # Water balance in
        QOUT = Tensor(-99.) # Water balance out
        WBS = Tensor(-99.) # Water balance S
      
    def __init__(self, day:datetime.date, parvalues:dict, device):
        """
        :param day: start date of the simulation
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        super().__init__(self, parvalues, device)

        # Define initial states
        p = self.params
        p.M = 1 - 1 / p.N

        # NOTE: This won't work for multi dim tensors
        z = torch.arange((p.SPACE_S / 2).cpu().numpy().item(), p.SD.cpu().numpy().item(), p.SPACE_S.cpu().numpy().item()).to(self.device)
        n = len(z)
        if self._INIT_COND["HS"] == p.INITCOND:
            INIT_PSI = z - p.SD
        elif self._INIT_COND["FP"] == p.INITCOND:
            INIT_PSI = torch.zeros(len(z)) + p.MATRIC_POT

        self.states = self.StateVariables(PSI_0=INIT_PSI, THETA=0., C=0., K=0., WB=0., QIN=0., QOUT=0., WBS=0.,)
        self.rates = self.RateVariables()

        t = torch.arange(0, p.RT.cpu().numpy().item()+ p.TS.cpu().numpy().item(),p.TS.cpu().numpy().item()).to(self.device)
        #t = np.arange(0, p.RT.cpu().numpy().item()+ p.TS.cpu().numpy().item(),p.TS.cpu().numpy().item())
        nt = len(t)

        # Boundary conditions:
        # Set the irrigation pulses for the simulation
        # TODO: This can be in the DRV as drv.IRRIG
        tI = torch.Tensor([0, 10, 1000]).to(self.device)
        Ipulses = torch.Tensor([0.05, 0.05]).to(self.device)
        I = torch.zeros(len(t), device=self.device)
        #tI = np.array([0, 10, 1000])
        #Ipulses = np.array([0.05, 0.05])
        #I = np.zeros(len(t))
        c = 0
        for ti in range(len(t)):
            if t[ti] >= tI[c+1]:
                c += 1
            I[ti] = Ipulses[c]

        BC_T = I + torch.zeros(nt, device=self.device)
        BC_B = p.MATRIC_POT + torch.zeros(nt, device=self.device)
        #BC_T = I + np.zeros(nt)
        #BC_B = p.MATRIC_POT.item() + np.zeros(nt)
        
        self.r = ode(self.odefun_blockcentered)
        self.r.set_integrator('vode',method='BDF',uband=1,lband=1)

        self.run_RE(t,n,BC_T,BC_B)

    def calc_rates(self, day, drv):
        """Calculates the rates for irrigation
        """

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """

    def get_output(self, vars:list=None):
        """
        Return the phenological stage as the floor value
        """
        if vars is None:
            return torch.unsqueeze(self.states.DVS, -1)
        else:
            output_vars = torch.empty(size=(len(vars),1)).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[i,:] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i,:] = getattr(self.states,v)
            return output_vars
  
    def reset(self, day:datetime.date):
        """
        Reset the model
        """
        # Define initial states
        self._STAGE = "ecodorm"
        self.states = self.StateVariables(TSUM=0., TSUME=0., DVS=0., CSUM=0.,
                                          PHENOLOGY=self._STAGE_VAL[self._STAGE])
        self.rates = self.RateVariables()

    def theta_func(self, psi):
        """
        Compute the theta function
        """
        p = self.params
        Se = ( 1 + ( psi * -p.ALPHA ) ** p.N ) ** (-p.M)
        #Se = ( 1 + (psi * - p.ALPHA.item()) ** p.N.item()) ** (-p.M.item())

        Se[ psi>0. ] = 1.0

        return p.THETA_R + (p.THETA_S - p.THETA_R) * Se
        #return p.THETA_R.item() + (p.THETA_S.item() - p.THETA_R.item()) * Se
    
    def c_func(self, psi):
        """
        Compute the C Function
        """
        p = self.params
        Se =(1 + ( psi * -p.ALPHA) ** p.N) ** (-p.M)
        #Se = ( 1 + (psi * - p.ALPHA.item()) ** p.N.item()) ** (-p.M.item())

        Se[ psi>0. ] = 1.0

        dSedh = p.ALPHA * p.M / ( 1 - p.M ) * Se ** ( 1 / p.M ) * ( 1 - Se ** (1/p.M) ) ** p.M
        #dSedh = p.ALPHA.item() * p.M.item() / ( 1 - p.M.item() ) * Se ** ( 1 / p.M.item() ) * ( 1 - Se ** (1/p.M.item()) ) ** p.M.item()

        return Se * p.SS+( p.THETA_S - p.THETA_R) * dSedh
        #return Se * p.SS.item()+( p.THETA_S.item() - p.THETA_R.item()) * dSedh

    def k_func(self, psi):
        """
        Compute the K function
        """
        p = self.params
        Se = ( 1 + (psi * - p.ALPHA) ** p.N) ** (-p.M)
        #Se = ( 1 + (psi * - p.ALPHA.item()) ** p.N.item()) ** (-p.M.item())
        Se[ psi>0. ] = 1.0

        return p.K_S * Se ** p.NETA * ( 1 - ( 1 - Se ** (1 / p.M ) ) ** p.M)**2
        #return p.K_S.item() * Se ** p.NETA.item() * ( 1 - ( 1 - Se ** (1 / p.M.item() ) ) ** p.M.item())**2

    def c_inv_func(self, psi, psi_n):
        """Inverse of C Function"""
        Cinv = 1 / self.c_func(psi)
        return Cinv

    def boundary_fluxes(self, BC_T, BC_B, psiTn, psiBn):
        """Compute boundary fluxes"""
        # Inputs:
        #  BC_T = specified flux at surface or specified pressure head at surface;
        #  BC_B = specified flux at base or specified pressure head at base;
        # psiTn = pressure head at node 0 (uppermost node)
        # psiBn = pressure head at node -1 (lowermost node)

        # Upper BC: Type 1 specified pressure head
        #     psiT=BC_T
        #     Kin=(KFun(np.array([psiT]),pars)+KFun(np.array([psiTn]),pars))/2.
        #     qT=-Kin[0]*((psiTn-psiT)/dz-1.)

        p = self.params
        # Upper BC: Type 2 specified infiltration flux
        qT = BC_T

        # Lower BC: Type 1 specified pressure head
        psiB = BC_B
        Kout = (self.k_func(psiBn) + self.k_func(psiB)) / 2.
        #Kout = (self.k_func(np.array([psiBn])) + self.k_func(np.array([psiB]))) / 2.

        qB = -Kout[0] * ( ( psiB - psiBn ) / ( p.SPACE_S / 2 ) - 1. )
        #qB = -Kout[0] * ( ( psiB - psiBn ) / ( p.SPACE_S.item() / 2 ) - 1. )

        return qT,qB
    
    def odefun_blockcentered(self, t, DV, n, BC_T, BC_B, psi_n):
        """Function to be called by ODE solver"""
        return self.odefuncall(t, DV, n, BC_T, BC_B, psi_n)

    def odefuncall(self, t, DV, n, BC_T, BC_B, psi_n):
        # In this function, we use a block centered grid approch, where the finite difference
        # solution is defined in terms of differences in fluxes. 

        p = self.params
        # Unpack the dependent variable
        psi = torch.tensor(DV[1:-1]).to(self.device)
        #psi = DV[1:-1]
        psi_n = psi_n[1:-1]

        q = torch.zeros(n+1,device=self.device)
        K = torch.zeros(n+1,device=self.device)
        #q = np.zeros(n+1)
        #K = np.zeros(n+1)
        K = self.k_func(psi)
        Kmid = (K[1:]+K[:-1])/2.
        
        # Boundary fluxes
        qT,qB = self.boundary_fluxes(BC_T, BC_B, psi[0], psi[-1])
        q[0] = qT
        q[-1] = qB

        # Internal nodes
        q[1:-1] = -Kmid * ( (psi[1:] - psi[:-1] ) / p.SPACE_S - 1)
        #q[1:-1] = -Kmid * ( (psi[1:] - psi[:-1] ) / p.SPACE_S.item() - 1)

        # Continuity
        Cinv = self.c_inv_func(psi, psi_n)
        dpsidt = -Cinv * ( q[1:] - q[:-1] ) / p.SPACE_S
        #dpsidt = -Cinv * ( q[1:] - q[:-1] ) / p.SPACE_S.item()

        # Pack up dependent variable:
        dDVdt = torch.cat( ( qT.unsqueeze(0), dpsidt, qB ),dim=0 ).to(torch.float32)
        #dDVdt = np.hstack((np.array([qT]),dpsidt,np.array([qB])))
        return dDVdt.cpu().numpy()
        #return dDVdt

    def run_RE(self, t, n, BC_T, BC_B):
        # 4. scipy function "ode", with the jacobian, solving one step at a time:
        s = self.states

        p = self.params
        DV = torch.zeros((len(t),n+2), device=self.device)
        #DV = np.zeros((len(t),n+2))
        DV[0,0] = 0.       # Cumulative inflow
        DV[0,-1] = 0.      # Cumulative outflow
        DV[0,1:-1] = s.PSI_0
        #DV[0,1:-1] = s.PSI_0.cpu().numpy()  # Matric potential

        r = ode(self.odefun_blockcentered)
        r.set_integrator('vode',method='BDF',uband=1,lband=1)

        for i,ti in enumerate(t[:-1]):
            r.set_initial_value(DV[i,:].cpu().numpy(), 0)
            #r.set_initial_value(DV[i,:], 0)

            params=(n, BC_T[i], BC_B[i], DV[i,:])
            r.set_f_params(*params)
            
            r.integrate(p.TS.cpu().numpy())
            DV[i+1,:] = torch.tensor(r.y,device=self.device)
            #DV[i+1,:] = r.y

        # Unpack output:
        QT = DV[:,0]
        QB = DV[:,-1]
        psi = DV[:,1:-1]
        qT = torch.cat((torch.tensor([0],device=self.device),torch.diff(QT))) / p.TS
        qB = torch.cat((torch.tensor([0],device=self.device),torch.diff(QB))) / p.TS
        #qT=np.hstack([0,np.diff(QT)])/p.TS.item()
        #qB=np.hstack([0,np.diff(QB)])/p.TS.item()


        # Water balance terms
        theta = self.theta_func(psi.reshape(-1))
        #theta = np.reshape(theta,psi.shape)
        theta = theta.view(psi.shape) 
        S = torch.sum(theta * p.SPACE_S,1)
        #S = np.sum(theta * p.SPACE_S.item(),1)

        # Pack output into a dataframe:
        WB = pd.DataFrame(index=t.cpu().numpy())
        WB['S'] = S.cpu().numpy()
        WB['QIN'] = qT.cpu().numpy()
        WB['QOUT'] = qB.cpu().numpy()
        print(WB['S'])
        return psi,WB

