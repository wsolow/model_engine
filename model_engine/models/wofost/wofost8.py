"""Main crop class for handling growth of the crop. Includes the base crop model
and WOFOST8 model for annual crop growth

Written by: Wil Solow, 2025
"""

from datetime import date
import torch

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

from model_engine.models.wofost.crop.phenology import WOFOST_Phenology
from model_engine.models.wofost.crop.respiration import WOFOST_Maintenance_Respiration as MaintenanceRespiration
from model_engine.models.wofost.crop.stem_dynamics import WOFOST_Stem_Dynamics as Stem_Dynamics
from model_engine.models.wofost.crop.root_dynamics import WOFOST_Root_Dynamics as Root_Dynamics
from model_engine.models.wofost.crop.leaf_dynamics import WOFOST_Leaf_Dynamics_NPK as Leaf_Dynamics
from model_engine.models.wofost.crop.storage_organ_dynamics import WOFOST_Storage_Organ_Dynamics as Storage_Organ_Dynamics
from model_engine.models.wofost.crop.assimilation import WOFOST_Assimilation as Assimilation
from model_engine.models.wofost.crop.partitioning import Partitioning_NPK as Partitioning
from model_engine.models.wofost.crop.evapotranspiration import EvapotranspirationCO2 as Evapotranspiration

from model_engine.models.wofost.crop.npk_dynamics import NPK_Crop_Dynamics as NPK_crop
from model_engine.models.wofost.crop.nutrients.npk_stress import NPK_Stress as NPK_Stress

class WOFOST80Crop(TensorModel):

    class Parameters(ParamTemplate):
        CVL = Tensor(-99.)
        CVO = Tensor(-99.)
        CVR = Tensor(-99.)
        CVS = Tensor(-99.)

    class StateVariables(StatesTemplate):
        TAGP = Tensor(-99.)
        GASST = Tensor(-99.)
        MREST = Tensor(-99.)
        CTRAT = Tensor(-99.)
        CEVST = Tensor(-99.)

    class RateVariables(RatesTemplate):
        GASS = Tensor(-99.)
        PGASS = Tensor(-99.)
        MRES = Tensor(-99.)
        ASRC = Tensor(-99.)
        DMI = Tensor(-99.)
        ADMI = Tensor(-99.)

    def __init__(self, day:date, kiosk, parvalues:dict, device):

        super().__init__(day, kiosk, parvalues, device)
        
        self.pheno = WOFOST_Phenology(day, kiosk,  parvalues)
        self.part = Partitioning(day, kiosk, parvalues)
        self.assim = Assimilation(day, kiosk, parvalues)
        self.mres = MaintenanceRespiration(day, kiosk, parvalues)
        self.evtra = Evapotranspiration(day, kiosk, parvalues)
        self.ro_dynamics = Root_Dynamics(day, kiosk, parvalues)
        self.st_dynamics = Stem_Dynamics(day, kiosk, parvalues)
        self.so_dynamics = Storage_Organ_Dynamics(day, kiosk, parvalues)
        self.lv_dynamics = Leaf_Dynamics(day, kiosk, parvalues)

        self.npk_crop_dynamics = NPK_crop(day, kiosk, parvalues)
        self.npk_stress = NPK_Stress(day, kiosk, parvalues)
    
        TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        self.states = self.StateVariables(kiosk=self.kiosk,
                publish=[],
                TAGP=TAGP, GASST=0.0, MREST=0.0, CTRAT=0.0, CEVST=0.0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, 
                    publish=["ADMI"])

    def calc_rates(self, day:date, drv):
        """Calculate state rates for integration 
        """
        p = self.params
        r  = self.rates
        k = self.kiosk

        self.pheno.calc_rates(day, drv)
        crop_stage = self.pheno._STAGE

        if crop_stage == "emerging":
            return

        r.PGASS = self.assim(day, drv)
        
        self.evtra(day, drv)

        NNI, NPKI, RFNPK = self.npk_stress(day, drv)

        reduction = torch.min(RFNPK, k.RFTRA)

        r.GASS = r.PGASS * reduction

        PMRES = self.mres(day, drv)
        r.MRES = torch.min(r.GASS, PMRES)

        r.ASRC = r.GASS - r.MRES

        pf = self.part.calc_rates(day, drv)
        CVF = 1. / ((pf.FL/p.CVL + pf.FS/p.CVS + pf.FO/p.CVO) *
                  (1.-pf.FR) + pf.FR/p.CVR)
        r.DMI = CVF * r.ASRC

        self.ro_dynamics.calc_rates(day, drv)
        
        r.ADMI = (1. - pf.FR) * r.DMI
        self.st_dynamics.calc_rates(day, drv)
        self.so_dynamics.calc_rates(day, drv)
        self.lv_dynamics.calc_rates(day, drv)
        
        self.npk_crop_dynamics.calc_rates(day, drv)

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        r = self.rates
        s = self.states

        self.pheno.integrate(day, delt)
        self.part.integrate(day, delt)
        
        self.ro_dynamics.integrate(day, delt)
        self.so_dynamics.integrate(day, delt)
        self.st_dynamics.integrate(day, delt)
        self.lv_dynamics.integrate(day, delt)
        self.npk_crop_dynamics.integrate(day, delt)
        
        s.TAGP = self.kiosk.TWLV + \
                      self.kiosk.TWST + \
                      self.kiosk.TWSO

        s.GASST = s.GASST +  r.GASS
        s.MREST = s.MREST + r.MRES
        
        s.CTRAT = s.CTRAT + self.kiosk.TRA
        s.CEVST = s.CEVST + self.kiosk.EVS


    def reset(self, day:date):
        """Reset the model
        """
        self.pheno.reset(day)
        self.part.reset(day)
        self.assim.reset(day)
        self.mres.reset(day)
        self.evtra.reset(day)
        self.ro_dynamics.reset(day)
        self.st_dynamics.reset(day)
        self.so_dynamics.reset(day)
        self.lv_dynamics.reset(day)

        self.npk_crop_dynamics.reset(day)
        self.npk_stress.reset(day)
    
        TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        self.states = self.StateVariables(kiosk=self.kiosk,
                publish=[],
                TAGP=TAGP, GASST=0.0, MREST=0.0, CTRAT=0.0, CEVST=0.0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, 
                    publish=["ADMI"])