"""Main crop class for handling growth of the crop. Includes the base crop model
and WOFOST8 model for annual crop growth

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date

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

    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE model instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        
        self.params = self.Parameters(parvalues)
        self.kiosk = kiosk
        
        
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

        self.states = self.StateVariables(kiosk,
                publish=["TAGP", "GASST", "MREST", "CTRAT", "CEVST", "HI", 
                         "DOF", "FINISH_TYPE", "FIN",],
                TAGP=TAGP, GASST=0.0, MREST=0.0, CTRAT=0.0, HI=0.0, CEVST=0.0,
                DOF=None, FINISH_TYPE=None, FIN=False)
        
        self.rates = self.RateVariables(kiosk, 
                    publish=["GASS", "PGASS", "MRES", "ASRC", "DMI", "ADMI"])

        
        checksum = parvalues["TDWI"] - self.states.TAGP - self.kiosk.TWRT
            

    @staticmethod
    def _check_carbon_balance(day, DMI:float, GASS:float, MRES:float, CVF:float, pf:float):
        """Checks that the carbon balance is valid after integration
        """
        (FR, FL, FS, FO) = pf
        checksum = (GASS - MRES - (FR+(FL+FS+FO)*(1.-FR)) * DMI/CVF) * \
                    1./(max(0.0001,GASS))
    
    def calc_rates(self, day:date, drv):
        """Calculate state rates for integration 
        """
        params = self.params
        rates  = self.rates
        k = self.kiosk

        
        self.pheno.calc_rates(day, drv)
        crop_stage = self.pheno.get_variable("STAGE")

        
        
        if crop_stage == "emerging":
            return

        
        rates.PGASS = self.assim(day, drv)
        
        
        self.evtra(day, drv)

        
        NNI, NPKI, RFNPK = self.npk_stress(day, drv)

        
        reduction = min(RFNPK, k.RFTRA)

        rates.GASS = rates.PGASS * reduction

        
        PMRES = self.mres(day, drv)
        rates.MRES = min(rates.GASS, PMRES)

        
        rates.ASRC = rates.GASS - rates.MRES

        
        
        pf = self.part.calc_rates(day, drv)
        CVF = 1./((pf.FL/params.CVL + pf.FS/params.CVS + pf.FO/params.CVO) *
                  (1.-pf.FR) + pf.FR/params.CVR)
        rates.DMI = CVF * rates.ASRC
        self._check_carbon_balance(day, rates.DMI, rates.GASS, rates.MRES,
                                   CVF, pf)

        
        
        self.ro_dynamics.calc_rates(day, drv)
        
        
        rates.ADMI = (1. - pf.FR) * rates.DMI
        self.st_dynamics.calc_rates(day, drv)
        self.so_dynamics.calc_rates(day, drv)
        self.lv_dynamics.calc_rates(day, drv)
        
        
        self.npk_crop_dynamics.calc_rates(day, drv)

    
    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        rates = self.rates
        states = self.states

        
        crop_stage = self.pheno.get_variable("STAGE")

        
        self.pheno.integrate(day, delt)

        
        
        
        
        if crop_stage == "emerging":
            self.touch()
            return

        
        self.part.integrate(day, delt)
        
        
        self.ro_dynamics.integrate(day, delt)
        self.so_dynamics.integrate(day, delt)
        self.st_dynamics.integrate(day, delt)
        self.lv_dynamics.integrate(day, delt)

        
        self.npk_crop_dynamics.integrate(day, delt)

        
        states.TAGP = self.kiosk.TWLV + \
                      self.kiosk.TWST + \
                      self.kiosk.TWSO

        
        states.GASST += rates.GASS
        states.MREST += rates.MRES
        
        
        states.CTRAT += self.kiosk.TRA
        states.CEVST += self.kiosk.EVS



