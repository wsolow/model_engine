"""
wofost.py

Main crop class for handling growth of the crop. Includes the base crop model
and WOFOST8 model for annual crop growth. All written on Tensors

Written by: Wil Solow, 2025
"""

from datetime import date
import torch

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

from model_engine.models.wofost.tensor_batch_crop.tensor_batch_phenology import WOFOST_Phenology_TensorBatch as WOFOST_Phenology
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_respiration import WOFOST_Maintenance_Respiration_TensorBatch as MaintenanceRespiration
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_stem_dynamics import WOFOST_Stem_Dynamics_TensorBatch as Stem_Dynamics
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_root_dynamics import WOFOST_Root_Dynamics_TensorBatch as Root_Dynamics
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_leaf_dynamics import WOFOST_Leaf_Dynamics_NPK_TensorBatch as Leaf_Dynamics
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_storage_organ_dynamics import WOFOST_Storage_Organ_Dynamics_TensorBatch as Storage_Organ_Dynamics
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_assimilation import WOFOST_Assimilation_TensorBatch as Assimilation
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_partitioning import Partitioning_NPK_TensorBatch as Partitioning
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_evapotranspiration import EvapotranspirationCO2_TensorBatch as Evapotranspiration

from model_engine.models.wofost.tensor_batch_crop.tensor_batch_npk_dynamics import NPK_Crop_Dynamics_TensorBatch as NPK_crop
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_nutrients.tensor_batch_npk_stress import NPK_Stress_TensorBatch as NPK_Stress

from model_engine.models.wofost.tensor_batch_soil.tensor_batch_classic_waterbalance import WaterbalanceFD_TensorBatch as WaterbalanceFD
from model_engine.models.wofost.tensor_batch_soil.tensor_batch_npk_soil import NPK_Soil_TensorBatch as NPK_Soil

class WOFOST_TensorBatch(BatchTensorModel):

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

    def __init__(self, day:date, kiosk, parvalues:dict, device, num_models:int=1):
        self.num_models = num_models

        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)
        
        self.pheno = WOFOST_Phenology(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.part = Partitioning(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.assim = Assimilation(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.mres = MaintenanceRespiration(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.evtra = Evapotranspiration(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.ro_dynamics = Root_Dynamics(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.st_dynamics = Stem_Dynamics(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.so_dynamics = Storage_Organ_Dynamics(day,  self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.lv_dynamics = Leaf_Dynamics(day, self.kiosk, parvalues, self.device, num_models=self.num_models)

        self.npk_crop_dynamics = NPK_crop(day,  self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.npk_stress = NPK_Stress(day,  self.kiosk, parvalues, self.device, num_models=self.num_models)

        self.waterbalance = WaterbalanceFD(day,  self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.npk_soil = NPK_Soil(day,  self.kiosk, parvalues, self.device, num_models=self.num_models)
    
        TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk,
                publish=[],
                TAGP=TAGP, GASST=0.0, MREST=0.0, CTRAT=0.0, CEVST=0.0)
        
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, 
                    publish=["ADMI", "DMI"])

    def calc_rates(self, day:date, drv):
        """
        Calculate state rates for integration 
        """
        p = self.params
        r  = self.rates
        k = self.kiosk

        self.pheno.calc_rates(day, drv)
        _emerging = self.pheno._emerging

        # Only evaluates to non-zero when _emerging is false
        r.PGASS = self.assim(day, drv, _emerging)
        
        self.evtra(day, drv, _emerging)

        NNI, NPKI, RFNPK = self.npk_stress(day, drv, _emerging)

        reduction = torch.min(RFNPK, k.RFTRA)

        r.GASS = r.PGASS * reduction

        PMRES = self.mres(day, drv, _emerging)
        r.MRES = torch.min(r.GASS, PMRES)

        r.ASRC = r.GASS - r.MRES

        self.part.calc_rates(day, drv, _emerging)
        CVF = 1. / ((k.FL/p.CVL + k.FS/p.CVS + k.FO/p.CVO) *
                (1.-k.FR) + k.FR/p.CVR)
        r.DMI = CVF * r.ASRC

        self.ro_dynamics.calc_rates(day, drv,_emerging)
        r.ADMI = (1. - k.FR) * r.DMI
        self.st_dynamics.calc_rates(day, drv, _emerging)
        self.so_dynamics.calc_rates(day, drv, _emerging)
        self.lv_dynamics.calc_rates(day, drv, _emerging)
        
        self.npk_crop_dynamics.calc_rates(day, drv, _emerging)

        r.GASS = torch.where(_emerging, 0.0, r.GASS)
        r.PGASS = torch.where(_emerging, 0.0, r.PGASS)
        r.MRES = torch.where(_emerging, 0.0, r.MRES)
        r.ASRC = torch.where(_emerging, 0.0, r.ASRC)
        r.DMI = torch.where(_emerging, 0.0, r.DMI)
        r.ADMI = torch.where(_emerging, 0.0, r.ADMI)

        # runs regardless of _emerging value
        self.waterbalance.calc_rates(day, drv)
        self.npk_soil.calc_rates(day, drv)

        self.rates._update_kiosk()

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

        self.waterbalance.integrate(day, delt)
        self.npk_soil.integrate(day, delt)

        self.states._update_kiosk()

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

        self.waterbalance.reset(day)
        self.npk_soil.reset(day)
    
        TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        self.states = self.StateVariables(kiosk=self.kiosk,
                publish=[],
                TAGP=TAGP, GASST=0.0, MREST=0.0, CTRAT=0.0, CEVST=0.0)
        
        self.rates = self.RateVariables(kiosk=self.kiosk, 
                    publish=["ADMI", "DMI"])
    
        
    def get_output(self, vars:list=None):
        """
        Return the output
        """
        if vars is None:
            return self.rates.ADMI.unsqueeze(-1)
        else:
            output_vars = torch.empty(size=(self.num_models,len(vars))).to(self.device)
            for i, v in enumerate(vars):
                if v in self.states.trait_names():
                    output_vars[:,i] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[:,i] = getattr(self.rates,v)
                elif v in self.kiosk:
                    output_vars[:,i] = getattr(self.kiosk, v)
            return output_vars
        
    def get_extra_states(self):
        """
        Get extra states
        """
        return {}

    def set_model_specific_params(self, k, v):
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)