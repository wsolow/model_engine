"""
Class to calculate various nutrient relates stress factors:
    NNI      nitrogen nutrition index   
    PNI      phosphorous nutrition index
    KNI      potassium nutrition index
    NPKI     NPK nutrition index (= minimum of N/P/K-index)
    NPKREF   assimilation reduction factor based on NPKI

Written by: Allard de Wit and Iwan Supi (allard.dewit@wur.nl), July 2015
Approach based on: LINTUL N/P/K made by Joost Wolf
Modified by Will Solow, 2024
"""

from datetime import date

from model_engine.models.base_model import TensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorAfgenTrait
from model_engine.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class NPK_Stress(TensorModel):
    """Implementation of NPK stress calculation through [NPK]nutrition index.
    """

    class Parameters(ParamTemplate):
        NMAXLV_TB = TensorAfgenTrait()  
        PMAXLV_TB = TensorAfgenTrait()  
        KMAXLV_TB = TensorAfgenTrait()  
        NCRIT_FR = Tensor(-99.)   
        PCRIT_FR = Tensor(-99.)   
        KCRIT_FR = Tensor(-99.)   
        NMAXRT_FR = Tensor(-99.)  
        NMAXST_FR = Tensor(-99.)  
        PMAXST_FR = Tensor(-99.)  
        PMAXRT_FR = Tensor(-99.)  
        KMAXRT_FR = Tensor(-99.)  
        KMAXST_FR = Tensor(-99.)  
        NRESIDLV = Tensor(-99.)  
        NRESIDST = Tensor(-99.)  
        PRESIDLV = Tensor(-99.)  
        PRESIDST = Tensor(-99.)  
        KRESIDLV = Tensor(-99.)  
        KRESIDST = Tensor(-99.)  
        NLUE_NPK = Tensor(-99.)  

    class RateVariables(RatesTemplate):
        NNI = Tensor()
        PNI = Tensor()
        KNI = Tensor()
        NPKI = Tensor()
        RFNPK = Tensor()

    def __init__(self, day:date, kiosk, parvalues:dict):
        """
        :param day: current date
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: ParameterProvider with parameter key/value pairs
        """

        self.kiosk = kiosk
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, 
                                publish=["NNI", "PNI", "KNI", "NPKI", "RFNPK"])

    
    def __call__(self, day:date, drv):
        """
        :param day: the current date
        :param drv: the driving variables
        :return: A tuple (NNI, NPKI, NPKREF)
        """
        p = self.params
        r = self.rates
        k = self.kiosk

        
        NMAXLV = p.NMAXLV_TB(k.DVS)
        PMAXLV = p.PMAXLV_TB(k.DVS)
        KMAXLV = p.KMAXLV_TB(k.DVS)

        
        NMAXST = p.NMAXST_FR * NMAXLV
        PMAXST = p.PMAXRT_FR * PMAXLV
        KMAXST = p.KMAXST_FR * KMAXLV
        
        
        VBM = k.WLV + k.WST
      
        
        
        NcriticalLV  = p.NCRIT_FR * NMAXLV * k.WLV
        NcriticalST  = p.NCRIT_FR * NMAXST * k.WST
        
        PcriticalLV = p.PCRIT_FR * PMAXLV * k.WLV
        PcriticalST = p.PCRIT_FR * PMAXST * k.WST

        KcriticalLV = p.KCRIT_FR * KMAXLV * k.WLV
        KcriticalST = p.KCRIT_FR * KMAXST * k.WST
        
        
        if VBM > 0.:
            NcriticalVBM = (NcriticalLV + NcriticalST)/VBM
            PcriticalVBM = (PcriticalLV + PcriticalST)/VBM
            KcriticalVBM = (KcriticalLV + KcriticalST)/VBM
        else:
            NcriticalVBM = PcriticalVBM = KcriticalVBM = 0.

        
        
        
        if VBM > 0.:
            NconcentrationVBM  = (k.NAMOUNTLV + k.NAMOUNTST)/VBM
            PconcentrationVBM  = (k.PAMOUNTLV + k.PAMOUNTST)/VBM
            KconcentrationVBM  = (k.KAMOUNTLV + k.KAMOUNTST)/VBM
        else:
            NconcentrationVBM = PconcentrationVBM = KconcentrationVBM = 0.

        
        
        
        if VBM > 0.:
            NresidualVBM = (k.WLV * p.NRESIDLV + k.WST * p.NRESIDST)/VBM
            PresidualVBM = (k.WLV * p.PRESIDLV + k.WST * p.PRESIDST)/VBM
            KresidualVBM = (k.WLV * p.KRESIDLV + k.WST * p.KRESIDST)/VBM
        else:
            NresidualVBM = PresidualVBM = KresidualVBM = 0.
            
        if (NcriticalVBM - NresidualVBM) > 0.:
            r.NNI = limit(0.001, 1.0, (NconcentrationVBM - NresidualVBM)/(NcriticalVBM - NresidualVBM))
        else:
            r.NNI = 0.001
            
        if (PcriticalVBM - PresidualVBM) > 0.:
            r.PNI = limit(0.001, 1.0, (PconcentrationVBM - PresidualVBM)/(PcriticalVBM - PresidualVBM))
        else:
           r.PNI = 0.001
            
        if (KcriticalVBM-KresidualVBM) > 0:
            r.KNI = limit(0.001, 1.0, (KconcentrationVBM - KresidualVBM)/(KcriticalVBM - KresidualVBM))
        else:
            r.KNI = 0.001
      
        r.NPKI = min(r.NNI, r.PNI, r.KNI)

        
        r.RFNPK = limit(0., 1.0, 1. - (p.NLUE_NPK * (1.0001 - r.NPKI) ** 2))
         
        return r.NNI, r.NPKI, r.RFNPK

    def reset(self):
        """Reset states and rates
        """
        r = self.rates

        r.NNI = r.PNI = r.KNI = r.NPKI = r.RFNPK = 0