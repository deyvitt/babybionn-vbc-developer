# __init__.py - Make operActions a proper Python package

from ._medical import MedicalActionVNI, MedicalOperActionConfig
from ._legal import LegalActionVNI, LegalOperActionConfig
from ._general import GeneralActionVNI, GeneralOperActionConfig
from ._technical import TechnicalActionVNI, TechnicalOperActionConfig

__all__ = [
    'MedicalActionVNI',
    'MedicalOperActionConfig',
    'LegalActionVNI', 
    'LegalOperActionConfig',
    'GeneralActionVNI',
    'GeneralOperActionConfig',
    'TechnicalActionVNI',
    'TechnicalOperActionConfig'
]  
