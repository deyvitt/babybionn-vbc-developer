# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributors

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
