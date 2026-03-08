# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026, BabyBIONN Contributorsnan

import os
import builtins

original_open = builtins.open

def patched_open(file, mode='r', *args, **kwargs):
    if 'knowledge_' in str(file):
        # Redirect all knowledge files to custom directory
        custom_file = f"/home/baby-bionn/.babybionn_knowledge/{os.path.basename(file)}"
        return original_open(custom_file, mode, *args, **kwargs)
    return original_open(file, mode, *args, **kwargs)

builtins.open = patched_open
