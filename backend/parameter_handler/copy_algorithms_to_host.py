from .initializeDBtables import InitializeDBtables
from django.conf import settings
import os
from shutil import copytree

InitializeDBtables.compareAlgorithmsAndTables()

alg_dir = settings.ALGORITHMS
for dir_name in [i for i in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, i))]:
        copytree(os.path.join(alg_dir, dir_name), os.path.join(settings.BASE_DIR, 'parameter_handler', 'algorithms_on_host', dir_name), dirs_exist_ok=True)