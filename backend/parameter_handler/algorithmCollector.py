import os
import importlib
import json
import sys
from django.conf import settings

class AlgorithmCollector():
    def collect_params():

        param_list = []
        algorithms = AlgorithmCollector.collect_algorithm_filenames_in_dir()


        #Algorithmus aus dir importieren und params appenden
        for algorithm in algorithms:
            with open(os.path.join(settings.ALG_METADATA,algorithm)) as metadata:
                data = json.load(metadata)
                param_list.append(data)
        return param_list

    def collect_db_fields():

        db_fields_list = []
        algorithms = AlgorithmCollector.collect_algorithm_filenames_in_dir()

        #Algorithmus aus dir importieren und params appenden
        for algorithm in algorithms:
            algorithm_split = algorithm.split('.')[0]
            imported_algorithm = importlib.import_module("parameter_handler.algorithms."+algorithm_split)
            db_fields = imported_algorithm.db_fields
            if(db_fields["filename"] != algorithm_split):
                sys.exit("'filename' in " + algorithm_split + ".db_fields does not match actual filename!")
            db_fields_list.append(db_fields)

        return db_fields_list

    def get_filename_display_name_pairs():
        pairs = []
        algorithms = AlgorithmCollector.collect_algorithm_filenames_in_dir()

        #Algorithmus aus dir importieren und params appenden
        for algorithm in algorithms:
            algorithm_split = algorithm.split('.')[0]
            imported_algorithm = importlib.import_module("parameter_handler.algorithms."+algorithm_split)
            pairs.append({
                "algorithm_display_name":imported_algorithm.params["algorithm_display_name"],
                "filename":imported_algorithm.db_fields["filename"]
                })

        return pairs

    def collect_params_and_db_fields():
        param_list = []
        db_fields_list = []

        algorithms = AlgorithmCollector.collect_algorithm_filenames_in_dir()

        #Algorithmus aus dir importieren und params appenden
        for algorithm in algorithms:
            algorithm_split = algorithm.split('.')[0]
            imported_algorithm = importlib.import_module("parameter_handler.algorithms."+algorithm_split)

            #params
            parameters = imported_algorithm.params
            parameters["algorithm_file_name"] = algorithm
            param_list.append(parameters)

            #dbFields
            db_fields = imported_algorithm.db_fields
            if(db_fields["filename"] != algorithm_split):
                sys.exit("'filename' in " + algorithm_split + ".db_fields does not match actual filename!")
            db_fields_list.append(db_fields)

        return({"params_list":param_list,"db_fields_list":db_fields_list})

    
    def collect_algorithm_filenames_in_dir():
        algorithms = []
        for f in os.listdir(settings.ALG_METADATA):
            if f.endswith(".json"):
                algorithms.append(f)

        return algorithms
