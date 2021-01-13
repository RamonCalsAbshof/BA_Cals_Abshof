import docker
import json
import os
from django.conf import settings

class DockerHandler():
    def start_container(data_as_string, filename, run_id):
        client = docker.from_env()
        tablename=filename.split('.')[0]
        filename_dir = tablename
        #Befehl, um das Python-Skript im Container mit den Parametern und dem Fremdschlüssel für run-Eintrag zu starten
        algorithm_startup = "python3 " + filename + " " + data_as_string + " " + str(run_id)
        #Path des Algorithmus-Directory wird in den Container gemountet
        algorithm_path = "algorithms/" + str(filename)

        init_script ="./init_script.sh"
        data_as_string = data_as_string[1:-1]
        #startet Container mit Python-Skript Startbefehl
        client.containers.run(
                "calc_wrapper", 
                command=algorithm_startup,
                remove=True, 
                detach=True,
                volumes={
                    os.path.join(settings.ALGORITHMS, filename_dir):{
                        'bind':'/usr/src/app/','mode':'rw'
                        }
                    },
                environment={
                    "MYSQL_USER":str(os.environ["MYSQL_USER"]), 
                    "MYSQL_PASSWORD":str(os.environ["MYSQL_PASSWORD"]), 
                    "MYSQL_IP":str(os.environ["MYSQL_IP"]),
                    "MYSQL_DB":str(os.environ["MYSQL_DB"]),
                    "MYSQL_DB_TABLE":str(tablename)
                    }
                )
