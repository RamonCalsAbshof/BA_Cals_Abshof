import pymysql
from .algorithmCollector import AlgorithmCollector 
import json
from .resultsCollector import ResultsCollector 
from sqlalchemy import *
from sqlalchemy.sql import func
import os

class InitializeDBtables():

    def compareAlgorithmsAndTables():
        #Datenbankverbindung herstellen
        engine = create_engine('mysql+pymysql://'\
                + os.environ['MYSQL_USER'] + ":" \
                + os.environ['MYSQL_PASSWORD'] + "@" \
                + os.environ['MYSQL_IP'] + "/" \
                + os.environ['MYSQL_DB']
                )
        metadata = MetaData()
        #alle Tabellen werden von sqlalchemy nur erstellt, falls noch nicht vorhanden

        #run-Tabelle erstellen, die die Metadaten eines Durchlaufs enthält
        run = Table('run', metadata,
                Column('id', Integer, primary_key=True),
                Column('algorithm_name', String(200)),
                Column('parameters', Text),
                Column('submit_date', DateTime(timezone=True),server_default=func.now())
                )

        #sammelt die Metadaten der Algorithmen aus den JSON-Dateien
        metadata_list = AlgorithmCollector.collect_params()
         
        #für jeden Algorithmus eine eigene Tabelle erstellen, 
        #so wie sie in der JSON-Datei in db_fields definiert sind
        for alg_metadata in metadata_list:

            columns = []
            #Jeder Tabelle werden noch ein 'id' Primärschlüssel, 'run_id' Fremdschlüssel
            #und 'finished_date' Zeitstempel hinzugefügt
            columns.append(Column('id', Integer, primary_key=True))
            columns.append(Column('run_id', None, ForeignKey('run.id')))
            columns.append(Column('finished_date', DateTime(timezone=True),server_default=func.now()))

            #Datenbankfelder aus der JSON-Datei evaluieren
            for field, field_type in alg_metadata["db_fields"].items():
                columns.append(Column(field, eval(field_type)))

            #Name der Tabelle ist der Name des Python-Skript ohne .py Endung
            tablename = alg_metadata["algorithm_filename"].split('.')[0]
            Table(tablename, metadata, *columns)

        metadata.create_all(engine)

