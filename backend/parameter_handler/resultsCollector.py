#from sqlalchemy import create_engine
from sqlalchemy import *
import json
import numpy as np
from decimal import Decimal
from .algorithmCollector import AlgorithmCollector
from sqlalchemy import *
import os

class ResultsCollector():

    #Input-Parameter über Fremdschlüssel 'run_id' anfordern, 
    #wenn der Benutzer die Erweiterte Ansicht anfordert
    def getRun(run_id):
        #Datenbankverbindung herstellen
        engine = create_engine('mysql+pymysql://'\
                + os.environ['MYSQL_USER'] + ":" \
                + os.environ['MYSQL_PASSWORD'] + "@" \
                + os.environ['MYSQL_IP'] + "/" \
                + os.environ['MYSQL_DB']
                )
        conn = engine.connect()
        metadata = MetaData()
        metadata.reflect(engine)

        run = Table('run',metadata,autoload=True,autoload_with=engine)

        sel = select([run]).where(run.c.id == run_id)

        result = conn.execute(sel)

        return(dict(zip([c.name for c in run.columns],list(result)[0])))

    #Daten aller Algorithmen sammeln und in einer Liste zusammen speichern
    def getTabledata():
        #Datenbankverbindung herstellen
        engine = create_engine('mysql+pymysql://'\
                + os.environ['MYSQL_USER'] + ":" \
                + os.environ['MYSQL_PASSWORD'] + "@" \
                + os.environ['MYSQL_IP'] + "/" \
                + os.environ['MYSQL_DB']
                )
        conn = engine.connect()
        metadata = MetaData()
        metadata.reflect(engine)

        table_data = []
        algorithm_display_name = ""
        
        metadata_list = AlgorithmCollector.collect_params()

        row_counter = 1
        #ein Schliefendurchlauf für jeden Algorithmus
        for table_index,tablename in enumerate(ResultsCollector.getAlgorithmTablenames()):

            for alg_metadata in metadata_list:
                if alg_metadata["algorithm_filename"].split('.')[0] == tablename:
                    algorithm_display_name = alg_metadata["algorithm_display_name"]

            table = Table(tablename,metadata,autoload=True,autoload_with=engine)

            #Attributnamen der Tabelle
            column_names = [c.name for c in table.columns]
            
            #Alle Daten der Algorithmus-Tabelle anfragen
            sel = select([table])
            result = conn.execute(sel)

            #Attibutnamen und Werte mit zip zu einem Dictionary zusammensetzen
            for row_index,row in enumerate(list(result)):
                tmp_row = dict(zip(column_names,row), algorithm_display_name=algorithm_display_name)
                if 'label' not in tmp_row.keys():
                    tmp_row.update({"label":""})

                #jeder Zeile eine eindeutige ID hinzufügen, 
                #damit jedes Objekt im DOM später eine eigene ID besitzt

                tmp_row.update({"id":row_counter})
                table_data.append(tmp_row)
                row_counter+=1

        return({"table_data":table_data})

    def getAlgorithmTablenames():
        engine = create_engine('mysql+pymysql://'\
                + os.environ['MYSQL_USER'] + ":" \
                + os.environ['MYSQL_PASSWORD'] + "@" \
                + os.environ['MYSQL_IP'] + "/" \
                + os.environ['MYSQL_DB']
                )
        conn = engine.connect()
        metadata = MetaData()
        metadata.reflect(engine)

        result_tables = list(metadata.tables.keys())
        result_tables.remove("run")

        return(result_tables)
