from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import Http404
from .dockerHandler import DockerHandler
from .algorithmCollector import AlgorithmCollector
from .resultsCollector import ResultsCollector
import json
from django.core.serializers.json import DjangoJSONEncoder
from django.http import HttpResponse
from django.views.generic import View
from sqlalchemy import *
import os


class RunView(View):
    def get(self,request, run_id):
        #Gibt die Daten für die Input Parameters in der Erweiterten Ansicht zurück 
        return HttpResponse(json.dumps(ResultsCollector.getRun(run_id), cls=DjangoJSONEncoder))

class TabledataView(View):
    def get(self,request):
        #Gibt die Daten für die Results-Seite zurück
        return HttpResponse(json.dumps(ResultsCollector.getTabledata(), cls=DjangoJSONEncoder))

class TablenameView(View):
    def get(self,request):
        #Gibt die  
        return HttpResponse(json.dumps(ResultsCollector.getAlgorithmTablenames()))

class SubmitView(View):
    def post(self,request, format=None):

        #Filename extrahieren und aus dict löschen
        body = request.body.decode('UTF-8') #ist schon "gedumpt" und wird mit "loads" zu dict
        body_dict = json.loads(body)
        filename = body_dict["algorithm_filename"]
        algorithm_name = filename.split('.')[0]
        body_dict.pop("algorithm_filename")

        body = json.dumps(body_dict)

        data_as_string = body.replace(" ","")
        data_as_string = "'" + data_as_string + "'";

        #Verbindung zur Datenbank herstellen
        engine = create_engine('mysql+pymysql://'\
                + os.environ['MYSQL_USER'] + ":" \
                + os.environ['MYSQL_PASSWORD'] + "@" \
                + os.environ['MYSQL_IP'] + "/" \
                + os.environ['MYSQL_DB']
                )
        conn = engine.connect()
        metadata = MetaData()
        metadata.reflect(engine)
        
        #Name des Algorithmus und Parameter für aktuellen Durchlauf speichern
        run = Table('run',metadata, autoload=True, autoload_with=engine)
        ins = run.insert().values(
                algorithm_name=str(algorithm_name),
                parameters=str(data_as_string)
                )

        #result des inserts speichern, um Primärschlüssel des Durchlauf-Datenbankeintrags zu erhalten
        result = conn.execute(ins)
        DockerHandler.start_container(data_as_string, filename, result.inserted_primary_key[0])

        return HttpResponse(status=status.HTTP_201_CREATED)

class AlgorithmView(View):
    def get(self,request):
        return HttpResponse(json.dumps(AlgorithmCollector.collect_params()))

