# Bachelorarbeit Cals Abshof
Um den Speicherbedarf des Repositories klein zu halten sind die csv-Dateien __testdateimitlabels&#46;csv, dataset_negative_audit_analytics&#46;csv, dataset_positive_audit_analytics&#46;csv__ und __dataset_negative_positive_audit_analytics&#46;csv__ nicht im repository enthalten. Diese csv-Dateien können nachträglich in den selben Ordner eingefügt werden, wo sich das Shell-Skript __cp_csv&#46;sh__ befindet. Wenn das Shell-Skript __cp_csv&#46;sh__ ausgeführt wird, werden die csv-Dateien automatisch in die richtigen Ordner kopiert.  

Mit __rm_csv&#46;sh__ können alle csv-Dateien wieder aus den Ordnern entfernt werden.  

## 1. Docker
Die Webanwendung benötigt ein Docker-Image zur Ausführung der Algorithmen in Docker-Containern. 
Dafür muss das Dockerfile __Dockerfile__ zu einem Image namens __calc_wrapper__ gebaut werden:  
__docker build &#46; -t calc_wrapper__

## 2. Back-End-Server/Django Server

Die Python-Abhängigkeiten für das Back-End befinden sich in __requirements&#46;txt__. Diese müssen zuerst heruntergeladen werden.

In den Shell-Skripten __/backend/django_development&#46;sh__ und __/backend/django_production&#46;sh__ müssen folgende Umgebungsvariablen gesetzt werden:   
__MYSQL_USER__ ist Benutzername des Nutzers, über den die Webanwendung auf die Datenbank zugreift. Es ist wichtig, dass MySQL selbst und der angegebene Datenbanknutzer externe Zugriffe für die Webanwendung erlauben.   
__MYSQL_PASSWORD__ ist das Passwort des MySQL-Nutzers.  
__MYSQL_IP__ ist die IP-Adresse des Systems, auf dem der Datenbank-Server läuft.  
__MYSQL_DB__ ist der Name einer neuen/leeren Datenbank innerhalb von MySQL, die von der Webanwendung verwaltet werden darf.  

__/backend/django_development&#46;sh__ startet den development Server von Django.  

__/backend/django_production&#46;sh__ startet einen production Server von gunicorn. Vor dem Start des production Servers sollte in __/backend/backend/settings&#46;py__ __debug=False__ gesetzt werden, damit fremden Nutzern keine sensiblen Daten angezeigt werden.  

Die Webanwendnung initialisiert für alle vorhandenen Algorithmen jeweils eine eigene Tabelle und eine Tabelle __run__ für Metadaten zu einzelnen Durchläufen in der MYSQL_DB Datenbank.

Es muss beachtet werden, dass der Algorithmus "Outlier Detector" Daten aus einer Datenbank der Heinrich-Heine-Universität bezieht. Deshalb muss entweder ein entsprechender VPN verwendet werden, oder ein direkter Zugriff auf das Universitäts-WLAN bestehen.


Um einen Administrator-Account zu erstellen muss in __/backend/__ folgender Befehl ausgeführt werden: __python3 manage&#46;py createsuperuser__
Es muss nicht zwingend eine Email-Adresse angegeben werden.  

Bei Schwierigkeiten mit Durchläufen kann in __/backend/parameter_handler/dockerHandler&#46;py__ __remove=False__ gesetzt werden, um den Log des Containers einer Berechnung zu überprüfen.  

## Front-End-Server/ReactJS-Server
### Development-Version 
Im Ordner __/backend/frontend/__ müssen folgende Befehle ausgeführt werden, um den development Server von ReactJS zu starten: 
1. __npm install__
Installiert alle Abhängigkeiten aus package.json  

Um den development Server zu starten:

2. __yarn start__ oder __npm start__
Der development Server von ReactJS läuft unter __localhost:3000__

### Production-Version
Im Ordner __/backend/frontend/__ muss folgender Befehl ausgeführt werden, um einen optimierten Production-Build zu erstellen:
1. __yarn build__
Erstellt ein Directory __/backend/frontend/build/__

Um den production Server zu starten:  

2. __serve -s build__
Der production Server läuft unter __localhost:5000__

## Algorithmus hinzufügen
Neue Algorithmen können vom Administrator unter __localhost:8000/admin__ hochgeladen werden. Es wird nach dem Benutzernamen und Passwort des Administrators gefragt, der zuvor in __/backend/__ mit __python3 manage&#46;py createsuperuser__ erstellt werden muss. Auf der Benutzeroberfläche wird die Relation __Algorithms__ angezeigt. Nach Betätigung wird eine Benutzerobefläche zur Verwaltung der Einträge bzw. Algorithmen angezeigt. In der oberen rechten Ecke befindet sich ein Knopf "Add Algorithms", über den man auf die Seite zum Hochladen der zip-Datei und JSON-Datei kommt.

