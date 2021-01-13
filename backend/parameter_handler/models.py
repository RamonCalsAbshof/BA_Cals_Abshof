from django.db import models
from django.utils.timezone import now
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .initializeDBtables import InitializeDBtables
import zipfile
import os
import shutil

#Speicherort für JSON-Dateien mit Metadaten
metadata_dir = FileSystemStorage(location=settings.ALG_METADATA)
class Algorithms(models.Model):
    algorithm_zip = models.FileField()
    json_metadata = models.FileField(storage=metadata_dir,null=True)

    def __str__(self):
        return self.algorithm_zip.name

#hochgeladene zip-Datei mit Algorithmus und Dependencies entpacken
@receiver(post_save, sender=Algorithms)
def unzip_algorithm_upload(sender, instance ,**kwargs):
    #Tabelle für den neuen Algorithmus erstellen
    InitializeDBtables.compareAlgorithmsAndTables()
    path = instance.algorithm_zip.path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path,"r") as zf:
            zf.extractall(settings.ALGORITHMS)
            zf.close()
    else:
        return


#JSON-Datei wird automatisch gelöscht, weil es ein einzelne Datei ist 
#und weil der Speicherort durch metadata_dir bekannt ist
#Das Directory des Algorithmus muss eigenhändig mit shutil gelöscht werden
@receiver(post_delete, sender=Algorithms)
def delete_dir_from_zip(sender, instance, **kwargs):
    dirname = instance.algorithm_zip.name.split('.')[0]
    shutil.rmtree(os.path.join(settings.ALGORITHMS, dirname))
    return 

