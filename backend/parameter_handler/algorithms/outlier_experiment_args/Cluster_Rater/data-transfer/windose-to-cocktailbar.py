from pymongo import MongoClient
from sklearn import preprocessing
import pandas as pd
import os

class MongoDB(object):

    db = ''
    SETTINGS = ''
    def __init__(self, settings):
        #source
        s_client = MongoClient('134.99.112.190',
                             username='read_user',
                             password='tepco11x?z',
                             authSource='finfraud3',
                             authMechanism='SCRAM-SHA-1',
                             port=27017)
        self.db = s_client.finfraud3.original


        #target
        s_client = MongoClient('localhost',
                               port=27017)
        self.db = s_client.finfraud3.original