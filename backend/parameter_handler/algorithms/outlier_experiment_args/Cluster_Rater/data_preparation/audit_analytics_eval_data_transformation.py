import pandas as pd
import numpy as np
from pymongo import MongoClient
import pickle

client = MongoClient('134.99.112.190',
                     username='read_user',
                     password='tepco11x?z',
                     authSource='finfraud3',
                     authMechanism='SCRAM-SHA-1',
                     port=27017)
db = client.finfraud3.original

raw_data = pd.read_csv('restatements-negative.csv', index_col=None)
raw_data.reset_index()

data_transformed = pd.DataFrame(columns=['cik', 'ric', 'year', 'effect'])

for index, row in raw_data.iterrows():
    # 0 - CIK
    # 30 - Restated Period Ended
    # 29 - Restated Period Begin

    if(row[30] is not np.nan) and (row[29] is not np.nan):
        for year in range(int(row[29][:4]), int(row[30][:4])+1):
            #print({'cik' : int(row[0]), 'ric' : '', 'year' : year, 'effect' : 'positive'})
            data_transformed = data_transformed.append({'cik' : int(row[0]), 'ric' : '', 'year' : year, 'effect' : 'positive'}, ignore_index = True)


print(data_transformed)

#cik_to_ric = dict()
#for document in db.find():
#   d = dict(document)
#    cik_to_ric[d['TR-CIKNUMBER']] = d['ric']


#with open('cik_to_ric.pickle', 'wb') as handle:
#    pickle.dump(cik_to_ric, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('cik_to_ric.pickle', 'rb') as handle:
    cik_to_ric = pickle.load(handle)


for cik in data_transformed['cik'].unique():
    #print(cik)
    try:
        data_transformed.loc[data_transformed['cik'] == cik, 'ric'] = cik_to_ric[str(cik)]
    except:
        print(str(cik)+ ' is not available')
        data_transformed.loc[data_transformed['cik'] == cik, 'ric'] = "NULL"

#for cik in data_transformed['cik'].unique():
#    company = dict(db.find_one({"TR-CIKNUMBER": str(cik)}))
#    data_transformed.loc[data_transformed['cik'] == cik, 'ric'] = company['ric']

data_transformed.to_csv('eval_audit_analytics_restatements_negative.csv', index=False)
print(data_transformed)





