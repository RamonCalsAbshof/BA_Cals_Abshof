globaleconomy = {
    'Name' : 'globaleconomy',
    'start_year' : 2010,
    'end_year' : 2017,
    'clusters_per_year': {       '2010': 5,
                                 '2011': 4,
                                 '2012': 5,
                                 '2013': 4,
                                 '2014': 3,
                                 '2015': 5,
                                 '2016': 3,
                                 '2017': 4},
    'num_timeseries' : 28
}

financial = {'Name' : 'financial',
          'features' : ['FF-SOFT_ASSETS', 'FF-PENSION0'],
          'feature_renames' : ['SoftAssets', 'Pension'],
          'start_year' : 2010,
          'end_year' : 2017,
          'no_companies': 10,
             'normalization_feature': 'TR-TotalAssetsReported',
             'clusters_per_year': {'2010': 5,
                                   '2011': 4,
                                   '2012': 5,
                                   '2013': 4,
                                   '2014': 3,
                                   '2015': 5,
                                   '2016': 3,
                                   '2017': 4},
             'num_timeseries': 10


}
