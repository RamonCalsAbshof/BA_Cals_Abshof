flights = {'Name' : 'flights',
           'filename': 'DelayedFlights.csv',
           'identifier' : 'UniqueCarrier',
           'features' : ['Distance', 'ArrDelay'],
           'feature_renames' : ['Distance', 'ArrivalDelay'],
           'time_column_name' : 'DayofMonth',
           'no_timepoints' : 8,
           'normalization' : 'yes',
           'minpts': 3,
           'eps': 0.03,
           'tau' : 0.4,
           'plot': {
               'time_axis_name': 'time stamp',
               'col_wrap': 4,
               'no_timepoints': 8,
               'title': 'Flights Data',
               '1DPlot': True,
           }
           }


flights_oned = {'Name' : 'flights',
           'filename': 'DelayedFlights.csv',
           'identifier' : 'UniqueCarrier',
           'features' : ['Distance'],
           'feature_renames' : ['Distance'],
           'time_column_name' : 'DayofMonth',
           'no_timepoints' : 20,
           'normalization' : 'yes',
           'minpts': 3,
           'eps': 0.03,
           'tau' : 0.5,
           'k' : 5,
           'plot': {
               'time_axis_name': 'year',
               'col_wrap': 4,
               'no_timepoints': 8,
               'title': 'Generated Data',
               '1DPlot': True,
           }
           }


generated_oned_long = {'Name': 'generated',
            'filename' : 'generated_data_mldm_long_40new.csv',
            'identifier': 'object_id',
            'features': ['feature1'],
            'feature_renames' : ['Feature1'],
            'time_column_name': 'time',
            'no_timepoints': 8,
            'normalization': 'no',
            'minpts': 3,
            'eps': 0.025,
            'tau': 0.7,
            'plot' : {
                'time_axis_name' : 'year',
                'col_wrap': 4,
                'no_timepoints': 8,
                'title' : 'Generated Data 1D',
                '1DPlot' : True,
            }}


generated_oned_kmeans = {'Name': 'generated',
            'filename' : 'generated_data_mldm_long_40_kmeans.csv',
            'identifier': 'object_id',
            'features': ['feature1'],
            'feature_renames' : ['Feature1'],
            'time_column_name': 'time',
            'no_timepoints': 8,
            'normalization': 'no',
            'minpts': 3,
            'eps': 0.025,
            'tau': 0.85,
            'k' : 4,
            'plot' : {
                'time_axis_name' : 'year',
                'col_wrap': 4,
                'no_timepoints': 8,
                'title' : 'Generated Data 1D',
                '1DPlot' : True,
            }}


economy = {'Name' : 'economy',
          'path' : './gps_data/',
          'features' : ['10', '150'],
          'feature_renames' : ['Unemployment Rate', 'Edu. Spendings'],
          'start_time' : 2010,
          'end_time' : 2015,
          'minpts': 2,
          'eps': 0.18,
          'k' : 6,
          'tau': 0.4,
           'plot': {
               'time_axis_name': 'Year',
               'col_wrap': 3,
                'no_timepoints': 6,
               'title': 'GlobalEconomy Data',
               '1DPlot' : False,
           }
}