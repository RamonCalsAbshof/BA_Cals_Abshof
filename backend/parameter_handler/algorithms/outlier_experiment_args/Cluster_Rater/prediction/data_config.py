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
           'CLOSE': {
                 'minpts_start': 2,
                 'minpts_end': 3,
                 'minpts_step': 1,
                 'eps_start': 0.09,
                 'eps_end': 0.1,
                 'eps_step': 0.01
             },
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
           'no_timepoints' : 8,
           'normalization' : 'yes',
           'minpts': 3,
           'eps': 0.03,
           'tau' : 0.4,
           'plot': {
               'time_axis_name': 'year',
               'col_wrap': 4,
               'no_timepoints': 8,
               'title': 'Generated Data',
               '1DPlot': True,
           }
           }

generated_oned = {'Name': 'generated',
            'filename' : 'generated_1d_data.csv',
            'identifier': 'object_id',
            'features': ['feature1'],
            'feature_renames' : ['Feature1'],
            'time_column_name': 'time',
            'no_timepoints': 8,
            'normalization': 'no',
            'minpts': 3,
            'eps': 0.05,
            'tau': 0.55,
            'plot' : {
                'time_axis_name' : 'year',
                'col_wrap': 4,
                'no_timepoints': 8,
                'title' : 'Generated Data 1D',
                '1DPlot' : True,
            }}


generated = {'Name': 'generated',
            'filename' : 'generated_data_pakdd.csv',
            'identifier': 'object_id',
            'features': ['feature1', 'feature2'],
             'feature_renames' : ['feature1', 'feature2'],
            'time_column_name': 'time',
            'no_timepoints': 8,
            'normalization': 'no',
            'start_year': 1,
            'end_year': 8,
            'minpts': 4,
            'eps': 0.11,
            'CLOSE': {
                 'minpts_start': 4,
                 'minpts_end': 4,
                 'minpts_step': 1,
                 'eps_start': 0.11,
                 'eps_end': 0.12,
                 'eps_step': 0.01
             },
            'tau': 0.5,
            'plot' : {
                'time_axis_name' : 'year',
                'col_wrap': 4,
                'no_timepoints': 8,
                'title' : 'Generated Data 2D',
                '1DPlot' : False,
            }}


financial = {'Name' : 'financial',
          #'features' : ['FF-SOFT_ASSETS', 'FF-PENSION0'],
          #'feature_renames' : ['SoftAssets', 'Pension'],
          'features' : ['TR-CapitalExpenditures', 'FF-LEASEDUM'],
          'feature_renames' : ['Expenditures', 'Leasdum'],
          # 'features' : ['TR-TtlPlanExpectedReturn', 'TR-NetSales'],
          # 'feature_renames' : ['ExpectedReturn', 'NetSales'],
          'normalization_feature': 'TR-TotalAssetsReported',
          'no_companies': 0,
          'start_year' : 2011,
          'end_year' : 2017,
          'minpts': 2,
          'eps': 0.08,
          'CLOSE' : {
                'minpts_start' : 2,
                'minpts_end' : 2,
                'minpts_step' : 1,
                'eps_start' : 0.005,
                'eps_end' : 0.006,
                'eps_step' : 0.001
          },
          # 'eps': 0.15,
          'tau' : 0.6,
           'plot': {
               'time_axis_name': 'year',
               'col_wrap': 3,
               'no_timepoints': 6,
               'title': 'Financial Data',
               '1DPlot' : False,
           }
}
financial_oned = {'Name' : 'financial',
          'features' : ['TR-Revenue'],
          'feature_renames' : ['TR-Revenue'],
          'normalization_feature': 'TR-TotalAssetsReported',
          'no_companies': 30,
          'start_year' : 2008,
          'end_year' : 2013,
          'minpts': 2,
          'eps': 0.1,
          'tau' : 0.2,
           'plot': {
               'time_axis_name': 'year',
               'col_wrap': 3,
               'no_timepoints': 6,
               'title': 'Generated Data',
               '1DPlot' : True,
           }
}

gps_data = {'Name' : 'gps_data',
          'path' : './gps_data/',
          'features' : ['Distance', 'AltDistance'],
          'feature_renames' : ['Distance', 'Altitude_distance'],
          'start_time' : 0,
          'end_time' : 24,
          'no_frames' : 8,
          'parse_user': '001',
          'no_days' : 30,
          'minpts': 2,
          'eps': 0.08,
           'plot': {
               'time_axis_name': 'Time',
               'col_wrap': 4,
               'no_timepoints': 8,
               'title': 'GPS Data',
               '1DPlot' : False,
           }
}