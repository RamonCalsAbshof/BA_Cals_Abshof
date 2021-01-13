from pymongo import MongoClient
from itertools import combinations

epsilon = range(8, 18, 2) # will be divided by 100
min_pts = range(2, 6)
taus = range(40, 70, 5) # will be devided by 100
num_features = [3]


# FEATURE_SET = ['TR-AccountsPayable',
# 'TR-AdvertisingExpense',
# 'TR-CapitalExpenditures',
# 'TR-CapitalLeaseObligation',
# 'TR-CashandEquivalents',
# 'TR-CashAndSTInvestments',
# 'TR-CashFromFinancingAct',
# 'TR-CashFromOperatingAct',
# 'TR-CommonStockNet',
# 'TR-CostOfRevenue',
# 'TR-CostOfRevenueTotal',
# 'TR-CostofRevenueTotal',
# 'TR-DepreciationAmort',
# 'TR-Employees',
# 'TR-IncomeTaxesPayable',
# 'TR-LTDebt',
# 'TR-LTDebtIssued',
# 'TR-LTDebtMaturingYear1',
# 'TR-LTDebtNet',
# 'TR-LTInvestments',
# 'TR-NetIncome',
# 'TR-NetIncomeAfterTaxes',
# 'TR-NetIncomeBeforeTaxes',
# 'TR-NetSales',
# 'TR-PptyPlantEqpmtTtlGross',
# 'TR-PreferredStockNet',
# 'TR-ResearchAndDevelopment',
# 'TR-Revenue',
# 'TR-SaleIssuanceOfCommon',
# 'TR-SaleIssuanceOfCommonPreferred',
# 'TR-SgaExpenseTotal',
# 'TR-ShortTermInvestments',
# 'TR-TaxDefTot',
# 'TR-TotalAssetsReported',
# 'TR-TotalCurrentAssets',
# 'TR-TotalCurrLiabilities',
# 'TR-TotalEquity',
# 'TR-TotalEquityAndMinorityInterest',
# 'TR-TotalInventory',
# 'TR-TotalLiabilities',
# 'TR-TotalOperatingExpense',
# 'TR-TotalOperatingLeasesSuppl',
# 'TR-TotalReceivablesNet',
# 'TR-TtlPlanExpectedReturn',
# 'TR-TtlPreferredSharesOut',
# 'TR-ValueBacklog']

FEATURE_SET = ['FF-CFOError', 'FF-COGSError', 'FF-INVError', 'FF-PRODError', 'FF-DISEXPError',
               'FF-ACC_JONESError', 'FF-ACC_KOTHARIError',
               'FF-WC_ACC', 'FF-CH_REC', 'FF-CH_INV', 'FF-SOFT_ASSETS', 'FF-CH_CS', 'FF-CH_CM', 'FF-ROA', 'FF-FCF',
               'FF-TAX', 'FF-CH_EMP', 'FF-CH_BACKLOG', 'FF-LEASEDUM', 'FF-OPLEASE', 'FF-PENSION', 'FF-CH_PENSION', 'FF-EXFIN',
               'FF-ISSUE', 'FF-CFF', 'FF-LEVERAGE', 'FF-BM', 'FF-EP', 'FF-RSST_ACC']

# Obtain the features from MongoDB
client = MongoClient('134.99.112.190',
                     username='read_user',
                     password='tepco11x?z',
                     authSource='finfraud3',
                     authMechanism='SCRAM-SHA-1',
                     port=27017)
db = client.finfraud3.original
#document = dict(db.find_one({'ric' : 'WMT'}))


feature_list = []
#document = dict(document['2018'])
#document = ['aa', 'bb', 'cc', 'cc', 'dd', 'ee']
#create all combinations of num_features length
for r in num_features:
    combs = combinations(FEATURE_SET, r)
    for combination in combs:
        feature_list.append(combination)

param_string="{'Name':'financial',"\
"'features':['FF-SOFT_ASSETS','FF-PENSION0'],"\
"'feature_renames':['SoftAssets','Pension'],"\
"'normalization_feature':'TR-TotalAssetsReported',"\
"'no_companies':0,"\
"'start_year':2008,"\
"'end_year':2013,"\
"'minpts':2,"\
"'eps':0.08,"\
"'tau':0.6}"

f = open("args_ff3.sh", "a")
f.write("export PARAMS=(")
for eps in epsilon:
    for min in min_pts:
        for tau in taus:
            for comb in feature_list:
                param_string = "\"{'Name':'financial'," \
                      "'features':"+str(list(comb)).replace(' ','')+","\
                      "'feature_renames':"+str(list(comb)).replace(' ','')+","\
                      "'normalization_feature':'TR-TotalAssetsReported',"\
                      "'no_companies':0,"\
                      "'start_year':2008,"\
                      "'end_year':2016,"\
                      "'minpts':"+str(min)+","\
                      "'eps':"+str(float(eps/100))+","\
                      "'tau':"+str(float(tau/100))+"}\""
                f.write(param_string + " ")
f.write(")")




