from pymongo import MongoClient
from itertools import combinations

epsilon = [8] # will be divided by 100
min_pts = [2]
taus = [40] # will be devided by 100
num_features = [3]
start_year = 2011
end_year = 2017


FEATURE_SET = ['TR-AccountsPayable',
            'TR-AdvertisingExpense',
            'TR-CapitalExpenditures',
            'TR-CapitalLeaseObligation',
            'TR-CashandEquivalents',
            'TR-CashAndSTInvestments',
            'TR-CashFromFinancingAct',
            'TR-CashFromOperatingAct',
            'TR-CommonStockNet',
            'TR-CostOfRevenue',
            'TR-CostOfRevenueTotal',
            'TR-CostofRevenueTotal',
            'TR-DepreciationAmort',
            'TR-Employees',
            'TR-IncomeTaxesPayable',
            'TR-LTDebt',
            'TR-LTDebtIssued',
            'TR-LTDebtMaturingYear1',
            'TR-LTDebtNet',
            'TR-LTInvestments',
            'TR-NetIncome',
            'TR-NetIncomeAfterTaxes',
            'TR-NetIncomeBeforeTaxes',
            'TR-NetSales',
            'TR-PptyPlantEqpmtTtlGross',
            'TR-PreferredStockNet',
            'TR-ResearchAndDevelopment',
            'TR-Revenue',
            'TR-SaleIssuanceOfCommon',
            'TR-SaleIssuanceOfCommonPreferred',
            'TR-SgaExpenseTotal',
            'TR-ShortTermInvestments',
            'TR-TaxDefTot',
            'TR-TotalAssetsReported',
            'TR-TotalCurrentAssets',
            'TR-TotalCurrLiabilities',
            'TR-TotalEquity',
            'TR-TotalEquityAndMinorityInterest',
            'TR-TotalInventory',
            'TR-TotalLiabilities',
            'TR-TotalOperatingExpense',
            'TR-TotalOperatingLeasesSuppl',
            'TR-TotalReceivablesNet',
            'TR-TtlPlanExpectedReturn',
            'TR-TtlPreferredSharesOut',
            'TR-ValueBacklog',
            'FF-CFOError',
            'FF-COGSError',
            'FF-INVError',
            'FF-PRODError',
            'FF-DISEXPError',
            'FF-ACC_JONESError',
            'FF-ACC_KOTHARIError',
            'FF-WC_ACC',
            'FF-CH_REC',
            'FF-CH_INV',
            'FF-SOFT_ASSETS',
            'FF-CH_CS',
            'FF-CH_CM',
            'FF-ROA',
            'FF-FCF',
            'FF-TAX',
            'FF-CH_EMP',
            'FF-CH_BACKLOG',
            'FF-LEASEDUM',
            'FF-OPLEASE',
            'FF-PENSION',
            'FF-CH_PENSION',
            'FF-EXFIN',
            'FF-ISSUE',
            'FF-CFF',
            'FF-LEVERAGE',
            'FF-BM',
            'FF-EP',
            'FF-RSST_ACC']

# Obtain the features from MongoDB
client = MongoClient('134.99.112.190',
                     username='read_user',
                     password='tepco11x?z',
                     authSource='finfraud3',
                     authMechanism='SCRAM-SHA-1',
                     port=27017)
db = client.finfraud3.original

feature_list = []

for r in num_features:
    combs = combinations(FEATURE_SET, r)
    for combination in combs:
        feature_list.append(combination)


f = open("args" + str(num_features) + "_" + str(start_year)[2:] + "-" + str(end_year)[2:] + ".sh", "a")
f.write("export PARAMS=(")
counter=0
for eps in epsilon:
    for min in min_pts:
        for tau in taus:
            for comb in feature_list:
                param_string = "\"{'Name':'financial'," \
                      "'features':"+str(list(comb)).replace(' ','')+","\
                      "'feature_renames':"+str(list(comb)).replace(' ','')+","\
                      "'normalization_feature':'TR-TotalAssetsReported',"\
                      "'no_companies':0,"\
                      "'start_year':"+str(start_year)+","\
                      "'end_year':"+str(end_year)+","\
                      "'minpts':"+str(min)+","\
                      "'eps':"+str(float(eps/100))+","\
                      "'tau':"+str(float(tau/100))+"}\""
                f.write(param_string + " ")
                counter+=1
print(counter)
f.write(")")




