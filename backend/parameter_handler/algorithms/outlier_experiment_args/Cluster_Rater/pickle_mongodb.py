import pickle
from pymongo import MongoClient
import time

QUARTERS = ['FQ0', 'FQ-1', 'FQ-2', 'FQ-3', 'FQ-4']

client = MongoClient('134.99.112.190',
                             username='read_user',
                             password='tepco11x?z',
                             authSource='finfraud3',
                             authMechanism='SCRAM-SHA-1',
                             port=27017)
db = client.finfraud3.original


def main():
    feature_set = ['TR-AccountsPayable',
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
    db_to_pickle(feature_set)


def db_to_pickle(feature_set):
    feature_set = ['ric'] + feature_set
    company_dict = dict()
    for document in db.find():
        doc = document.copy()
        for y in range(2011, 2018):
            if str(y) in document.keys():
                document[str(y)] = dict(document[str(y)])
                doc[str(y)] = dict(doc[str(y)])
                for k in document[str(y)].keys():
                    if k in QUARTERS:
                        doc[str(y)][k] = doc[str(y)][k].clear()
                    if k not in feature_set:
                        doc[str(y)].pop(k, None)

        company_dict[doc['ric']] = doc

    pickle.dump(company_dict, open('mongodb_dump.txt', 'wb'))


def read_pickle(filename):
    while True:
        try:
            return pickle.load(open(filename, 'rb'))
        except:
            time.sleep(1000)


if __name__ == "__main__":
    main()