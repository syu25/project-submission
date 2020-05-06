# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:35:36 2020

@author: siddh
"""

import pandas as pd
import numpy as np
import math as mt
import sklearn as sk1
import datetime as dt
import matplotlib.pyplot as plt
import math 
import seaborn as plt1
from sklearn.model_selection import train_test_split
import pickle


testBeneData  = pd.read_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\\Data-Collection-and-Exploratory-Data-Analysis-master\\Test_Beneficiarydata-1542969243754.csv")
testInpatData = pd.read_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\\Data-Collection-and-Exploratory-Data-Analysis-master\\Test_Inpatientdata-1542969243754.csv")
testTagdData = pd.read_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\Data-Collection-and-Exploratory-Data-Analysis-master\\Test-1542969243754.csv")
                        

testBeneData["RenalDiseaseIndicator"]= testBeneData["RenalDiseaseIndicator"].replace("Y", "1")
testBeneData["RenalDiseaseIndicator"]= testBeneData["RenalDiseaseIndicator"].replace("N", "0")

testBeneData["RenalDiseaseIndicator"] = pd.to_numeric(testBeneData["RenalDiseaseIndicator"])

testBeneData['DOB'] = testBeneData['DOB'].astype('datetime64[ns]')
testBeneData['DOD'] = testBeneData['DOD'].astype('datetime64[ns]')

testBeneData.loc[(testBeneData['DOD'].isnull()),'DOD']= dt.datetime.today()
testBeneData['age'] = [math.ceil(td.days/365) for td in (testBeneData["DOD"] - testBeneData["DOB"])]

testBeneData = testBeneData.drop(columns = ['DOB','DOD'])

## Data  Discreetisation in testBeneData:
testBeneData.loc[testBeneData['ChronicCond_Alzheimer']==2,'ChronicCond_Alzheimer'] = 0
testBeneData.loc[testBeneData['ChronicCond_Heartfailure']==2,'ChronicCond_Heartfailure'] = 0
testBeneData.loc[testBeneData['ChronicCond_KidneyDisease']==2,'ChronicCond_KidneyDisease'] = 0
testBeneData.loc[testBeneData['ChronicCond_Cancer']==2,'ChronicCond_Cancer'] = 0
testBeneData.loc[testBeneData['ChronicCond_ObstrPulmonary']==2,'ChronicCond_ObstrPulmonary'] = 0
testBeneData.loc[testBeneData['ChronicCond_Depression']==2,'ChronicCond_Depression'] = 0
testBeneData.loc[testBeneData['ChronicCond_IschemicHeart']==2,'ChronicCond_IschemicHeart'] = 0
testBeneData.loc[testBeneData['ChronicCond_Osteoporasis']==2,'ChronicCond_Osteoporasis'] = 0
testBeneData.loc[testBeneData['ChronicCond_rheumatoidarthritis']==2,'ChronicCond_rheumatoidarthritis'] = 0
testBeneData.loc[testBeneData['ChronicCond_stroke']==2,'ChronicCond_stroke'] = 0
testBeneData.loc[testBeneData['ChronicCond_Diabetes']==2,'ChronicCond_Diabetes'] = 0

testInpatData = testInpatData.drop(columns=['ClaimStartDt','ClaimEndDt'])

# For conditions where AttendingPhy there but no operating and other phy 
BeneID_of_only_AttendingPhy = testInpatData.loc[(testInpatData['AttendingPhysician'].notnull()) & (testInpatData['OperatingPhysician'].isnull()),'BeneID']

for idx in BeneID_of_only_AttendingPhy:
    testInpatData.loc[((testInpatData['BeneID']==idx) & (testInpatData['AttendingPhysician'].notnull()) & (testInpatData['OperatingPhysician'].isnull())),'OperatingPhysician'] = testInpatData.loc[((testInpatData['BeneID']==idx) & (testInpatData['AttendingPhysician'].notnull()) & (testInpatData['OperatingPhysician'].isnull())),'AttendingPhysician']
    
    
# For conditions where no AttendingPhy there but operating there and no other phy 
BeneID_of_only_OperatingPhy = testInpatData.loc[((testInpatData['AttendingPhysician'].isnull()) & (testInpatData['OperatingPhysician'].notnull())),'BeneID']

for id1 in BeneID_of_only_OperatingPhy:
    testInpatData.loc[((testInpatData['BeneID']==id1) & (testInpatData['AttendingPhysician'].isnull()) & (testInpatData['OperatingPhysician'].notnull())),'AttendingPhysician'] = testInpatData.loc[((testInpatData['BeneID']==id1) & (testInpatData['AttendingPhysician'].isnull()) & (testInpatData['OperatingPhysician'].notnull())),'OperatingPhysician']
    
# For conditions where AttendingPhy there but no other phy  
BeneID_of_no_OtherPhy = testInpatData.loc[((testInpatData['AttendingPhysician'].notnull()) & (testInpatData['OtherPhysician'].isnull())),'BeneID']

for id2 in BeneID_of_no_OtherPhy:
    testInpatData.loc[((testInpatData['BeneID']==id2) & (testInpatData['AttendingPhysician'].notnull()) & (testInpatData['OtherPhysician'].isnull())),'OtherPhysician'] = testInpatData.loc[((testInpatData['BeneID']==id2) & (testInpatData['AttendingPhysician'].notnull()) & (testInpatData['OtherPhysician'].isnull())),'OtherPhysician']

Phy_dict = pd.read_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\\Physician_dict_df.csv")

if(Phy_dict.columns.contains("Unnamed: 0")):
    Phy_dict = Phy_dict.drop(columns=['Unnamed: 0'])

Phydict = Phy_dict.set_index('Index').to_dict()

print(Phydict)

Phydict = Phydict['Values']

try:
    testInpatData['AttendingPhysician'] = [Phydict[o] for o in testInpatData['AttendingPhysician']]
except KeyError as e:
    cause = e.args[0]
    Phydict['cause'] = len(Phydict)+1
    try:
        testInpatData['AttendingPhysician'] = [Phydict[o] for o in testInpatData['AttendingPhysician']]
    except KeyError as ex:
        cause1 = e.args[0]
        Phydict['cause1'] = len(Phydict)+1
        testInpatData['AttendingPhysician'] = [Phydict[o] for o in testInpatData['AttendingPhysician']]
    

try:    
    testInpatData['OperatingPhysician'] = [Phydict[o] for o in testInpatData['OperatingPhysician']]
except KeyError as e:
    cause = e.args[0]
    Phydict['cause'] = len(Phydict)+1
    try:
        testInpatData['OperatingPhysician'] = [Phydict[o] for o in testInpatData['OperatingPhysician']]
    except KeyError as ex:
        cause2 = e.args[0]
        Phydict['cause2'] = len(Phydict)+1
        testInpatData['OperatingPhysician'] = [Phydict[o] for o in testInpatData['OperatingPhysician']]
try:
    testInpatData['OtherPhysician'] = [Phydict[o] for o in testInpatData['OtherPhysician']]
except KeyError as e:
    cause = e.args[0]
    Phydict['cause'] = len(Phydict)+1
    try:
        testInpatData['OtherPhysician'] = [Phydict[o] for o in testInpatData['OtherPhysician']]
    except KeyError as ex:
        cause = e.args[0]
        Phydict['cause'] = len(Phydict)+1
        testInpatData['OtherPhysician'] = [Phydict[o] for o in testInpatData['OtherPhysician']]
        
Phy_df = testInpatData[['BeneID','AttendingPhysician','OperatingPhysician','OtherPhysician']].copy()
testBeneData = pd.merge(left=testBeneData, right = Phy_df, left_on="BeneID", right_on="BeneID",how='inner')


#Diagnosis GroupCode:
testInpatData.loc[testInpatData['DiagnosisGroupCode']=="OTH",'DiagnosisGroupCode'] = 0
DiagnosisCode_df = testInpatData[['BeneID','DiagnosisGroupCode']].copy()
testBeneData = pd.merge(left=testBeneData, right = DiagnosisCode_df, left_on="BeneID", right_on="BeneID",how='inner')

testBeneData['DiagnosisGroupCode'] = testBeneData['DiagnosisGroupCode'].astype('int')

# Discreetizing the AdmissionDt and Discharge dt of inpatients:
adm_dscdt_df = testInpatData[['BeneID','AdmissionDt','DischargeDt']].copy()

testBeneData = pd.merge(left = testBeneData, right = adm_dscdt_df,left_on='BeneID',right_on = 'BeneID',how = 'inner')
   #testBeneData = testBeneData.join(adm_dscdt_df,on = 'BeneID', how='inner')

testBeneData['AdmissionDt'] = testBeneData['AdmissionDt'].astype('datetime64[ns]')
testBeneData['DischargeDt'] = testBeneData['DischargeDt'].astype('datetime64[ns]')

testBeneData['Duration_of_stay'] = [math.ceil(rd.days) for rd in (testBeneData['DischargeDt'] - testBeneData['AdmissionDt'])]


testBeneData = testBeneData.drop(columns=['AdmissionDt','DischargeDt'])


# ClaimAdmitDiagnosisCode column Addition:
clmdiagcode_df = testInpatData[['BeneID','ClmAdmitDiagnosisCode']].copy()

clmdiagcode_df.loc[clmdiagcode_df['ClmAdmitDiagnosisCode'].isna(),'ClmAdmitDiagnosisCode'] = 'NO'

clmdiagcode_dict = pd.read_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\\ClmAdmitDiagnosisCodeDict_df.csv")

clmdiagdict = clmdiagcode_dict.set_index("Index").to_dict()

clmdiagdict = clmdiagdict['Values']

try:
    clmdiagcode_df['ClmAdmitDiagnosisCode'] = [clmdiagdict[o] for o in clmdiagcode_df['ClmAdmitDiagnosisCode']]
except KeyError as e1:
    cae = e1.args[0]
    clmdiagdict['cae'] = len(clmdiagdict) + 1 
    clmdiagcode_df['ClmAdmitDiagnosisCode'] = [clmdiagdict[o] for o in clmdiagcode_df['ClmAdmitDiagnosisCode']]

testBeneData = pd.merge(left = testBeneData,right = clmdiagcode_df,left_on = 'BeneID',right_on = 'BeneID', how = 'inner')


# Deductible Amt Paid Column:
AmtDeductible_df = testInpatData[['BeneID','DeductibleAmtPaid']].copy()

testBeneData = pd.merge(left = testBeneData,right = AmtDeductible_df,left_on = 'BeneID', right_on = 'BeneID',how= 'inner')
   #testBeneData = testBeneData.join(clmdiagcode_df,on = 'BeneID', how='inner')
testBeneData['DeductibleAmtPaid'].fillna(0.0,inplace =True)

testBeneData['DeductibleAmtPaid'].isnull().values.sum()

## ClmDiagnosisCode_1-10 columns:
ClmDiagnosisCodedict = pd.read_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\\For-feature-addition-DiagCodes\\transfer_list1.csv")

if ClmDiagnosisCodedict.columns.contains('Unnamed: 0'):
    ClmDiagnosisCodedict = ClmDiagnosisCodedict.drop(columns=['Unnamed: 0'])
    
ClmDiagCodesColumns = list(ClmDiagnosisCodedict['0'].copy())

ClmDiagCodesColumns.remove("BeneID")

for u in ClmDiagCodesColumns:
    testBeneData[u] = 0

testInpatData['ClmDiagnosisCode'] = 0
testInpatData['ClmDiagnosisCode'] = testInpatData['ClmDiagnosisCode'].astype('object')
    
for k in range(0,testInpatData.shape[0]):
  testInpatData.at[k,'ClmDiagnosisCode'] = pd.Series([testInpatData.at[k,'ClmDiagnosisCode_1'],testInpatData.at[k,'ClmDiagnosisCode_2'],testInpatData.at[k,'ClmDiagnosisCode_3'],testInpatData.at[k,'ClmDiagnosisCode_4'],testInpatData.at[k,'ClmDiagnosisCode_5'],testInpatData.at[k,'ClmDiagnosisCode_6'],testInpatData.at[k,'ClmDiagnosisCode_7'],testInpatData.at[k,'ClmDiagnosisCode_8'],testInpatData.at[k,'ClmDiagnosisCode_9'],testInpatData.at[k,'ClmDiagnosisCode_10']])
  testInpatData.at[k,'ClmDiagnosisCode'] = set(testInpatData.at[k,'ClmDiagnosisCode'])
  
DiagTest_df = testInpatData[['BeneID','ClmDiagnosisCode']].copy()
diagcodesdf = DiagTest_df.set_index('BeneID').to_dict()

diagcodesdf = diagcodesdf['ClmDiagnosisCode']

for idc in diagcodesdf.keys():
    ColList = ["ClmDiagCode_"+str(h) for h in diagcodesdf[idc]]
    for y in ColList:
        if testBeneData.columns.contains(y):
            testBeneData.loc[testBeneData['BeneID']==idc,y] = 1

## ClMProcedureCode:
testInpatData.loc[testInpatData['ClmProcedureCode_1'].isna(),'ClmProcedureCode_1'] = 0
testInpatData.loc[testInpatData['ClmProcedureCode_2'].isna(),'ClmProcedureCode_2'] = 0
testInpatData.loc[testInpatData['ClmProcedureCode_3'].isna(),'ClmProcedureCode_3'] = 0
testInpatData.loc[testInpatData['ClmProcedureCode_4'].isna(),'ClmProcedureCode_4'] = 0
testInpatData.loc[testInpatData['ClmProcedureCode_5'].isna(),'ClmProcedureCode_5'] = 0
testInpatData.loc[testInpatData['ClmProcedureCode_6'].isna(),'ClmProcedureCode_6'] = 0
clmDiagCodes_df1 = testInpatData[['BeneID','ClmProcedureCode_1', 'ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6']].copy()
testBeneData = pd.merge(left = testBeneData,right = clmDiagCodes_df1,left_on= 'BeneID', right_on='BeneID', how = 'inner') 


Providerdf = testInpatData[['BeneID','Provider']].copy()
testBeneData = pd.merge(left = testBeneData,right = Providerdf,left_on= 'BeneID', right_on='BeneID', how = 'inner')
# Provider Merge:
testBeneData = pd.merge(left = testBeneData,right = testTagdData,left_on= 'Provider', right_on='Provider', how = 'inner') 



#Columns left:
Providerdf1 = testInpatData[['BeneID','InscClaimAmtReimbursed','DiagnosisGroupCode']].copy()
testBeneData = pd.merge(left = testBeneData,right = Providerdf1,left_on= 'BeneID', right_on='BeneID', how = 'inner')

testBeneData.to_csv("C:\\Users\\siddh\\Downloads\\ML_Final_Project\\For-feature-addition-DiagCodes\\testData.csv")