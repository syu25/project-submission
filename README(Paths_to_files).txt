POWERPOINT SLIDES:
	DETECTION OF FRAUD IN MEDICARE INPATIENT CLAIM DATA
YouTube Link:
	Youtube link.pdf


INITIAL DATA:
	Data-Collection-and-Exploratory-Data-Analysis-master

DATA TRANSFORMATION AND CLEANING:
	1. Final_Report_CSE574_Updated_Final
	2. Code_Base.ipnyb
	3. Dictonaries for Label Encoded Columns:
		Dictionaries_for_processing--------------Contains all the dictionaries used for Label Encoded columns									(Physicians,clmDiagnosisCodes_1-10,ClmAdmitDiagCodes)

DATA BEALANCING:
	1. Final_Report_CSE574_Updated_Final
	2. Code_Base.ipynb

FINAL DATASETS AFTER DATA TRANSFORMATIONS: 
	FinalDataSets_After_transformation:
		work_dat_notDead_large-Final model specific----- The set with BeneID removed and Provider label Encoded so 								 cannot be retraced back to the original beneficiaries.
								 But its completely fit to direct scale and Model

		work_dat_notDead_large-whole-------------------- This is the dataset with just the transformed data with 									 BeneID

DATA INSIGHTS:
       UI folder---------1. Correlation_Insights_data.ipynb(UI folder)
			 2. Relation-UI---- folder

DATA SCALING AND MODELLING:
       Models-----folder-----(here all the models with their trained models and scaler have been saved.)

UI:
       UI folder---------Models_UI(HTML page(Visualization_initiator_models) for Models page and all its componenets)
			 Relation_UI(HTML page(Visualization_initiator) for Relations page and all its componenets)


For Future use of models:-------Models folder has all the models trained along with the scaler used.\
				UI folder has the NewDataModNB.py file that can convert any RAW csv file in same format as 				the initial datasets can convert the data to transformed data without the scaling.			