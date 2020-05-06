import pandas as pd 
import math
import numpy as np 


#takes in a csv file from the CDC causes of death dataset and returns a frame
#containing the data we want for our machine learning algorithms
def format_input(data, rows):
    df = pd.read_csv(data, nrows=rows,)

    #makes a dataframe with only desired columns
    df = df[["education_2003_revision", "month_of_death", "sex", "place_of_death_and_decedents_status", "marital_status", "manner_of_death", "injury_at_work", "race", "detail_age"]]
    
    #rename columns for better usability
    col_dict = {"education_2003_revision": "education", "month_of_death": "month", "place_of_death_and_decedents_status": "place", "marital_status": "marital", "manner_of_death": "manner", "injury_at_work": "inj_work", "detail_age": "age"}
    df.rename(columns = col_dict, inplace=True)

    #limits data to only natural deaths
    df = df[df.manner != "Blank"]
    #print(len(df))

    #remove unavailable age information
    df = df[df.age != 999]

    #converting all values to numbers for use in learning algorithms
    race_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 0: 10, 29: 11, 38: 12, 48: 13, 58: 14, 68: 15, 78: 16,}
    marital_dict = {'S': 1, 'M': 2, 'W': 3, 'D': 4, 'U': 5}
    sex_dict = {'M': 1, 'F': 2}
    inj_work_dict = {'Y': 1, 'N': 2, 'U': 3}
    df["inj_work"] = df["inj_work"].map(inj_work_dict)
    df["sex"] = df["sex"].map(sex_dict)
    df["marital"] = df["marital"].map(marital_dict)
    df["race"] = df["race"].map(race_dict)
    df.dropna(inplace=True)
   
    

    return df
