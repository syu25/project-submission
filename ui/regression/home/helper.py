import json
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

#Call the api or read from file
def load_from_api():
    #returns a dictionary of (id,first_name) mapped to {features mapped to lists where indicies rep game number }
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")#Change url to api we are using
    content_string = response.text
    json_content = json.loads(content_string)
    elements = json_content["elements"]
    player_id = {}
    count = 0
    for player in elements:
        if ('first_name' in player):
            response_1 = requests.get('https://fantasy.premierleague.com/api/element-summary/' + str(player['id']) + '/')
            player_data = response_1.text
            json_player_data = json.loads(player_data)
            if ('history' in json_player_data):
                points = json_player_data['history']
                game_number = []
                player_points = []
                time_played = []
                clean_sheets = []
                goals_scored = []
                #time_played = json_player_data
                for i,p in enumerate(points):
                    #games are in order
                    game_number.append(i)
                    player_points.append(p['total_points'] + 1)
                    time_played.append(p['minutes'])
                    clean_sheets.append(p['clean_sheets'])
                    goals_scored.append(p['goals_scored'])
                # print(player_points)
                # print("Hi")
                name = (player['id'], player['first_name']+ ' '+player['second_name'])
                values = {"points":player_points, "game_number":game_number, "time": time_played, "sheets":clean_sheets, "goals":goals_scored}
                player_id[name] = values
                #clean_sheets, goals_scored, minutes(Time played)
        count += 1
    return player_id


def calculate_average(player_points_dictionary):
    new_dictionary = {}
    for player,points in player_points_dictionary.items():
        length = len(points)
        average = 0
        if(length == 0):
            average = 0
        else:
            my_sum = 0
            for point in points:
                my_sum = my_sum + point
            average = my_sum/length
        new_dictionary[player] = average
    return new_dictionary

def convert_func(player_points_dictionary):
    """
    returns
    X_train
    Y_train -> an
    Mapping_list ->list of players
    """
    X_train = []
    Y_train = []
    Mapping_list = []
    for player,values in player_points_dictionary.items():
        x_bad_format = []
        for i, val in enumerate(values["goals"]):
            x_bad_format.append(values['game_number'][i])
            x_bad_format.append(values['goals'][i])
            x_bad_format.append(values['sheets'][i])
            x_bad_format.append(values['time'][i])
        x = np.array(np.array(x_bad_format))
        num_of_features = len(values)
        new_x = x.reshape(len(values['goals']),num_of_features-1)
        if(new_x.shape[0]<28):
            continue
        X_train.append(new_x)
        Y_train.append(np.array(values["points"]))
        Mapping_list.append(player)
    return X_train,Y_train,Mapping_list

def next_game_averages(X_train):
    X_pred = []
    for player_data in X_train:
        predicted_data = np.mean(player_data, axis=0)
        predicted_data[0] = player_data[-1][0]+1
        X_pred.append(predicted_data)
    return X_pred



def linear_reg(X_train,Y_train,X_pred):
    """
    X_train: list where indicies are people and elements are (a,b) matrix where a = num weeks, b= num features
    Y_train: list ''                ''      and elements are (a,b) matrix where a = num weeks, b =1: elements are points
    retval: list of predicted next week, indicies are the same with X_traiin,Y_train
    so far we assume 1 feature
    """
    result = []
    for list_index,X_matrix in enumerate(X_train):
        reg = LinearRegression().fit(X_matrix,Y_train[list_index])
        new_y = np.atleast_2d(X_pred[list_index])
        pred = reg.predict(new_y)
        result.append(pred[0])
    return result

def turn_into_dic(Mappings, Y_pred):
    new_dic = {}
    for i in range(len(Mappings)):
        new_dic[Mappings[i]] = Y_pred[i]
    return new_dic

def write_to_file(dictionary):
    with open('data.json', 'w') as fp:
        json.dump(dictionary, fp)

def read_from_file():
    with open('data.json') as json_file:
        data = json.load(json_file)
    return data
    

def find_index_of_player(player_mappings,player_id = -1, player_first_name = "" ):
    if(player_id == -1):
        struct_val = 1
        element = player_first_name
    else:
        struct_val = 0
        element = player_id
    return_index = -1
    for i, val in enumerate(player_mappings):
        if val == element:
            return i

def linear_reg_single(index_of_player, X_train_dataset, Y_train_dataset):
    return LinearRegression().fit(X_train_dataset[index_of_player],Y_train_dataset[index_of_player])

def start_and_duaration(X_train,Y_train, start, duration):
    #this takes the training sets, cuts appropriate games out, then returns the new training sets
    #start is an index of games, duration is an integer for how many games
    #for example: start = 0 and duration = 3 means we are going to consider game 1,2 and 3 and try 
    #and try to predict 4
    X_train_new = []
    Y_train_new = []
    for index,X_matrix in enumerate(X_train):
        X_train_new.append(X_matrix[start:start+duration][:])
        Y_train_new.append(Y_train[index][start:start+duration])
    return X_train_new, Y_train_new

def sort_dictionary_by_value(dictionary):
    return {key: dictionary[key] for key in sorted(dictionary, key=dictionary.get, reverse=True)}



def test():
    count = 0
    start = 4
    duration = 5
    end = 4+ 5
    player_id = read_from_file()
    X,Y,Mappings = convert_func(player_id)
    X_train, Y_train = start_and_duaration(X,Y,start,duration)
    X_pred = next_game_averages(X_train)
    Y_pred = linear_reg(X_train,Y_train, X_pred)
    players_new = turn_into_dic(Mappings,Y_pred)
    players_new = sort_dictionary_by_value(players_new)
    
    print("top 10:")
    for i,v in enumerate(players_new):
        print(i+1,":",v,"value of :",players_new[v])
        if(i == 9):
            break
    print("actual top 10:")
    player_actual = []
    for i, Y_value in enumerate(Y):
        player_actual.append((Mappings[i],Y_value[end]))
    player_actual.sort(key=lambda x: x[1], reverse=True)
    for i,v in enumerate(player_actual):
        print(i+1,":",v[0],"value of :",v[1])
        if(i == 9):
            break

#never mind
def write_dataset():
    player_id = load_from_api()
    write_to_file(player_id)
'''
step 1: either load the dataset with load_from_api or call write_dataset and then read_from_file
step 2: call master_run_function and pass the dataset in
step 3: grab variables and output them somehow
'''

def master_run_func(start,duration,data_set):
    '''
    parameters: start: index of start e.g  =0 means we start at the first game
                duration:how many games long e.g. =4 means there are 4 games taken into account
                data_set: the data set

    returns in this order predicted, actual, mappings, accuracy, top10pred, top10actual
        predicted:      a list of numbers
                        :all 900 predicted results of the run, indicies follow mappings data so 
                         predicted[i] is the predicted value of mappings[i] at time start+duration
        actual:         a list of numbers
                        :all 900 actual points of the run, indicies follow mappings data so 
                         actual[i] is the actual value of mappings[i] at time start+duration
        accuracy:        a list of numbers
                        :this corresponds to the accuracy of the players, similar structure to
                        predicted and actual
        top10pred:      a list of 4-ples, so [(name,predicted,actual,accuracy),(name,pred,actual,accuracy)...]
                        :this is the top 10 players that are predicted, top10pred[0] is first place
        top10actual:    similar to top10pred except the top 10 player are the actual not predicted
    '''
    end = start+duration
    X,Y,mappings = convert_func(data_set)
    X_train, Y_train = start_and_duaration(X,Y,start,duration)
    X_pred = next_game_averages(X_train)
    predictedFloat = linear_reg(X_train,Y_train, X_pred)
    predicted = []
    for p in predictedFloat:
        predicted.append(round(p))
    actual = []
    accuracy = []
    for i, Y_value in enumerate(Y):
        actual.append(Y_value[end])
        #accuracy->
        #accuracy.append((Y_value[end]-predicted[i])/Y_value[end])
        p = predicted[i]
        a = Y_value[end]
        if a == 0:
            a = 1
        accuracy.append(abs((p - a) / a) * 100)
    combined_list = []
    for i,val in enumerate(mappings):
        combined_list.append((val[1],predicted[i],actual[i]))
    #sort based on predicted
    combined_list.sort(key=lambda x: x[1], reverse=True)
    top10pred = []
    for i,v in enumerate(combined_list):
        top10pred.append(v)
        if i >=10:
            break
    combined_list.sort(key=lambda x: x[2], reverse=True)
    top10actual = []
    for i,v in enumerate(combined_list):
        top10actual.append(v)
        if i >=10:
            break
    return predicted,actual,accuracy,top10pred,top10actual