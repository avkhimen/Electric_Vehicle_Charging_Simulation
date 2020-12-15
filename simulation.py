import numpy as np
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import pdb
import argparse
import os

def get_input_args():
    """
    Returns input arguments for main file execution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type = int, default = 10,
                        help = 'Number of episodes to run')
    parser.add_argument('--id_run', type = str, default = 'test_run',
                        help = 'id of run')
    parser.add_argument('--pen', type = float, default = 0.1,
                        help = 'market penetration of evs')
    parser.add_argument('--avg_param', type = int, default = 1,
                        help = 'if avg == 1, non-one avg and non-zero max are used')
    parser.add_argument('--alpha', type = float, default = 0.01,
                        help = 'alpha for learning')
    parser.add_argument('--scale', type = int, default = 1,
                        help = 'scale')
    return parser.parse_args()

# Get args
n_episodes = get_input_args().n
id_run = get_input_args().id_run
pen = get_input_args().pen
avg = get_input_args().avg_param
alpha = get_input_args().alpha
scale = get_input_args().scale

# Get Alberta Average demand and prices
df = pd.read_csv('AESO_2020_demand_price.csv')
HE = []
end_index = df.shape[0]//(48 * 2) + 1
for day in range(1, end_index):
    for hour in range(1, (2 * 48) + 1):
        HE.append(hour)
df['HE'] = HE
df = df.drop(df.columns[[0, 2]], axis = 1)
df = df.set_index('HE', drop = True)
df = df.groupby('HE', as_index=True).mean()
df_to_plot = df.drop(df.columns[[0]], axis = 1)

alberta_avg_power_price = np.array(df.iloc[:, 0])
alberta_avg_demand = np.array(df.iloc[:, 1])/scale

# https://open.alberta.ca/dataset/d6205817-b04b-4360-8bb0-79eaaecb9df9/
# resource/4a06c219-03d1-4027-9c1f-a383629ab3bc/download/trans-motorized-
# vehicle-registrations-select-municipalities-2020.pdf
total_cars_in_alberta = 100
ev_market_penetration = 0.1
min_soc_by_8_am = 0.5
max_soc_allowed = 1
min_soc_allowed = 0.1
charging_soc_addition_per_time_unit_per_ev = 0.15
discharging_soc_reduction_per_time_unit_per_ev = -0.15
charging_soc_mw_addition_to_demand_per_time_unit_per_ev = 0.01
discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev = 0.01
driving_soc_reduction_per_time_unit_per_ev = 0.005
forecast_flag = False
n_percent_honesty = ['0.25', '0.5', '0.75']

# Time conversion
index_of_time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
time_of_day = [17,18,19,20,21,22,23,0,1,2,3,4,5,6,7]

index_to_time_of_day_dict = {}
for item in range(len(index_of_time)):
    index_to_time_of_day_dict[index_of_time[item]] = time_of_day[item]
pprint(index_to_time_of_day_dict)

# Define experiment params
experiment_params = {'n_episodes': n_episodes, 
                     'n_hours': 15, 
                     'n_divisions_for_soc': 4, 
                     'n_divisions_for_percent_honesty': 3,
                     'max_soc_allowed': 1,
                     'min_soc_allowed': 0.1,
                     'alpha': alpha,
                     'epsilon': 0.1,
                     'gamma': 1,
                     'total_cars_in_alberta': 1000000/scale,
                     'ev_market_penetration': pen,
                     'charging_soc_addition_per_time_unit_per_ev': 0.15, 
                     'discharging_soc_reduction_per_time_unit_per_ev': 0.15, 
                     'charging_soc_mw_addition_to_demand_per_time_unit_per_ev': 0.01, 
                     'discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev': 0.01,
                     'driving_soc_reduction_per_km_per_ev': 0.0035,
                     'alberta_average_demand': alberta_avg_demand,
                     'index_to_time_of_day_dict': index_to_time_of_day_dict,
                     'forecast_flag': forecast_flag,
                     'n_percent_honesty': n_percent_honesty,
                     'which_avg_param': avg
                    }

# Experiment function
class Experiment():
    
    def __init__(self, experiment_params={}):
        
        # Initialize all experiment params
        self.n_episodes = experiment_params.get('n_episodes')
        self.n_hours = experiment_params.get('n_hours')
        self.n_divisions_for_soc = experiment_params.get('n_divisions_for_soc')
        self.n_divisions_for_percent_honesty = experiment_params.get('n_divisions_for_percent_honesty')
        self.max_soc_allowed = experiment_params.get('max_soc_allowed')
        self.min_soc_allowed = experiment_params.get('in_soc_allowed')
        self.alpha = experiment_params.get('alpha')
        self.epsilon = experiment_params.get('epsilon')
        self.gamma = experiment_params.get('gamma')
        self.total_cars_in_alberta = experiment_params.get('total_cars_in_alberta')
        self.ev_market_penetration = experiment_params.get('ev_market_penetration')
        self.charging_soc_addition_per_time_unit_per_ev = experiment_params.get('charging_soc_addition_'\
                                                                           'per_time_unit_per_ev')
        self.discharging_soc_reduction_per_time_unit_per_ev = experiment_params.get('discharging_'\
                                                                               'soc_reduction_per_time_unit_per_ev')
        self.charging_soc_mw_addition_to_demand_per_time_unit_per_ev = experiment_params.get('charging_'\
                                                                                        'soc_mw_addition_to_demand_'\
                                                                                        'per_time_unit_per_ev') 
        self.discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev = experiment_params.get('discharging_'\
                                                                                              'soc_mw_reduction_'\
                                                                                              'from_demand_per_'
                                                                                              'time_unit_per_ev')
        self.driving_soc_reduction_per_km_per_ev = experiment_params.get('driving_soc_reduction_per_km_per_ev')
        self.alberta_average_demand = experiment_params.get('alberta_average_demand')
        self.index_to_time_of_day_dict = experiment_params.get('index_to_time_of_day_dict')
        self.forecast_flag = experiment_params.get('forecast_flag')
        self.n_percent_honesty = experiment_params.get('n_percent_honesty')
        self.which_avg_param = experiment_params.get('which_avg_param')
            
        # Initialize q-value table    
        self.Q = self.initialize_action_value()

        self.v_get_soc_bin = np.vectorize(self.get_soc_bin)
        self.v_get_soc_and_charging_load = np.vectorize(self.get_soc_and_charging_load)
        self.v_get_soc_and_discharging_load = np.vectorize(self.get_soc_and_discharging_load)
        self.v_get_soc_from_driving = np.vectorize(self.get_soc_from_driving)
        self.v_get_soc_of_evs = np.vectorize(self.get_soc_of_evs)
        
        # Display params
        print('Experiment parameters are: ')
        print(*experiment_params.items(), sep='\n')
            
    def start_experiment(self):
        """Initialize the experiment"""
        
        # Calculate the number of EVs in the province
        self.num_of_evs = self.total_cars_in_alberta * self.ev_market_penetration
             
        # Initialize an array of SOCs for each EV
        self.soc_of_evs = abs(np.random.normal(0.3, 0.1, int(self.num_of_evs)))
        self.soc_of_evs = self.v_get_soc_of_evs(self.soc_of_evs)
        
        # Initialize the last total load and average
        if  self.which_avg_param == 1:
            self.last_max_load = 10139.13/scale #alberta_avg_demand[8:17].max()
            self.last_average = 10052.55/scale #alberta_avg_demand[8:17].mean()
        else:
            self.last_max_load = 0 #alberta_avg_demand[8:17].max()
            self.last_average = 1 #alberta_avg_demand[8:17].mean()
        self.last_percent_honest = np.random.choice(self.n_percent_honesty)
        self.last_Q = self.Q.copy()

    def run(self):
        """Main method to run the experiment with initialized params"""
        
        # Monitor the trace as the program runs
        #pdb.set_trace()
        
        # Initialize stats lists
        self.reward_list = []
        self.average_list = []
        self.PAR_list = []
        self.max_load_list = []
        self.Q_change_list = []
        self.Q_list = []

        # Repeat for every episode
        for episode in tqdm(range(self.n_episodes), ncols=100):
            
            # Initialize the experiment
            self.start_experiment()
            
            # Repeat for every hour in the number of hours
            for hour in range(0, self.n_hours):
                #print('\n', 'Hour is: ', hour)

                # Calculate the percent honesty of people 
                percent_honest = self.last_percent_honest
                #print('Percent honest: ', percent_honest)
                
                if forecast_flag:
                    next_percent_honest = np.random.choice(self.n_percent_honesty, p = [0.25, 0.25, 0.25, 0.25])
                else:
                    if hour >= 9:
                        next_percent_honest = self.n_percent_honesty[-1]
                    else:
                        next_percent_honest = np.random.choice(self.n_percent_honesty)
                #print('Next percent honest: ', next_percent_honest)
                    
                # Get the SOC division for each EV
                soc_div_index = self.v_get_soc_bin(self.soc_of_evs)
                
                # Get the indicator which shows whether each EV is
                # keeping to its original intention and make sure 
                # its applied randomly to each EV via shuffling
                status_evs = np.concatenate((np.ones(int(self.num_of_evs * float(percent_honest))), np.zeros(int(self.num_of_evs * (1 - float(percent_honest))))), axis = 0)
                # status_evs = ([0] * int(self.num_of_evs * (1 - float(percent_honest))) 
                #               + [1] * int(self.num_of_evs * float(percent_honest)))
                np.random.shuffle(status_evs)
                
                # Dictionary keeping track of what actions
                # were taken for each SOC division
                div_to_action_dict = {}
                
                # Loop for every SOC division
                for soc_bin in range(0, self.n_divisions_for_soc):
                    
                    # Extract the q-value for the division, hour, 
                    # and percent of EVs st
                    Q = self.Q[soc_bin][hour][int(float(percent_honest)/0.25 - 1)]
                    
                    # Choose an action using a policy 
                    # (ex: epsilon-greedy)
                    action = self.choose_action(Q)
                    
                    # Calculate the load for each SOC division
                    if self.index_to_time_of_day_dict[hour] in [17,18,19,20,21,22,23,0,1,2]:

                        charging_load = 0
                        discharging_load = 0

                        driving_distance_of_evs = abs(np.random.normal(5, 5, int(self.num_of_evs)))
                        soc_reduction_for_evs = self.driving_soc_reduction_per_km_per_ev * driving_distance_of_evs

                        if action == 0:
                            self.soc_of_evs, charging_load_index = self.v_get_soc_and_charging_load(soc_bin, self.soc_of_evs, soc_div_index, status_evs, soc_reduction_for_evs)
                            charging_load = self.charging_soc_mw_addition_to_demand_per_time_unit_per_ev * charging_load_index.sum()

                        elif action == 1:
                            self.soc_of_evs, discharging_load_index = self.v_get_soc_and_discharging_load(soc_bin, self.soc_of_evs, soc_div_index, status_evs, soc_reduction_for_evs)
                            discharging_load = self.discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev * discharging_load_index.sum()
                        else:
                            self.soc_of_evs = self.v_get_soc_from_driving(soc_bin, self.soc_of_evs, soc_div_index, status_evs, soc_reduction_for_evs)

                        e_evs = charging_load - discharging_load

                    elif self.index_to_time_of_day_dict[hour] in [3,4,5,6,7]:

                        charging_load = 0
                        discharging_load = 0

                        driving_distance_of_evs = abs(np.random.normal(0, 0, int(self.num_of_evs)))
                        soc_reduction_for_evs = self.driving_soc_reduction_per_km_per_ev * driving_distance_of_evs

                        if action == 0:
                            self.soc_of_evs, charging_load_index = self.v_get_soc_and_charging_load(soc_bin, self.soc_of_evs, soc_div_index, status_evs, soc_reduction_for_evs)
                            charging_load = self.charging_soc_mw_addition_to_demand_per_time_unit_per_ev * charging_load_index.sum()

                        elif action == 1:
                            self.soc_of_evs, discharging_load_index = self.v_get_soc_and_discharging_load(soc_bin, self.soc_of_evs, soc_div_index, status_evs, soc_reduction_for_evs)
                            discharging_load = self.discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev * discharging_load_index.sum()
                        else:
                            self.soc_of_evs = self.v_get_soc_from_driving(soc_bin, self.soc_of_evs, soc_div_index, status_evs, soc_reduction_for_evs)

                        e_evs = charging_load - discharging_load
                    
                    #print(f'Divsion {division}, hour {hour}, load = {load_from_division}')
                    # Populate division-to-action dictionary
                    # to preserve the action that was picked
                    # for each SOC division
                    div_to_action_dict[soc_bin] = (action, e_evs)
                
                # Get next hour based on current hour
                next_hour = self.get_next_hour(hour)
                
                # Calculate the total load based on
                # the loads from each SOC division
                total_load = 0
                for div in div_to_action_dict.keys():
                    total_load += div_to_action_dict[div][1]
                
                # Calculate the total power demand by adding the
                # power demand with the additional demand from EVs
                total_load = max(total_load + self.alberta_average_demand[self.index_to_time_of_day_dict[hour]], 0)
                
                # Calculate the PAR ratio, the reward, the average
                # and the penalty
                #pdb.set_trace()
                average = ((hour + 9) * self.last_average + total_load) / (hour + 1 + 9)
                average_charge_penalty = self.get_final_soc_penalty(hour)
                new_max_load =  max(total_load, self.last_max_load)
                if average > 0:
                    PAR = new_max_load / average
                else:
                    PAR = 1
                reward = -PAR + average_charge_penalty
                
                # Update the qction-value function for each
                # SOC division, hour, and percent honesty
                for soc_bin in range(0, self.n_divisions_for_soc):

                    if hour < self.n_hours - 1:
                        delta = (reward 
                                 + self.gamma * np.max(self.Q[soc_bin][next_hour][int(float(next_percent_honest)/0.25-1)])
                                 - self.Q[soc_bin][hour][int(float(percent_honest)/0.25-1)][div_to_action_dict[soc_bin][0]])
                        self.Q[soc_bin][hour][int(float(percent_honest)/0.25-1)][div_to_action_dict[soc_bin][0]] += self.alpha * delta
                    else:
                        delta = reward - self.Q[soc_bin][hour][int(float(percent_honest)/0.25-1)][div_to_action_dict[soc_bin][0]]
                        self.Q[soc_bin][hour][int(float(percent_honest)/0.25-1)][div_to_action_dict[soc_bin][0]] += self.alpha * delta
                
                # Store the total load, PAR, and
                # last percent honest
                self.last_max_load = new_max_load
                self.last_average = average
                self.last_percent_honest = next_percent_honest

            # print stats
            print('\n')
            print('Run name: ', id_run)
            print('Path and file: ', os.path.abspath(__file__))
            print('Last max load: ', self.last_max_load)
            print('Last average: ', self.last_average)
            print('Reward: ', reward)
            print('PAR: ', PAR)

            # Record stats
            self.reward_list.append(reward)
            self.average_list.append(average)
            self.PAR_list.append(PAR)
            self.max_load_list.append(new_max_load)
            self.Q_change_list.append(self.compare_Q())
            self.last_Q = self.Q.copy()
            if episode % 1000 == 0:
                self.Q_list.append(self.Q.copy())
        
        #print(self.Q)
        # Save statistics
        np.save('average_runs/' + id_run + '_reward_list.npy', self.reward_list)
        np.save('average_runs/' + id_run + '_average_list.npy', self.average_list)
        np.save('average_runs/' + id_run + '_Q.npy', self.Q)
        np.save('average_runs/' + id_run + '_PAR_list.npy', self.PAR_list)
        np.save('average_runs/' + id_run + '_max_list.npy', self.max_load_list)
        np.save('average_runs/' + id_run + '_Q_change_list.npy', self.Q_change_list)
        np.save('average_runs/' + id_run + '_Q_list.npy', self.Q_list)

    # Initialize action-values array
    def initialize_action_value(self):

        Q = np.zeros(shape = (self.n_divisions_for_soc, self.n_hours, self.n_divisions_for_percent_honesty, 3))
        return Q

    # Choose action using epsilon-greedy    
    def choose_action(self, Q):
        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = self.argmax(Q)
            
        return action
    
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)
    
    # Get the next hour based on
    # the current hour
    def get_next_hour(self, hour):

        if hour < 23:
            next_hour = hour + 1
        else:
            next_hour = 0
            
        return next_hour
    
    # Get the SOC bin
    # based on the SOC
    def get_soc_bin(self,x):
    
        if x <= 0.25:
            index = 0
        elif x <= 0.5:
            index = 1
        elif x <= 0.75:
            index = 2
        elif x <= 1.0:
            index = 3
        
        return index

    def get_soc_and_charging_load(self,division,
                              soc_of_evs,
                              soc_div_index,
                              status_evs,
                              soc_reduction_for_evs):
        if soc_div_index == division:
            if status_evs == 1:
                if soc_of_evs < 1:
                    new_soc = min(1, soc_of_evs + self.charging_soc_addition_per_time_unit_per_ev)
                    return new_soc, 1
                else:
                    return soc_of_evs, 0
            elif status_evs == 0:
                if soc_of_evs > 0.1:
                    new_soc = soc_of_evs  - soc_reduction_for_evs
                    return new_soc, 0
                else:
                    return soc_of_evs, 0
        else:
            return soc_of_evs, 0
        
    def get_soc_and_discharging_load(self, division,
                                     soc_of_evs,
                                     soc_div_index,
                                     status_evs,
                                     soc_reduction_for_evs):
        if soc_div_index == division:
            if status_evs == 1:
                if soc_of_evs >= 0.1:
                    new_soc = soc_of_evs - self.charging_soc_addition_per_time_unit_per_ev
                    return new_soc, 1
                else:
                    return soc_of_evs, 0
            elif status_evs == 0:
                if soc_of_evs > 0.1:
                    new_soc = soc_of_evs  - soc_reduction_for_evs
                    return new_soc, 0
                else:
                    return soc_of_evs, 0
        else:
            return soc_of_evs, 0
        
    def get_soc_from_driving(self, division,
                             soc_of_evs,
                             soc_div_index,
                             status_evs,
                             soc_reduction_for_evs):
        if soc_div_index == division:
            if status_evs == 1:
                return soc_of_evs
            elif status_evs == 0:
                if soc_of_evs > 0.1:
                    new_soc = soc_of_evs - soc_reduction_for_evs
                    return new_soc
                else:
                    return soc_of_evs
        else:
            return soc_of_evs

    def get_final_soc_penalty(self, hour):
        
        penalty = 0

        if hour >= 12 and hour < 15:
            mu = np.mean(self.soc_of_evs)
            if mu >= 0.48 - (14 - hour) * self.charging_soc_addition_per_time_unit_per_ev:
                penalty = 3
            else:
                penalty = -1

        return penalty

    def compare_Q(self):
        change_sum = 0
        for i in range(3):
            l_2 = []
            for row in self.last_Q:
                l = []
                for column in row:
                    max_action = np.argmax(column[i])
                    l.append(max_action)
                l_2.append(np.array(l))
            l_Q1 = np.array(l_2)
        
            l_2 = []
            for row in self.Q:
                l = []
                for column in row:
                    max_action = np.argmax(column[i])
                    l.append(max_action)
                l_2.append(np.array(l))
            l_Q2 = np.array(l_2)
            change_sum += np.count_nonzero(l_Q1-l_Q2)
        
        return change_sum
    
    def get_soc_of_evs(self, x):
        if x > 1:
            return 1
        elif x < 0:
            return 0
        else:
            return x
        
#         for ii, ev in enumerate(self.soc_of_evs):
#             if ev > 1:
#                 self.soc_of_evs[ii] = 1.0
#             elif ev < 0:
#                 self.soc_of_evs[ii] = 0.0

if __name__ == '__main__':
    # Run experiment
    experiment = Experiment(experiment_params)
    experiment.run()
