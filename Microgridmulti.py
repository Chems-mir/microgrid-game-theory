import Microgrid as M
import random
import numpy as np
import copy
from gym.spaces import Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Microgridmulti(MultiAgentEnv):
    def __init__(self, config):
        
        self.eval = False
        self.liste_microgrids = config['L_microgrids']
        self.liste_prix = config['liste_prix']
        self.nb_microgrids = len(self.liste_microgrids)
        self.liste_microgrids_init = copy.deepcopy(self.liste_microgrids)
        self.current_timestep = 0
        self.max_timesteps = 23
        
        self.liste_microgrids_buyers = []
        self.liste_microgrids_sellers = []

        #for one agent:
        self.action_space = Discrete(len(self.liste_prix)) #correspond à l'indice dans la liste de prix possible
        self.observation_space = Box(low=np.array([-1, 0 ,0, 0, 0, 0, 0], dtype=np.float32), high=np.array([1, 10, 100, 100, 100, 100, 1], dtype=np.float32), shape=(7,), dtype=np.float32)
        
        self.info = {(i,j): {} for i in range(self.nb_microgrids) for j in range(self.liste_microgrids[i].nb_agents)} #i correspond au num du microgrid, et j au numero de house
    

    def get_observation(self, microgrid_id, agent_id):
        if self.liste_microgrids[microgrid_id].agents[agent_id].statu == 'seller':
            return np.array([1, self.liste_microgrids[microgrid_id].agents[agent_id].supply , self.liste_microgrids[microgrid_id].Demand_total_old, self.liste_microgrids[microgrid_id].Demand_total, self.liste_microgrids[microgrid_id].Supply_total_old,  self.liste_microgrids[microgrid_id].Supply_total, self.liste_microgrids[microgrid_id].avg_price_old], dtype=np.float32)
        elif self.liste_microgrids[microgrid_id].agents[agent_id].statu == 'buyer':
            return np.array([-1, self.liste_microgrids[microgrid_id].agents[agent_id].demand, self.liste_microgrids[microgrid_id].Demand_total_old,  self.liste_microgrids[microgrid_id].Demand_total, self.liste_microgrids[microgrid_id].Supply_total_old,  self.liste_microgrids[microgrid_id].Supply_total, self.liste_microgrids[microgrid_id].avg_price_old], dtype=np.float32)
        else:
            return np.array([0, 0, self.liste_microgrids[microgrid_id].Demand_total_old, self.liste_microgrids[microgrid_id].Demand_total, self.liste_microgrids[microgrid_id].Supply_total_old, self.liste_microgrids[microgrid_id].Supply_total, self.liste_microgrids[microgrid_id].avg_price_old], dtype=np.float32)

    def get_reward(self, microgrid_id, agent_id):
        return self.liste_microgrids[microgrid_id].agents[agent_id].payoff

    def is_done(self, microgrid_id, agent_id):
        if self.current_timestep > self.max_timesteps :
            return True
        else:
            return False


    def reset(self):
        res ={}
        for m in range(len(self.liste_microgrids)):
            d = self.liste_microgrids[m].reset() #retourne un dictionnaire d'observation du microgrid
            for key, value in d.items():
                res[(m,key)] = value  
        return res

    def closest_pair_microgrids(self):

        if len(self.liste_microgrids_buyers) == 0 or len(self.liste_microgrids_sellers) == 0:
            return None  
        # Sort the lists in ascending order based on the 'number' attribute
        self.liste_microgrids_buyers.sort(key=lambda x: x.avg_price)
        self.liste_microgrids_sellers.sort(key=lambda x: x.avg_price)

        closest_pair = None
        min_distance = float('inf')

        # Iterate through elements in liste_microgrids_buyers, comparing with a limited window in liste_microgrids_sellers
        for i, x in enumerate(self.liste_microgrids_buyers):
            # Define a search window in liste_microgrids_sellers based on the current element in liste_microgrids_buyers
            low = max(0, i - len(self.liste_microgrids_sellers) // 2)  # Lower bound (avoid negative index)
            high = min(len(self.liste_microgrids_sellers), i + len(self.liste_microgrids_sellers) // 2 + 1)  # Upper bound (avoid exceeding list length)

            for j in range(low, high):
                y = self.liste_microgrids_sellers[j]
                distance = abs(x.avg_price - y.avg_price)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (x, y)

            # Early termination if a pair with distance 0 is found
            if min_distance == 0:
                break

        return closest_pair


    def step(self, action_dict):
        self.current_timestep +=1
        Liste_payoffs = [[None] * self.liste_microgrids[m].nb_agents for m in range(self.nb_microgrids)]
        # Execute action
        print("current max_timesteps:", self.current_timestep-1)
        for m in range(len(self.liste_microgrids)):
            print(self.liste_microgrids[m])
            for i in range(self.liste_microgrids[m].nb_agents):
                self.liste_microgrids[m].agents[i].price = self.liste_prix[action_dict[(m,i)]]
        
            self.liste_microgrids[m].Demand_total_old = self.liste_microgrids[m].Demand_total
            self.liste_microgrids[m].Supply_total_old = self.liste_microgrids[m].Supply_total

            #determination des coeff lambda and avg
            self.liste_microgrids[m].avg_price = self.liste_microgrids[m].get_weighted_moy()
            for i in range(self.liste_microgrids[m].nb_agents):
                self.liste_microgrids[m].info[(m,i)] = {"avg_price": self.liste_microgrids[m].avg_price, "agent_price": self.liste_microgrids[m].agents[i].price}

            #calcul des payoffs
            nb_agents_actifs = 0
            Liste_payoffs[m] = self.liste_microgrids[m].payoffs(self.liste_microgrids[m].avg_price)
            for id in range(self.liste_microgrids[m].nb_agents):
                self.liste_microgrids[m].agents[id].payoff += Liste_payoffs[m][id]
                if self.liste_microgrids[m].agents[id].statu != 'observator':
                    nb_agents_actifs += 1

            redistribution_fees_per_house = self.liste_microgrids[m].penalization_total / nb_agents_actifs
            self.liste_microgrids[m].distributions_fees(redistribution_fees_per_house)

            self.liste_microgrids[m].avg_price_old = self.liste_microgrids[m].avg_price

            #Check statu of microgrid for microgrid level exchange
            self.liste_microgrids[m].microgrid_energy = self.liste_microgrids[m].Supply_total - self.liste_microgrids[m].Demand_total 
            
            if self.liste_microgrids[m].microgrid_energy > 0:
                self.liste_microgrids[m].microgrid_statu = 'surplus'
            elif self.liste_microgrids[m].microgrid_energy < 0:
                self.liste_microgrids[m].microgrid_statu = 'shortage'
            else:
                self.liste_microgrids[m].microgrid_statu = 'ok'

        self.liste_microgrids_buyers = []
        self.liste_microgrids_sellers = []
        #Transaction at microgrids level :
        for i in self.liste_microgrids:
            if i.microgrid_statu == 'shortage':
                self.liste_microgrids_buyers.append(i)
            elif i.microgrid_statu == 'surplus':
                self.liste_microgrids_sellers.append(i)
            else:
                pass


        while(len(self.liste_microgrids_sellers) > 0 and len(self.liste_microgrids_buyers) > 0):
            
            microgrid_buyer, microgrid_seller = self.closest_pair_microgrids()
            print('m buyer', microgrid_buyer)
            print('m seller: ', microgrid_seller)

            qtity_total = min(abs(microgrid_buyer.microgrid_energy), microgrid_seller.microgrid_energy)
            qtity_total_s = qtity_total

            microgrid_buyer.microgrid_energy += qtity_total
            microgrid_seller.microgrid_energy -= qtity_total_s

            nb_buyers = len(microgrid_buyer.liste_buyers)
            nb_sellers = len(microgrid_seller.liste_sellers)
            print('nbbyers, nbsellers', nb_buyers,nb_sellers)
            qtity_per_buyer = qtity_total/nb_buyers
            qtity_per_seller = qtity_total_s/nb_sellers

            microgrid_buyer.liste_buyers = sorted(microgrid_buyer.liste_buyers, key=lambda x: x.price)
            microgrid_buyer.liste_sellers = sorted(microgrid_buyer.liste_sellers, key=lambda x: x.price)

            
            for k in range(len(microgrid_buyer.liste_buyers)):
                if qtity_per_buyer > microgrid_buyer.liste_buyers[k].demand:
                    qtity_total -= microgrid_buyer.liste_buyers[k].demand
                    qtity_per_buyer = qtity_total/(nb_buyers-k)
                    microgrid_buyer.liste_buyers[k].demand = 0
                    microgrid_buyer.liste_buyers[k].payoff -= microgrid_seller.avg_price * microgrid_buyer.liste_buyers[k].demand 
                else:
                    qtity_total -= qtity_per_buyer
                    microgrid_buyer.liste_buyers[k].demand -= qtity_per_buyer
                    microgrid_buyer.liste_buyers[k].payoff -= microgrid_seller.avg_price * qtity_per_buyer

            for k in range(len(microgrid_seller.liste_sellers)):
                if qtity_per_seller > microgrid_seller.liste_sellers[k].supply:
                    qtity_total_s -= microgrid_seller.liste_sellers[k].supply
                    qtity_per_seller = qtity_total_s/(nb_sellers-k)
                    microgrid_seller.liste_sellers[k].supply = 0
                    microgrid_seller.liste_sellers[k].payoff += microgrid_seller.avg_price * microgrid_seller.liste_sellers[k].supply

                else:
                    qtity_total_s -= qtity_per_seller
                    microgrid_seller.liste_sellers[k].supply -= qtity_per_seller
                    microgrid_seller.liste_sellers[k].payoff += microgrid_seller.avg_price * qtity_per_seller
            
            if microgrid_buyer.microgrid_energy == 0:
                self.liste_microgrids_buyers.remove(microgrid_buyer)
            if microgrid_seller.microgrid_energy == 0:
                self.liste_microgrids_sellers.remove(microgrid_seller)   

        #fin du while
        

        #------------ mettre à jour avec les valeurs du nouveau state 
        if self.current_timestep < 24:
            for m in range(len(self.liste_microgrids)):

                self.liste_microgrids[m].liste_buyers = []
                self.liste_microgrids[m].liste_sellers = []

                for a in range(self.liste_microgrids[m].nb_agents):

                    self.liste_microgrids[m].agents[a].demand = self.liste_microgrids[m].L_conso[a][self.current_timestep]
                    self.liste_microgrids[m].agents[a].supply = max(0,self.liste_microgrids[m].L_prod[a][self.current_timestep] - self.liste_microgrids[m].agents[a].demand)
                    
                    if self.liste_microgrids[m].agents[a].supply > self.liste_microgrids[m].agents[a].demand and self.liste_microgrids[m].agents[a].supply > 0.0:
                        self.liste_microgrids[m].agents[a].statu = 'seller'
                        self.liste_microgrids[m].liste_sellers.append(self.liste_microgrids[m].agents[a])

                    elif self.liste_microgrids[m].agents[a].supply < self.liste_microgrids[m].agents[a].demand and self.liste_microgrids[m].agents[a].demand > 0.0:
                        self.liste_microgrids[m].agents[a].statu = 'buyer'
                        self.liste_microgrids[m].liste_buyers.append(self.liste_microgrids[m].agents[a])

                    else:
                        self.liste_microgrids[m].agents[a].statu = 'observator'

                self.liste_microgrids[m].Demand_total = sum(buyer.demand for buyer in self.liste_microgrids[m].liste_buyers)
                self.liste_microgrids[m].Supply_total = sum(seller.supply for seller in self.liste_microgrids[m].liste_sellers) 

        rewards = {(m,i): self.get_reward(m,i) for m in range(self.nb_microgrids) for i in range(self.liste_microgrids[m].nb_agents)}
        observations = {(m,i): self.get_observation(m, i) for m in range(self.nb_microgrids) for i in range(self.liste_microgrids[m].nb_agents)}
        done = {(m,i): self.is_done(m,i) for m in range(self.nb_microgrids) for i in range(self.liste_microgrids[m].nb_agents)}
        done["__all__"] = all(done.values())
        


        return observations, rewards, done, self.info

    def render(self):
        pass


if __name__ == '__main__':

    #Création de 3 microgrids
    #MICRIGRID 1
    # Set up the microgrid instance with test configuration
    consomations_H1 = np.array([0.31, 0.292, 0.308, 0.286, 0.29,  0.285, 0.84,  1.82,  2.22,  2.18,  1.9, 1.93, 1.647, 0.998, 1.173, 0.982, 1.262, 1.491, 1.396, 3.311, 2.332, 2.664, 2.212, 0.402])
    consomations_H2 = np.array([0.283, 0.187, 0.328, 0.265, 0.265, 0.339, 0.681, 3.343, 1.656, 2.57 , 1.867, 1.924, 1.386, 0.847, 1.177, 1.032, 1.009, 1.625, 1.393, 2.533, 2.307, 1.932, 2.665, 0.253])
    consomations_H3 = np.array([0.185, 0.189, 0.363, 0.242, 0.205, 0.275, 0.55 , 2.986, 2.192, 0.647, 1.826, 1.959, 2.783, 0.635, 1.111, 1.317, 1.224, 1.524, 1.502, 3.494, 3.550, 2.773, 3.694, 0.202])
    consomations_H4 = np.array([0.253, 0.204, 0.384, 0.274, 0.318, 0.339, 0.476, 1.754, 1.466, 3.052, 1.86 , 1.929, 1.423, 0.989, 1.186, 0.649, 1.005, 1.482, 1.483, 4.395, 3.034, 2.67 , 1.23 , 0.798])
    consomations_H5 = np.array([0.108, 0.246, 0.404, 0.205, 0.235, 0.237, 0.615, 2.971, 2.259, 1.716, 2.464, 1.890, 1.941, 1.530, 1.096, 1.305, 1.266, 1.479, 1.387, 3.556, 3.079, 2.725, 3.652, 0.218])
    consomations_H6 = np.array([0.236, 0.071, 0.356, 0.297, 0.306, 0.360, 0.710, 3.365, 1.644, 2.587, 1.881, 1.849, 1.283, 0.777, 1.116, 0.978, 0.953, 1.601, 1.386, 2.500, 2.162, 1.908, 2.611, 0.118])

    L_conso = [consomations_H1, consomations_H2, consomations_H3, consomations_H4, consomations_H5, consomations_H6]

    produced_solar_H1 = np.array([0]*24)
    produced_solar_H2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 0.23, 1.037, 1.80, 2.32, 2.56, 2.56, 2.18, 1.9, 1.3, 0.43, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    produced_solar_H3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.024, 0.33, 1.12, 1.92, 2.36, 2.66, 2.7, 2.5, 1.97, 1.5, 0.51, 0.132, 0.00, 0.0, 0.0, 0.0, 0.0 ])
    produced_solar_H4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.04, 0.40, 1.054, 1.88, 2.43, 2.67, 2.85, 2.65, 1.97, 1.5, 0.66, 0.067, 0.008, 0.0, 0.0, 0.0, 0.0] )
    produced_solar_H5 = np.array([0]*24)
    produced_solar_H6 =  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034, 0.38, 2.0, 2.92, 2.56, 2.76, 3.5, 3.2, 2.97, 2.5, 0.91, 0.2, 0.10, 0.0, 0.0, 0.0, 0.0 ])


    L_solar = [produced_solar_H1, produced_solar_H2, produced_solar_H3, produced_solar_H4, produced_solar_H5, produced_solar_H6]

    produced_eolien_H1 = np.array([0]*24)
    produced_eolien_H2 = np.array([0.247, 0.242, 0.217, 0.199, 0.18 , 0.159, 0.156, 0.152, 0.134, 0.095, 0.07 , 0.073, 0.065, 0.058, 0.056, 0.057, 0.056, 0.057, 0.067, 0.077, 0.102, 0.126, 0.135, 0.153])*2
    produced_eolien_H3 = np.array([0.238, 0.233, 0.23 , 0.198, 0.182, 0.164, 0.152, 0.15 , 0.138, 0.111, 0.088, 0.071, 0.065, 0.056, 0.056, 0.055, 0.054, 0.056, 0.061, 0.077, 0.1  , 0.132, 0.136, 0.149])*2
    produced_eolien_H4 = np.array([0.259, 0.252, 0.23 , 0.215, 0.19 , 0.17 , 0.164, 0.159, 0.146, 0.119, 0.083, 0.079, 0.069, 0.065, 0.057, 0.059, 0.059, 0.06 , 0.07 , 0.084, 0.098, 0.131, 0.14 , 0.155])*2
    produced_eolien_H5 = np.array([0]*24)
    produced_eolien_H6 = np.array([0.28, 0.27, 0.27 , 0.238, 0.22, 0.194, 0.182, 0.17 , 0.17, 0.15, 0.13, 0.10, 0.085, 0.066, 0.06, 0.06, 0.055, 0.0656, 0.0761, 0.087, 0.13  , 0.152, 0.156, 0.16])*2


    L_eolien = [produced_eolien_H1, produced_eolien_H2, produced_eolien_H3, produced_eolien_H4, produced_eolien_H5, produced_eolien_H6]
    price_sp_rachat = 0.06
    price_sp =  0.28
    step = 0.01

    liste_prix = np.linspace(price_sp_rachat, price_sp, num = int((price_sp - price_sp_rachat)/step) +1 )
    demand_total_prec = 3.54
    supply_total_prec = 4.2
    avg_price_old = (price_sp_rachat + price_sp)/2.


    microgrid_config1 = {
            "n": len(L_eolien),
            "liste_prix" : liste_prix,
            "demand_total_prec" : demand_total_prec,
            "supply_total_prec": supply_total_prec,
            "avg_price_old": avg_price_old,
            "L_conso" : L_conso,
            "L_solar": L_solar,
            "L_eolien": L_eolien
        }
    m1 = Microgrid(microgrid_config1)


    microgrid_config_2 = {
            "n": len(L_eolien),
            "liste_prix" : liste_prix,
            "demand_total_prec" : demand_total_prec,
            "supply_total_prec": supply_total_prec,
            "avg_price_old": avg_price_old,
            "L_conso" : L_conso,
            "L_solar": L_solar,
            "L_eolien": L_eolien
    }
    m2 = Microgrid(microgrid_config2)


    microgrid_config_3 = {
            "n": len(L_eolien),
            "liste_prix" : liste_prix,
            "demand_total_prec" : demand_total_prec,
            "supply_total_prec": supply_total_prec,
            "avg_price_old": avg_price_old,
            "L_conso" : L_conso,
            "L_solar": L_solar,
            "L_eolien": L_eolien
    }
    m3 = Microgrid(microgrid_config3)

    # Create instances of Microgridmulti:
    Liste_microgrids = [m1, m2, m3]     #liste of microgrid
    multi_config = {
        "L_microgrids": Liste_microgrids
    }
    Multi_microgrid = Microgridmulti(config_1)


