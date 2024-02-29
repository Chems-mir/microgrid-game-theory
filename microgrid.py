import Joueur as J
import Trx as T
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box
import numpy as np
import copy
import random


class Microgrid():
    def __init__(self, config):
        self.nb_agents = config['n']
        self.L_conso = config['L_conso']
        self.L_prod =  [x + y for x, y in zip(config['L_solar'], config['L_eolien'])]
        assert(self.nb_agents == len(self.L_conso) and self.nb_agents == len(self.L_prod))
        
        self.eval = False
        self.agents = {} #contient les joueurs en tant qu'objet
        J.cpt =0
        #J(demand, produced, localisation, action):
        for i in range(self.nb_agents):
            h = J.Joueur(self.L_conso[i][0], self.L_prod[i][0], [0,1])
            self.agents[i] = h

        self._agents_ids = set(self.agents.keys())
        self.liste_prix = config['liste_prix']
        self.demand_total_init = config['demand_total_prec']
        self.supply_total_init = config['supply_total_prec']
        self.avg_price_init = config['avg_price_old']
        self.Demand_total_old = config['demand_total_prec']
        self.Supply_total_old = config['supply_total_prec']
        self.avg_price_old = config['avg_price_old']

        self.liste_buyers = []
        self.liste_sellers = []

        for i in self._agents_ids:
            if self.agents[i].demand > 0.0 and self.agents[i].demand >self.agents[i].supply:
                self.agents[i].statu = 'buyer'
                self.liste_buyers.append(self.agents[i])
            elif self.agents[i].supply > 0.0 and self.agents[i].supply >self.agents[i].demand:
                self.agents[i].statu = 'seller'
                self.liste_sellers.append(self.agents[i])

        self.Demand_total = sum(buyer.demand for buyer in self.liste_buyers)
        self.Supply_total = sum(seller.supply for seller in self.liste_sellers)
        self.avg_price = 0
        self.microgrid_statu = 'ok' #can be 'ok', 'shortage', 'surplus'
        self.microgrid_energy = 0 #contains the need or surplus of energy for the microgrid after the trx within its members.

        self.action_space = Discrete(len(self.liste_prix))
        self.observation_space = Box(low=np.array([-1, 0 ,0, 0, 0, 0, 0], dtype=np.float32), high=np.array([1, 10, 100, 100, 100, 100, 1], dtype=np.float32), shape=(7,), dtype=np.float32)
        self.info = {i: {} for i in self._agents_ids}



    def get_observation(self, agent_id):
        if self.agents[agent_id].statu == 'seller':
            return np.array([1, self.agents[agent_id].supply , self.Demand_total_old, self.Demand_total, self.Supply_total_old,  self.Supply_total, self.avg_price_old], dtype=np.float32)
        elif self.agents[agent_id].statu == 'buyer':
            return np.array([-1, self.agents[agent_id].demand, self.Demand_total_old,  self.Demand_total, self.Supply_total_old,  self.Supply_total, self.avg_price_old], dtype=np.float32)
        else:
            return np.array([0, 0, self.Demand_total_old, self.Demand_total, self.Supply_total_old, self.Supply_total, self.avg_price_old], dtype=np.float32)

    def get_reward(self, agent_id):
        return self.agents[agent_id].payoff

    def is_done(self, agent_id):
        if self.current_timestep >= self.max_timesteps :
            return True
        else:
            return False

    def reset(self):

        for i in range(len(self.L_conso)):
            if self.eval == False:
                self.L_conso[i] = randomize_data(self.L_conso[i])
                self.L_prod[i] =  randomize_data(self.L_prod[i])
                if any(x > 6 for x in  self.L_conso[i]):
                    print('erreur 900 ', self.L_conso[i])
                if any(x > 6 for x in  self.L_prod[i]):
                    print('erreur 900 ', self.L_prod[i])
            else:
                self.L_conso[i] = randomize_data(self.L_conso[i], eval = True)
                self.L_prod[i] =  randomize_data(self.L_prod[i], eval = True)

        self.agents = {} #contient les joueurs en tant qu'objet

        J.Joueur.cpt =0
        #J(demand, produced, localisation, action):
        for i in range(self.nb_agents):
            h = J.Joueur(self.L_conso[i][0], self.L_prod[i][0], [0,1])
            self.agents[i] = h

        self.nb_agents = len(self.agents)
        self._agents_ids = set(self.agents.keys())
        self.Demand_total_old = self.demand_total_init
        self.Supply_total_old = self.supply_total_init
        self.avg_price_old = self.avg_price_init
        self.liste_buyers = []
        self.liste_sellers = []

        for i in self._agents_ids:
          #  print(self.agents[i])
            if self.agents[i].demand > 0.0 and self.agents[i].demand >self.agents[i].supply:
                self.agents[i].statu = 'buyer'
                self.liste_buyers.append(self.agents[i])
            elif self.agents[i].supply > 0.0 and self.agents[i].supply >self.agents[i].demand:
                self.agents[i].statu = 'seller'
                self.liste_sellers.append(self.agents[i])

        self.Demand_total = sum(buyer.demand for buyer in self.liste_buyers)
        self.Supply_total = sum(seller.supply for seller in self.liste_sellers)
        self.avg_price = 0

        return {i: self.get_observation(i) for i in self._agents_ids}

    def get_weighted_moy(self):
        average_buyers = 0
        average_sellers = 0
        #print("demand total, supply total ",self.Demand_total, self.Supply_total)

        if (self.Demand_total >0 and  self.Supply_total>0):
            lambda_b = 1 - (self.Demand_total/(self.Demand_total + self.Supply_total))
            lambda_s = 1 - (self.Supply_total/(self.Demand_total + self.Supply_total))
        elif (self.Demand_total >0 and  self.Supply_total == 0):
            lambda_b = 1
            lambda_s = 0
        elif (self.Demand_total == 0 and  self.Supply_total >0):
            lambda_b = 0
            lambda_s = 1
        else:
            lambda_b = 0
            lambda_s = 0

        if self.Demand_total >0:
            for i in range(len(self.liste_buyers)):
                if self.liste_buyers[i].statu == "buyer":
                    average_buyers += (self.liste_buyers[i].demand/self.Demand_total)* self.liste_buyers[i].price

        if self.Supply_total >0:
            for i in range(len(self.liste_sellers)):
                if self.liste_sellers[i].statu == "seller":
                    average_sellers += (self.liste_sellers[i].supply/self.Supply_total)* self.liste_sellers[i].price

        # print('\n lambda_b', lambda_b)
        # print('lambda_s', lambda_s)
        # print("self.Supply_total", self.Supply_total)
        # print("self.Demand_total", self.Demand_total)
        # print("average_sellers", average_sellers)
        # print('average_buyers ', average_buyers)

        moy = lambda_b*average_buyers + lambda_s*average_sellers
        # print("moy", moy)
        # print('\n')
        return moy


    def max_qtity_trx(self, buyer, seller, S, D, nb_buyers, nb_sellers):
        Qi = buyer.demand
        Qj = seller.supply
        if D >= S:
            alpha =S /(nb_buyers)
        else:
            alpha = D /(nb_sellers)

        if min(Qi, Qj) <= alpha :
            Qij = min(Qi, Qj)
        else:
            Qij = alpha
        return Qij

    def penalization(self, x, m, eps):
        if (abs(x - m) > eps*m) and (x > m):
            return abs(x - m*(1+eps))
        elif (abs(x - m) > eps*m) and (x < m):
            return abs(x - m*(1-eps))
        else:
            return 0


    def find_closest_to_target(liste, target):
        if not liste:
            return None
        else:
            return min(liste, key=lambda obj: abs(obj.price - target))


    def find_closest_pair(self, team1, team2):
        team1.sort(key=lambda obj: obj.price) 
        team2.sort(key=lambda obj: obj.price) 

        i, j = 0, 0  # Pointers for team1 and team2
        min_distance = float('inf')
        closest_pair = None

        while i < len(team1) and j < len(team2):
            distance = abs(team1[i].price - team2[j].price)
            if distance < min_distance:
                min_distance = distance
                closest_pair = [team1[i], team2[j]]

            # Adjust pointers strategically. If team1[i]'s price is smaller, increment i. Otherwise, increment j 
            if team1[i].price < team2[j].price:
                i += 1
            else:
                j += 1

        return closest_pair


    def tournoi(self, m_global, choix = 'var1'):
        #retourne toute les transactions; 1 transaction =  (id_acheteur, id_vendeur, quantity, price)

        liste_trx = []
        S = self.Supply_total
        D = self.Demand_total
        nb_buyers = len(self.liste_buyers)
        nb_sellers = len(self.liste_sellers)

        if choix == 'var1':
            team_b1 = copy.copy(self.liste_buyers)
            team_s1 = copy.copy(self.liste_sellers)

            while(len(self.liste_buyers) > 0 and len(self.liste_sellers)>0):
                if len(team_b1) == 0:
                    team_b1 = copy.copy(self.liste_buyers)
                if len(team_s1) == 0:
                    team_s1 = copy.copy(self.liste_sellers)

                while(len(team_b1) > 0 and len(team_s1)>0):
                    
                    #print team b1 et s1 et liste sellers buyers pour les comparer 
                    #et s'assurer que la mise à jour de chaque joueur est correcte
                    print("-----")
                    print("-----")
                    print("team b1:")
                    for i in team_b1:
                        print(i)
                    print("-----")

                    print("liste_buyers:")
                    for i in self.liste_buyers:
                        print(i)
                    print("-----")

                    print('team s1')
                    for j in team_s1:
                        print(j)
                    print("-----")
                    print("liste_sellers")
                    for j in self.liste_sellers:
                        print(j)
                    print("-----")



                    player_s = random.choice(team_s1)
                    player_b = random.choice(team_b1)

                    if abs(player_s.price - m_global) < abs(player_b.price - m_global):
                        trx = T.Trx(player_b, player_s, self.max_qtity_trx(player_b, player_s, self.Supply_total, self.Demand_total, nb_buyers, nb_sellers), player_s.price)
                        liste_trx.append(trx)
                        #print(trx)
                        player_b.demand -= trx.quantity
                        player_s.supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity
                        team_b1.remove(player_b)
                        team_s1.remove(player_s)
                        if player_b.demand == 0:
                            self.liste_buyers.remove(player_b)
                            ##print('b delete')
                        if player_s.supply == 0:
                            self.liste_sellers.remove(player_s)
                            ##print("s delete")

                    elif abs(player_s.price - m_global) > abs(player_b.price - m_global):

                        trx = T.Trx(player_b, player_s, self.max_qtity_trx(player_b, player_s, self.Supply_total,  self.Demand_total, nb_buyers, nb_sellers), player_b.price)
                        liste_trx.append(trx)
                        #print(trx)
                        player_b.demand -= trx.quantity
                        player_s.supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity
                        team_b1.remove(player_b)
                        team_s1.remove(player_s)
                        if player_b.demand == 0:
                            self.liste_buyers.remove(player_b)
                        if player_s.supply == 0:
                            self.liste_sellers.remove(player_s)
                    else :

                        trx = T.Trx(player_b, player_s, self.max_qtity_trx(player_b, player_s, self.Supply_total,  self.Demand_total, nb_buyers, nb_sellers), m_global)
                        liste_trx.append(trx)
                        #print(trx)
                        player_b.demand -= trx.quantity
                        player_s.supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity
                        team_b1.remove(player_b)
                        team_s1.remove(player_s)
                        if player_b.demand == 0:
                            self.liste_buyers.remove(player_b)
                        if player_s.supply == 0:
                            self.liste_sellers.remove(player_s)

                #end while 1

            #end while2

            self.Demand_total_old = self.Demand_total
            self.Supply_total_old = self.Supply_total

            self.Demand_total = D
            self.Supply_total = S

            return liste_trx

        elif choix == 'var2':
            while((len(self.liste_buyers) >0) and (len(self.liste_sellers) > 0)):

                winner_buyer = self.find_closest_to_target(self.liste_buyers, m_global)
                winner_seller = self.find_closest_to_target(self.liste_sellers, m_global)

                if abs(winner_seller.price - m_global) < abs(winner_buyer.price - m_global):
                    #winner = seller
                    trx = T.Trx(winner_buyer, winner_seller, max(winner_buyer.demand, winner_seller.supply), winner_seller.price)
                    liste_trx.append(trx)
                    winner_buyer.demand -= trx.quantity
                    winner_seller.supply -= trx.quantity
                    D -= trx.quantity
                    S -= trx.quantity
                    if winner_buyer.demand == 0:
                        self.liste_buyers.remove(winner_buyer)
                    if winner_seller.supply == 0:
                        self.liste_sellers.remove(winner_seller)

                elif abs(winner_seller.price - m_global) > abs(winner_buyer.price - m_global):
                    # winner = player_b
                    trx = T.Trx(winner_buyer, winner_seller, max(winner_buyer.demand, winner_seller.supply), winner_buyer.price)
                    liste_trx.append(trx)
                    winner_buyer.demand -= trx.quantity
                    winner_seller.supply -= trx.quantity
                    D -= trx.quantity
                    S -= trx.quantity
                    if winner_buyer.demand == 0:
                        self.liste_buyers.remove(winner_buyer)
                    if winner_seller.supply == 0:
                        self.liste_sellers.remove(winner_seller)
                else:
                    trx = T.Trx(winner_buyer, winner_seller, max(winner_buyer.demand, winner_seller.supply), m_global)
                    liste_trx.append(trx)
                    winner_buyer.demand -= trx.quantity
                    winner_seller.supply -= trx.quantity
                    D -= trx.quantity
                    S -= trx.quantity
                    if winner_buyer.demand == 0:
                        self.liste_buyers.remove(winner_buyer)
                    if winner_seller.supply == 0:
                        self.liste_sellers.remove(winner_seller)


            self.Demand_total_end = D
            self.Supply_total_end = S

            return liste_trx

        elif choix == 'var3':
            while(D > 0 and S > 0 ):

                listes_couples =[]
                team_s = copy.copy(self.liste_sellers)
                team_b = copy.copy(self.liste_buyers)
                while(len(team_s)> 0 and len(team_b)>0):
                    results = self.find_closest_pair(team_b, team_s)
                    listes_couples.append(results)

                    if abs(results[1].price - m_global) < abs(results[0].price - m_global):
                        #winner = seller
                        trx = T.Trx(results[0], results[1], max(results[0].demand, results[1].supply), results[1].price)
                        liste_trx.append(trx)
                        results[0].demand -= trx.quantity
                        results[1].supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity

                    elif abs(results[1].price - m_global) > abs(results[0].price - m_global):
                        # winner = player_b
                        trx = T.Trx(results[0], results[1], max(results[0].demand, results[1].supply), results[0].price)
                        liste_trx.append(trx)
                        results[0].demand -= trx.quantity
                        results[1].supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity

                    else:
                        trx = T.Trx(results[0], results[1], max(results[0].demand, results[1].supply), m_global)
                        liste_trx.append(trx)
                        results[0].demand -= trx.quantity
                        results[1].supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity

                    team_b.remove(results[0])
                    team_s.remove(results[1])

            self.Demand_total_end = D
            self.Supply_total_end = S

            return liste_trx

        else:
            return liste_trx


    def payoffs(self, moy, choice=None): #attention ici en argument ce sont des indices car les actions seront des element discret representant l'indice du prix choisi ds la liste strategies
        L_payoffs = [0]*self.nb_agents

        if choice is not None:
            transactions = self.tournoi(moy, choice)
        else:
            transactions = self.tournoi(moy)
    # [print(t)  for t in transactions]

        #update statu of microgrid
        self.microgrid_energy = self.Supply_total - self.Demand_total 
        if self.microgrid_energy > 0:
            self.microgrid_statu = 'surplus'
        elif self.microgrid_energy < 0:
            self.microgrid_statu = 'shortage'
        else:
            self.microgrid_statu = 'ok'


        # calcul le payoff u_i(x_i, x__i) du joueur i, avec l'action joué x=(x_1, ...., xn)
        for t in transactions:
            pen_seller = self.penalization(self.agents[t.seller.id].price, moy, 0.12)
            print('pen seller', pen_seller)
            L_payoffs[t.seller.id] += t.price * t.quantity - pen_seller  # - cost(t.quantity/2.)

            pen_buyer = self.penalization(self.agents[t.buyer.id].price, moy, 0.12)
            print('pen buyer', pen_buyer)
            L_payoffs[t.buyer.id] += -1*(t.price * t.quantity) - pen_buyer  # - cost(t.quantity/2.)

        return L_payoffs


if __name__ == '__main__':

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


    env_config = {
            "n": len(L_eolien),
            "liste_prix" : liste_prix,
            "demand_total_prec" : demand_total_prec,
            "supply_total_prec": supply_total_prec,
            "avg_price_old": avg_price_old,
            "L_conso" : L_conso,
            "L_solar": L_solar,
            "L_eolien": L_eolien
        }


    env = Microgrid(env_config)
    for identifiant, agent in env.agents.items():
        agent.price = random.choice(liste_prix)

    moyenne = env.get_weighted_moy()
    payoffs = env.payoffs(moyenne)
    #liste_trx = env.tournoi(moyenne)
    print(f'moyenne:{moyenne}')

    for k in payoffs:
        print(k)
