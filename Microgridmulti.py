import Microgrid as M

class Microgridmulti(MultiAgentEnv):
    def __init__(self, config):
        
        self.eval = False
        self.liste_microgrids = config['L_microgrids']
        self.nb_microgrids = len(self.liste_microgrids)
        self.liste_microgrids_init = copy.deepcopy(self.liste_microgrids)
        self.current_timestep = 0
        self.max_timesteps = 23
        
        self.liste_microgrids_buyers = []
        self.liste_microgrids_sellers = []

        #for one agent:
        self.action_space = Discrete(len(self.liste_prix)) #correspond à l'indice dans la liste de prix possible
        self.observation_space = Box(low=np.array([-1, 0 ,0, 0, 0, 0, 0], dtype=np.float32), high=np.array([1, 10, 100, 100, 100, 100, 1], dtype=np.float32), shape=(7,), dtype=np.float32)
        
        self.info = {(i,j): {} for i in range(self.nb_microgrids) for j in range(len(self.liste_microgrids[i].nb_agents))} #i correspond au num du microgrid, et j au numero de house
    

    def get_observation(self, microgrid_id, agent_id):
        if self.liste_microgrids[microgrid_id].agents[agent_id].statu == 'seller':
            return np.array([1, self.liste_microgrids[microgrid_id].agents[agent_id].supply , self.liste_microgrids[microgrid_id].Demand_total_old, self.liste_microgrids[microgrid_id].Demand_total, self.liste_microgrids[microgrid_id].Supply_total_old,  self.liste_microgrids[microgrid_id].Supply_total, self.liste_microgrids[microgrid_id].avg_price_old], dtype=np.float32)
        elif self.agents[agent_id].statu == 'buyer':
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

        if not self.liste_microgrids_buyers or not self.liste_microgrids_sellers:
            return None  
        # Sort the lists in ascending order based on the 'number' attribute
        self.liste_microgrids_buyers.sort(key=lambda x: x.price)
        self.liste_microgrids_sellers.sort(key=lambda x: x.price)

        closest_pair = None
        min_distance = float('inf')

        # Iterate through elements in liste_microgrids_buyers, comparing with a limited window in liste_microgrids_sellers
        for i, x in enumerate(self.liste_microgrids_buyers):
            # Define a search window in liste_microgrids_sellers based on the current element in liste_microgrids_buyers
            low = max(0, i - len(self.liste_microgrids_sellers) // 2)  # Lower bound (avoid negative index)
            high = min(len(self.liste_microgrids_sellers), i + len(self.liste_microgrids_sellers) // 2 + 1)  # Upper bound (avoid exceeding list length)

            for j in range(low, high):
                y = self.liste_microgrids_sellers[j]
                distance = abs(x.price - y.price)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (x, y)

            # Early termination if a pair with distance 0 is found
            if min_distance == 0:
                break

        return closest_pair


    def step(self, action_dict):

        Liste_payoffs = [[None] * self.liste_microgrids[m].nb_agents for m in range(self.nb_microgrids)]
        # Execute action
        for m in range(len(self.liste_microgrids)):
            for i in range(self.liste_microgrids[m].nb_agents):
                self.liste_microgrids[m].agents[i].price = self.liste_prix[action_dict[(m,i)]]
        
            self.liste_microgrids[m].Demand_total_old = self.liste_microgrids[m].Demand_total
            self.liste_microgrids[m].Supply_total_old = self.liste_microgrids[m].Supply_total

            #determination des coeff lambda and avg
            self.liste_microgrids[m].avg_price = self.liste_microgrids[m].get_weighted_moy()
            for i in self.liste_microgrids[m]._agents_ids:
                self.liste_microgrids[m].info[(m,i)] = {"avg_price": self.liste_microgrids[m].avg_price, "agent_price": self.liste_microgrids[m].agents[i].price}

            #calcul des payoffs
            Liste_payoffs[m] = self.liste_microgrids[m].payoffs(self.liste_microgrids[m].avg_price)
            for id in self.liste_microgrids[m]._agents_ids:
                self.liste_microgrids[m].agents[id].payoff += Liste_payoffs[m][id]

         # Update players listes to reflect the new state after transactions
         # mettre a jour la nouvelle demand et supply du nouveau timestep, les listes de players; buyers, seller
         # et demand total supply total precedent

            self.liste_microgrids[m].avg_price_old = self.liste_microgrids[m].avg_price

            for a in self.liste_microgrids[m]._agents_ids:
                self.liste_microgrids[m].agents[a].demand = self.liste_microgrids[m].L_conso[a][self.current_timestep]
                self.liste_microgrids[m].agents[a].supply = max(0,self.liste_microgrids[m].L_prod[a][self.current_timestep] - self.liste_microgrids[m].agents[a].demand)
                if self.liste_microgrids[m].agents[a].supply > self.liste_microgrids[m].agents[a].demand:
                    self.liste_microgrids[m].agents[a].statu = 'seller'
                elif self.liste_microgrids[m].agents[a].supply < self.liste_microgrids[m].agents[a].demand:
                    self.liste_microgrids[m].agents[a].statu = 'buyer'
                else:
                    self.liste_microgrids[m].agents[a].statu = 'observator'


            self.liste_microgrids[m].liste_buyers = []
            self.liste_microgrids[m].liste_sellers = []

            for i in self.liste_microgrids[m]._agents_ids:
                #print(i)
                #print(self.agents[i])
                if self.liste_microgrids[m].agents[i].demand > 0.0 and self.liste_microgrids[m].agents[i].demand >self.liste_microgrids[m].agents[i].supply:
                    self.liste_microgrids[m].agents[i].statu = 'buyer'
                    self.liste_microgrids[m].liste_buyers.append(self.liste_microgrids[m].agents[i])
                elif self.liste_microgrids[m].agents[i].supply > 0.0 and self.liste_microgrids[m].agents[i].supply >self.liste_microgrids[m].agents[i].demand:
                    self.liste_microgrids[m].agents[i].statu = 'seller'
                    self.liste_microgrids[m].liste_sellers.append(self.liste_microgrids[m].agents[i])

            #self.liste_microgrids[m].Demand_total = sum(buyer.demand for buyer in self.liste_microgrids[m].liste_buyers)
            #self.liste_microgrids[m].Supply_total = sum(seller.supply for seller in self.liste_microgrids[m].liste_sellers) 

            #update statu of microgrid
            self.liste_microgrids[m].microgrid_energy = self.liste_microgrids[m].Supply_total - self.liste_microgrids[m].Demand_total 
            
            if self.liste_microgrids[m].microgrid_energy > 0:
                self.liste_microgrids[m].microgrid_statu = 'surplus'
            elif self.liste_microgrids[m].microgrid_energy < 0:
                self.liste_microgrids[m].microgrid_statu = 'shortage'
            else:
                self.liste_microgrids[m].microgrid_statu = 'ok'


        for i in self.liste_microgrids:
            if i.microgrid_statu == 'shortage':
                self.liste_microgrids_buyers.append(i)
            elif i.microgrid_statu == 'surplus':
                self.liste_microgrids_sellers.append(i)
            else:
                pass


        while(len(self.liste_microgrids_sellers) > 0 and len(self.liste_microgrids_buyers) > 0):
            microgrid_buyer, microgrid_seller = self.closest_pair_microgrids()
            qtity = min(abs(microgrid_buyer.microgrid_energy), microgrid_seller.microgrid_energy)

            nb_buyers = len(microgrid_buyer.liste_buyers)
            nb_sellers = len(microgrid_seller.liste_sellers)
            
            for b in microgrid_buyer.liste_buyers:
                b.demand -= qtity/nb_buyers

            for s in microgrid_seller.liste_sellers:
                s.supply -= qtity/nb_sellers

            

            microgrid_buyer.microgrid_energy += qtity
            microgrid_seller.microgrid_energy -= qtity
            if microgrid_buyer.microgrid_energy == 0:
                self.liste_microgrids_buyers.remove(microgrid_buyer)
            if microgrid_seller.microgrid_energy == 0:
                self.liste_microgrids_sellers.remove(microgrid_seller)     






#---------------------- completer les echanges

        rewards = {(m,i): self.get_reward(m,i) for m in range(self.nb_microgrids) for i in self.liste_microgrids[m]._agents_ids}
        observations = {(m,i): self.get_observation(m, i) for m in range(self.nb_microgrids) for i in self.liste_microgrids[m]._agents_ids}
        done = {(m,i): self.is_done(m,i) for m in range(self.nb_microgrids) for i in self.liste_microgrids[m]._agents_ids}
        done["__all__"] = all(done.values())
        self.current_timestep +=1


        return observations, rewards, done, self.info

    def render(self):
        pass


if __name__ == '__main__':

#Création de 3 microgrids

