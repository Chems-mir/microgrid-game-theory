class microgrid():
    def __init__(self, config):
        self.nb_agents = config['n']
        self.L_conso = config['L_conso']
        self.L_prod =  [x + y for x, y in zip(config['L_solar'], config['L_eolien'])]
        assert(self.nb_agents == len(self.L_conso) and self.nb_agents == len(self.L_prod))
        
        self.eval = False
        self.agents = {} #contient les joueurs en tant qu'objet
        Joueur.cpt =0
        #J(demand, produced, localisation, action):
        for i in range(self.nb_agents):
            h = Joueur(self.L_conso[i][0], self.L_prod[i][0], [0,1])
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

        Joueur.cpt =0
        #J(demand, produced, localisation, action):
        for i in range(self.nb_agents):
            h = Joueur(self.L_conso[i][0], self.L_prod[i][0], [0,1])
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

    def tournoi_selection(self, liste, target):
        lst = copy.copy(liste)
        if len(lst) < 1:
            print("pas de players")
            return None

        while len(lst) != 1:
            qualified = []
            while(lst):
                if len(lst) >= 2:
                    fight1 = random.sample(lst, 2)
                    ## print("fight :", fight1)
                    if abs(fight1[0].price - target) < abs(fight1[1].price - target):
                        qualified.append(fight1[0])
                    else:
                        qualified.append(fight1[1])

                    lst = list(set(lst) - set(fight1))

                else:
                    qualified.append(lst[0])
                    lst = []

            lst = qualified

        return lst[0]

    def find_closest_pair(self, team1, team2):
        closest_pair = []
        min_distance = float('inf')
        for a in team1:
            for b in team2:
                distance = abs(a.price - b.price)  # Compute the absolute difference
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = [a, b]

        return closest_pair


    def tournoi(self, m_global, choix = 'var1'):
        #retourne toute les transactions; 1 transaction =  (id_acheteur, id_vendeur, quantity, price)

        liste_trx = []
        S = self.Supply_total
        D = self.Demand_total
        nb_buyers = len(self.liste_buyers)
        nb_sellers = len(self.liste_sellers)
        #m_global = self.get_weighted_moy()

        if choix == 'var1':
            while(len(self.liste_buyers) > 0 and len(self.liste_sellers)>0):
                team_b1 = copy.copy(self.liste_buyers)
                team_s1 = copy.copy(self.liste_sellers)

                while(len(team_b1) > 0 and len(team_s1)>0):

                    player_s = random.choice(team_s1)
                    player_b = random.choice(team_b1)

                    if abs(player_s.price - m_global) < abs(player_b.price - m_global):
                        trx = Trx(player_b, player_s, self.max_qtity_trx(player_b, player_s, self.Supply_total, self.Demand_total, nb_buyers, nb_sellers), player_s.price)
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

                        trx = Trx(player_b, player_s, self.max_qtity_trx(player_b, player_s, self.Supply_total,  self.Demand_total, nb_buyers, nb_sellers), player_b.price)
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

                        trx = Trx(player_b, player_s, self.max_qtity_trx(player_b, player_s, self.Supply_total,  self.Demand_total, nb_buyers, nb_sellers), m_global)
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
            while((len(self.liste_buyers) >0) and (len(team_s) > 0)):

                winner_buyer = self.tournoi_selection(self.liste_buyers, m_global)
                winner_seller = self.tournoi_selection(team_s, m_global)

                if abs(winner_seller.price - m_global) < abs(winner_buyer.price - m_global):
                    #winner = seller
                    trx = Trx(winner_buyer, winner_seller, max(winner_buyer.demand, winner_seller.supply), winner_seller.price)
                    liste_trx.append(trx)
                    winner_buyer.demand -= trx.quantity
                    winner_seller.supply -= trx.quantity
                    D -= trx.quantity
                    S -= trx.quantity
                    if winner_buyer.demand == 0:
                        self.liste_buyers.remove(winner_buyer)
                    if winner_seller.supply == 0:
                        team_s.remove(winner_seller)

                elif abs(winner_seller.price - m_global) > abs(winner_buyer.price - m_global):
                    # winner = player_b
                    trx = Trx(winner_buyer, winner_seller, max(winner_buyer.demand, winner_seller.supply), winner_buyer.price)
                    liste_trx.append(trx)
                    winner_buyer.demand -= trx.quantity
                    winner_seller.supply -= trx.quantity
                    D -= trx.quantity
                    S -= trx.quantity
                    if winner_buyer.demand == 0:
                        self.liste_buyers.remove(winner_buyer)
                    if winner_seller.supply == 0:
                        team_s.remove(winner_seller)
                else:
                    trx = Trx(winner_buyer, winner_seller, max(winner_buyer.demand, winner_seller.supply), m_global)
                    liste_trx.append(trx)
                    winner_buyer.demand -= trx.quantity
                    winner_seller.supply -= trx.quantity
                    D -= trx.quantity
                    S -= trx.quantity
                    if winner_buyer.demand == 0:
                        self.liste_buyers.remove(winner_buyer)
                    if winner_seller.supply == 0:
                        team_s.remove(winner_seller)


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
                        trx = Trx(results[0], results[1], max(results[0].demand, results[1].supply), results[1].price)
                        liste_trx.append(trx)
                        results[0].demand -= trx.quantity
                        results[1].supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity

                    elif abs(results[1].price - m_global) > abs(results[0].price - m_global):
                        # winner = player_b
                        trx = Trx(results[0], results[1], max(results[0].demand, results[1].supply), results[0].price)
                        liste_trx.append(trx)
                        results[0].demand -= trx.quantity
                        results[1].supply -= trx.quantity
                        D -= trx.quantity
                        S -= trx.quantity

                    else:
                        trx = Trx(results[0], results[1], max(results[0].demand, results[1].supply), m_global)
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

        # calcul le payoff u_i(x_i, x__i) du joueur i, avec l'action joué x=(x_1, ...., xn)
        for t in transactions:
            pen_seller = self.penalization(self.agents[t.seller.id].price, moy, 0.1)
            L_payoffs[t.seller.id] += t.price * t.quantity - pen_seller  # - cost(t.quantity/2.)

            pen_buyer = self.penalization(self.agents[t.buyer.id].price, moy, 0.1)
            L_payoffs[t.buyer.id] += -1*(t.price * t.quantity) - pen_buyer  # - cost(t.quantity/2.)

        return L_payoffs
