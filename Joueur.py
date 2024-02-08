class Joueur():
    cpt =0
    def __init__(self, demand, produced, localisation):
        self.id = Joueur.cpt #attention ne pas creer de joueur qui ne sont pas dans le jeu sinon les ids ne seront pas continues
        Joueur.cpt += 1
        self.demand = demand
        self.produced = produced
        self.localisation = localisation

        if self.demand > self.produced:
            self.statu = 'buyer'
            self.supply = 0
            self.demand = self.demand - self.produced
        elif self.demand < self.produced:
            self.statu = 'seller'
            self.supply = self.produced - self.demand
            self.demand = 0
        else:
            self.statu = 'observator'
            self.demand = 0
            self.supply = 0

        self.demand_initial = self.demand #permet de savoir à la fin si toute la demande a ete rempli ou non
        self.supply_initial = self.supply
        self.price = None
        self.payoff = 0

    def __str__(self):
        return f"House id {self.id} localisation ({self.localisation[0]},{self.localisation[1]}); \n Demand :{self.demand}; \n Statu:{self.statu}; \n Supply:{self.supply}\n price proposed : {self.price}\n"

    def distance_houses(self, h2):
        dist = math.sqrt((self.localisation[0] - h2.localisation[0])**2 + (self.localisation[1] - h2.localisation[1])**2)
        return dist
