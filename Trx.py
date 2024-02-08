
class Trx():
    cpt = 0
    def __init__(self, acheteur, vendeur, quantity, price):
        self.num = Trx.cpt
        Trx.cpt +=1
        self.buyer = acheteur
        self.seller = vendeur
        self.quantity = quantity
        self.price = price


    def __str__(self):
        return (f'transaction num : {self.num}, \n buyer : {self.buyer.id}  \n seller : {self.seller.id} \n price per unit:{self.price} \n amount of trx : {self.quantity} \n Total price : {self.price*self.quantity} ')
