
from ray.rllib.algorithms.ppo import PPOConfig
import matplotlib.pyplot as plt
from ray.tune.logger import pretty_print


def randomize_data(data, scale_factor=0.05, eval=False):
    """
    Randomizes data by adding/subtracting a random value.
    :param data: Original numpy array to randomize.
    :param scale_factor: Percentage of mean to use for randomization range.
    :return: Randomized numpy array.
    """
    if eval == False:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        mean_value = np.mean(data)
        # Generate noise and apply to the data
        noise = np.random.uniform(-mean_value * scale_factor, mean_value * scale_factor, data.shape)
        #noise = np.random.normal(0, mean_value * scale_factor, data.shape)

        new_data = data + noise
        # Ensure values are not negative and not too big
        new_data = np.clip(new_data, 0, data + mean_value * scale_factor)

        #new_data = np.clip(new_data, 0, None)
        return new_data
    else:
        return data





#Création de 3 microgrids
#MICROGRID 1
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


#MICROGRID2
L_eolien2 = []
L_conso2 = []
L_solar2 = []
for i in range(len(L_eolien)): 
    L_eolien2.append(randomize_data(L_eolien[i], scale_factor=0.05))
    L_solar2.append(randomize_data(L_solar[i], scale_factor=0.05))
    L_conso2.append(randomize_data(L_conso[i], scale_factor=0.05))
microgrid_config_2 = {
        "n": len(L_eolien2),
        "liste_prix" : liste_prix,
        "demand_total_prec" : demand_total_prec,
        "supply_total_prec": supply_total_prec,
        "avg_price_old": avg_price_old,
        "L_conso" : L_conso2,
        "L_solar": L_solar2,
        "L_eolien": L_eolien2
}
m2 = Microgrid(microgrid_config2)


#MICROGRID3
L_eolien3 = []
L_conso3 = []
L_solar3 = []
for i in range(len(L_eolien)): 
    L_eolien3.append(randomize_data(L_eolien[i], scale_factor=0.05))
    L_solar3.append(randomize_data(L_solar[i], scale_factor=0.05))
    L_conso3.append(randomize_data(L_conso[i], scale_factor=0.05))
microgrid_config_3 = {
        "n": len(L_eolien3),
        "liste_prix" : liste_prix,
        "demand_total_prec" : demand_total_prec,
        "supply_total_prec": supply_total_prec,
        "avg_price_old": avg_price_old,
        "L_conso" : L_conso3,
        "L_solar": L_solar3,
        "L_eolien": L_eolien3
}
m3 = Microgrid(microgrid_config3)



# Create instances of Microgridmulti:
Liste_microgrids = [m1, m2, m3]     #liste of microgrid
multi_config = {
    "L_microgrids": Liste_microgrids
}
Multi_microgrid = Microgridmulti(config_1)

print(L_eolien3)
print(L_solar3)
print(L_conso3)





