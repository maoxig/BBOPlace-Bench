REGISTRY = {}

from .ea.vanilla_ea import VanillaEA
from .bo.bo import BO 
from .sa.sa import SA
from .ea.es import ES
from .ea.pso import PSO


from .mo.nsga2 import NSGAII
from .mo.nsga3 import NSGAIII
from .mo.moead import MOEADDE
from .mo.spea2 import SPEA2_Algo
from .mo.smsemoa import SMSEMOA_Algo


REGISTRY["ea"] = VanillaEA
REGISTRY["bo"] = BO  
REGISTRY["sa"] = SA
REGISTRY["es"] = ES
REGISTRY["pso"] = PSO


REGISTRY['nsga2'] = NSGAII
REGISTRY['nsga3'] = NSGAIII
REGISTRY['moead'] = MOEADDE
REGISTRY['spea2'] = SPEA2_Algo
REGISTRY['smsemoa'] = SMSEMOA_Algo