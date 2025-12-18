REGISTRY = {}

from .ea.vanilla_ea import VanillaEA
from .bo.bo import BO 
from .sa.sa import SA
from .ea.es import ES
from .ea.pso import PSO
from .mo.nsgaii import NSGAII

REGISTRY["ea"] = VanillaEA
REGISTRY["bo"] = BO
REGISTRY["sa"] = SA
REGISTRY["es"] = ES
REGISTRY["pso"] = PSO
REGISTRY['nsga2'] = NSGAII