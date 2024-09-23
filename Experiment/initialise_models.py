import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import pickle
from cgqemcmc.basic_utils import *

from cgqemcmc.Model_Maker import Model_Maker

#from cgqemcmc.Model_Maker import Model_Maker



# Basic helper code to initialise a list Ising models of type required by cgqemcmc
# Once created, Models are pickled so they can be easily accessed later.
# 

n_spins = 9

gamma = (0.25,0.6)
time = (2,20)


reps = 1
temp = 1

dir = os.path.dirname(os.path.abspath(__file__))
model_dir = dir+'/models/000.obj'

#change file names for easy file organisation
l_model_dir = list(model_dir)
if n_spins >=100:
    l_model_dir[-5] = str(n_spins)[2]
    l_model_dir[-6] = str(n_spins)[1]
    l_model_dir[-7] = str(n_spins)[0]
elif n_spins >=10:
    l_model_dir[-5] = str(n_spins)[1]
    l_model_dir[-6] = str(n_spins)[0]
else:
    l_model_dir[-5] = str(n_spins)[0]

model_dir = ''.join(l_model_dir)




models = []
for i in tqdm(range(0,reps)):
    MK = Model_Maker(n_spins, "Fully Connected Ising", str(n_spins) +" number: " +str(i))
    model = MK.model
    models.append(model)
    
    

fileObj = open(model_dir, 'wb')
pickle.dump(models,fileObj)
fileObj.close()
