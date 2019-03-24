
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook
import sklearn
from sklearn.ensemble import GradientBoostingClassifier 
prefix='../'


# In[2]:


def get_event(event):
    hits= pd.read_csv(prefix+'train_1/%s-hits.csv'%event)
    cells= pd.read_csv(prefix+'train_1/%s-cells.csv'%event)
    truth= pd.read_csv(prefix+'train_1/%s-truth.csv'%event)
    particles= pd.read_csv(prefix+'train_1/%s-particles.csv'%event)
    return hits, cells, truth, particles


# In[3]:


def convert_to_cylindrical_coordinates(hits):
    x, y = hits['x'], hits['y']
    hits['r'] = np.sqrt(x**2 + y**2)
    hits['a0'] = np.arctan2(y, x)
    return hits


# In[ ]:


# you can jump to step4 for test only.
train = True
if train:
    Train = []
    for i in tqdm_notebook(range(10,20)):
        event = 'event0000010%02d'%i
        hits, cells, truth, particles = get_event(event)
        hit_cells = cells.groupby(['hit_id']).value.count().values
        hit_value = cells.groupby(['hit_id']).value.sum().values
        #features = np.array(hits[['x','y','z']]/1000).reshape(len(hit_cells), 3)#, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
        hits = convert_to_cylindrical_coordinates(hits)
        features = np.hstack((np.hstack((np.array(hits['z']/1000).reshape(np.array(hits['x']).shape[0], 1), 
                                         np.array(hits['r']/1000).reshape(np.array(hits['r']).shape[0], 1))), 
                              np.array(hits['a0']).reshape(len(hits['a0'].iloc[:]), 1)))
        hit_signals = np.hstack((hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
        features = np.hstack((features, hit_signals))
        features = np.hstack((features, np.array(hits[['x','y','z']]/1000).reshape(len(hit_cells), 3)))
        particle_ids = truth.particle_id.unique()
        particle_ids = particle_ids[np.where(particle_ids!=0)[0]]

        pair = []
        for particle_id in particle_ids:
            hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
            for i in hit_ids:
                for j in hit_ids:
                    if i != j:
                        pair.append([i,j])
        pair = np.array(pair)   
        Train1 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.ones((len(pair),1))))

        if len(Train) == 0:
            Train = Train1
        else:
            Train = np.vstack((Train,Train1))

        n = len(hits)
        size = len(Train1)*3
        p_id = truth.particle_id.values
        i =np.random.randint(n, size=size)
        j =np.random.randint(n, size=size)
        pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))
        pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]

        Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))

        print(event, Train1.shape)

        Train = np.vstack((Train,Train0))
    del Train0, Train1

    np.random.shuffle(Train)
    print(Train.shape)


# In[54]:


s_Train = np.load('train10events.npy')


# In[49]:


clf = GradientBoostingClassifier(max_depth=10)


# In[50]:


get_ipython().run_cell_magic('time', '', 'clf.fit(Train[:20000,:-1], Train[:20000, -1])')


# In[51]:


get_ipython().run_cell_magic('time', '', 'clf.score(Train[-200000:, :-1], Train[-200000:, -1])')


# #### Using 1000 trees

# In[39]:


import sys
sys.getsizeof(Train)
with open('train10events.npy', 'wb') as file:
    np.save(file, Train)


# In[87]:


clf2 = GradientBoostingClassifier(max_depth = None, n_estimators = 1000)


# In[88]:


get_ipython().run_cell_magic('time', '', 'clf2.fit(Train[:100000,:-1], Train[:100000, -1])')


# In[59]:


get_ipython().run_cell_magic('time', '', '#clf.score(Train[-150000:, :-1], Train[-150000:, -1])\nclf.score(s_Train[:100000,:-1], s_Train[:100000, -1])')


# In[90]:


clf.n_estimators


# In[93]:


import pickle
with open('100_Gradient_Boost_96-5.p', 'wb') as file:
    pickle.dump(clf, file)


# In[6]:


import pickle
with open('100_Gradient_Boost_96-5.p', 'rb') as file:
    clf = pickle.load(file)


# In[ ]:


train = True
if train:
    Train = []
    for i in tqdm_notebook(range(10,20)):
        event = 'event0000010%02d'%i
        hits, cells, truth, particles = get_event(event)
        hit_cells = cells.groupby(['hit_id']).value.count().values
        hit_value = cells.groupby(['hit_id']).value.sum().values
        features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
        particle_ids = truth.particle_id.unique()
        particle_ids = particle_ids[np.where(particle_ids!=0)[0]]

        pair = []
        for particle_id in particle_ids:
            hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
            for i in hit_ids:
                for j in hit_ids:
                    if i != j:
                        pair.append([i,j])
        pair = np.array(pair)   
        Train1 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.ones((len(pair),1))))

        if len(Train) == 0:
            Train = Train1
        else:
            Train = np.vstack((Train,Train1))

        n = len(hits)
        size = len(Train1)*3
        p_id = truth.particle_id.values
        i = np.random.randint(n, size=size)
        j = np.random.randint(n, size=size)
        pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))
        pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]

        Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))

        print(event, Train1.shape)

        Train = np.vstack((Train,Train0))
    del Train0, Train1

    np.random.shuffle(Train)
    print(Train.shape)


# ####  Data Exploration

# In[6]:


hits, cells, truth, particles = get_event('event000001002')


# In[7]:


hit_cells = cells.groupby(['hit_id']).value.count().values


# In[13]:


hit_value = cells.groupby(['hit_id']).value.sum().values


# In[19]:


features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))


# In[21]:


features.shape


# In[40]:


Train_standard = None


# In[ ]:


94.88 only x, y, z


# In[ ]:


95.7


# In[16]:




