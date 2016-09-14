
# coding: utf-8

# ## 1. Basic string operations
# 
# Install Jupyter notebook : <a href="http://jupyter.readthedocs.io/en/latest/install.html">http://jupyter.readthedocs.io/en/latest/install.html</a>

# ### 1.1 Concatenate two strings

# In[1]:

"Tensorflow " + "is awesome"


# ### 1.2 Split and tokenize a string

# In[2]:

"It is hard to install Tensorflow on windows.".split()


# In[3]:

from nltk import word_tokenize


# In[4]:

word_tokenize("It is hard to install Tensorflow on windows.")


# ### 1.3 Remove line separators and white space at the begin and the end of a string.

# In[5]:

"It is hard to install Tensorflow on windows.\n".strip()


# ### 1.4 String formatting.

# In[6]:

num_sens = 100
print("The number of sentences in this document is %s ." % num_sens)


# In[7]:

num_gold_medals = 3
athlete = 'Usain Bolt'
print('{} won {} gold medals in Rio.'.format(athlete, num_gold_medals))


# ## 2. List

# ### 2.1 Iterate over a list

# In[8]:

gold_medals = [46, 27, 26, 19, 17]
for g in gold_medals:
    print('number of gold medals is {}.'.format(g))
    


# ### 2.2 Extend a list with another list

# In[9]:

gold_medals.extend([12,10])
print(gold_medals)


# ### 2.3 Length of a list

# In[10]:

print(len(gold_medals))


# ### 2.4 Append one element to the end of a list

# In[11]:

gold_medals.append(9)
print(gold_medals)


# ## 3 Dictionary

# ## 3.1 Iterate over a dictionary.

# In[16]:

country_gold_medals = {'USA':46,'UK':27}
for c in country_gold_medals:
    print('{} : {}'.format(c, country_gold_medals[c]))


# ### 3.2 Update dictionary.

# In[17]:

country_gold_medals['China'] = 26


# ### 3.3 Check if a key exists in a dictionary.

# In[19]:

if 'China' in country_gold_medals:
    print(country_gold_medals['China'])


# ## 4. Tuple

# In[20]:

c_g = ('USA', 47)
print(c_g)


# In[23]:

print('{} : {}'.format(c_g[0],c_g[1]))


# ## 5. Tensorflow Operations

# In[24]:

import tensorflow as tf
import numpy as np


# In[40]:

with tf.Session() as sess:
    a = [[1,2,3],[3,2,1],[1,0,1]]
    v = [[0,1,0],[2,1,2]]
    print(sess.run(tf.matmul(a, v,transpose_b=True)))
    print("row_mean = %s " % sess.run(tf.reduce_mean(v, 0)))


# In[ ]:



