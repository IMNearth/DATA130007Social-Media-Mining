#!/usr/bin/env python
# coding: utf-8

# draw a pic

# In[87]:


import pandas as pd
import csv
import networkx as nx
import matplotlib.pyplot as plt
import calendar
from collections import Counter
import math
import numpy as np

datapath = 'final_df_sum.csv'
data = pd.read_csv(datapath , encoding = 'utf8')

data = data.sort_values(by=['place_frd_possi'], ascending=False)

############################################################

print('data.shape[0]')
print(data.shape[0])

final_df = data[~data['entropy'].isin([0])]

print('final_df.shape[0]')
print(final_df.shape[0])

final_df = data[~data['place_frd_possi'].isin([0])]

print('final_df.shape[0]--2')
print(final_df.shape[0])

final_df['counter'] = range(len(final_df))

final_df['score'] = final_df['score'].map(lambda x: math.log10(x))

final_df['place_frd_possi'] = final_df['place_frd_possi'].map(lambda x: math.log10(x))

#final_df.head(50)
#plt.rcParams['figure.figsize'] = (12, 8)
#plt.scatter(final_df["counter"], final_df["entropy"], s = 15)

xxx = final_df['place_frd_possi']
y_entropy = final_df['entropy']
y_cnts = final_df['score']

###############################################

fig = plt.figure()

plt.rcParams['figure.figsize'] = (9, 6)

plt.scatter(y_cnts, y_entropy, s = 15, color='black', label = 'scatter')

fit1 = np.polyfit(y_cnts, y_entropy, 2)
p1 = np.poly1d(fit1)
x1 = np.linspace(0.3,4,100)
pp1 = p1(x1)

plt.plot(x1,pp1,color='b',label='functionfit')#100个x及对应y值绘制的曲线

plt.xlabel('number of "check-ins"')
plt.ylabel('entropy')

plt.xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
          ['$\mathregular{10^{0.5}}$', '$\mathregular{10^{1}}$', '$\mathregular{10^{1.5}}$','$\mathregular{10^{2}}$', '$\mathregular{10^{2.5}}$','$\mathregular{10^{3}}$', '$\mathregular{10^{3.5}}$', '$\mathregular{10^{4}}$'])

#plt.yticks([0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0],
#          ['$\mathregular{10^0}$', '$\mathregular{10^{-0.5}}$', '$\mathregular{10^{-1}}$','$\mathregular{10^{-1.5}}$', '$\mathregular{10^{-2}}$','$\mathregular{10^{-2.5}}$', '$\mathregular{10^{-3}}$'])


plt.legend()
plt.show()



'''
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.scatter(xxx, y_entropy, s = 15, label = 'entropy')
ax.plot(xxx, y_entropy, '-', label = 'entropy')
ax2 = ax.twinx()
#ax2.plot(xxx, y_cnts, '-r', label = 'cnts')
plt.scatter(xxx, y_cnts, s = 15, label = 'cnts')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("numbers")
ax.set_ylabel(r"entropy")
ax2.set_ylabel(r"cnts")
ax.set_ylim(0,5.5)
ax2.set_ylim(0,10)
ax2.legend(loc=0)
#plt.savefig('0.png')

plt.show
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




