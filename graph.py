import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pkl
import main
#from __future__ import division

sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['len of url'],color='green',label='Benign URLs')
sns.distplot(featureSet[featureSet['label']=='1']['len of url'],color='red',label='Phishing URLs')
import matplotlib.pyplot as plt
plt.title('Url Length Distribution')
plt.legend(loc='upper right')
plt.xlabel('Length of URL')
plt.show()

