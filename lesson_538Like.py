import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/daily-show-guests/daily_show_guests.csv')

df.head()
df.rename(columns={'YEAR': 'Year', 'Raw_Guest_List': 'Guest'}, inplace=True)

def get_occupation(group):
    if group in ['Acting', 'Comedy', 'Musician']:
        return 'Acting, Comedy & Music'
    elif group in ['Media', 'media']:
        return 'Media'
    elif group in ['Government', 'Politician', 'Political Aide']:
        return 'Government and Politics'
    else:
        return 'Other'
df['Occupation'] = df['Group'].apply(get_occupation)
#  create a table with the percentage of guests according to occupations each year
ct = pd.crosstab(df.Year, df.Occupation, normalize='index')*100
ct = ct.drop(columns=['Other'])
year = ct.index.tolist() #make a list of all the years in our table

# Start to make some nice graphs
# Setting FiveThirtyEight style

# Setting FiveThirtyEight style
plt.style.use('fivethirtyeight')

# Setting size of our plot
fig, ax = plt.subplots(figsize=(9,6))
    
# Plotting each occupation category
ax1 = sns.lineplot(x=year, y=ct['Acting, Comedy & Music'].tolist(), color='#0F95D7', lw=2.5)
ax2 = sns.lineplot(x=year, y=ct['Government and Politics'].tolist(), color='#FF2700', lw=2.5)
ax3 = sns.lineplot(x=year, y=ct['Media'].tolist(), color='#810F7C', lw=2.5)

# Y axis past 0 & above 100 -- grid line will pass 0 & 100 marker
plt.ylim(-5,110)

# Bolded horizontal line at y=0
ax1.axhline(y=0, color='#414141', linewidth=1.5, alpha=.5)

# Y-labels to only these
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(labels=['0', '25', '50', '75', '100%'], fontsize=14, color='#414141')

# X-labels and changing label names
ax.set_xticks([2000, 2004, 2008, 2012])
ax.set_xticklabels(['2000', '2004', '20008', '2012'], fontsize=14, color='#414141')

# Title text
ax.text(x=1996.7, y=118, s="Who Got to Be On 'The Daily Show'?", fontsize=18.5, fontweight='semibold', color='#414141')

# Subtitle text
ax.text(x=1996.7, y=112, s='Occupation of guests, by year', fontsize=16.5, color='#414141')

# Text labels for each plotted line
ax.text(x=2000.5, y=81, s="Acting, Comedy & Music", fontsize=13, fontweight='semibold', color='#0F95D7')
ax.text(x=2008.5, y=6, s="Government and Politics", fontsize=13, fontweight='semibold', color='#FF2700')
ax.text(x=2007.1, y=52, s="Media", fontsize=13, fontweight='semibold', color='#810F7C')

# Line at bottom for signature line
ax1.text(x = 1996.7, y = -18.5,
    s = '   ©Carlos Gutierrez                                                           Source: FiveThirtyEight   ',
    fontsize = 14, color = '#f0f0f0', backgroundcolor = '#414141');
plt.show()
# plt.savefig('figName.eps')  
