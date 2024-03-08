import pandas as pd
from plotnine import *

df = pd.read_csv('full_dataset.csv')
#print(ggplot(df) + aes(x='is_human', y='reaction_time') + geom_point())
#print(ggplot(df) + facet_grid(facets='. ~ is_human') + aes(x='block_type', y='reaction_time') + geom_point())
df['condition'] = ''
for i, row in df.iterrows():
    if row['block_type'] == 'stroop':
        if row['stroop_text'] == row['stroop_color']:
            df.at[i, 'condition'] = 'congruent'
        else:
            df.at[i, 'condition'] = 'incongruent'
    elif row['block_type'] == 'flanker':
        df.at[i, 'condition'] = row['flanker_type']
#print(ggplot(df) + facet_grid(facets='block_type ~ is_human') + aes(x='condition', y='reaction_time') + geom_point())
#print(ggplot(df) + facet_grid(facets='block_type ~ is_human') + aes(x='condition', y='reaction_time') + geom_jitter(width=0.2))
#print(ggplot(df) + facet_grid(facets='block_type ~ .') + aes(x='block_type', y='reaction_time', color='is_human') + geom_jitter(width=0.2))
#print(ggplot(df) + facet_grid(facets='block_type ~ .') + aes(x='block_type', y='reaction_time', color='is_human') + geom_jitter(position=position_jitterdodge(0.2)))
print(ggplot(df)+ facet_grid(facets="block_type ~ .") + aes(x='condition', y="reaction_time", color='is_human') + geom_jitter(position=position_jitterdodge(0.2)) + labs(x="Experimental condition", y="Reaction Time", title="Reaction times in our study",))

