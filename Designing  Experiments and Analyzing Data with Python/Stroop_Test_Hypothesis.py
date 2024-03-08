import pandas as pd
from pingouin import normality, wilcoxon
import random
import pingouin as pg  # we will use this library for hypothesis testing
from plotnine import *

def sample_from_data(dataframe):
    sorted_df = dataframe.sort_values(by=['participant', 'stroop_type'])
    def choose_trials_to_keep():
        ones = [1] * 127  # Corresponds to the congruent trials
        ones_and_zeros = [1] * 127 + [0] * 27  # Corresponds to the incongruent ones
        random.shuffle(ones_and_zeros)  # Shuffle the list randomly
        return ones + ones_and_zeros

    sorted_df['keep'] = choose_trials_to_keep()
    final_df = sorted_df[sorted_df['keep'] == 1].copy()

    return final_df

df = pd.read_csv('full_dataset.csv')

participants = df['participant'].unique()
# select only the Stroop data
stroop_df = df[df['block_type'] == 'stroop']

# get the "interquartile range" of the reaction times
q1 = stroop_df['reaction_time'].quantile(0.25)
q3 = stroop_df['reaction_time'].quantile(0.75)
iqr = q3 - q1
# calculate the threshold from which the values are considered "outliers"
outlier_threshold_top = q3 + 1.5 * iqr
outlier_threshold_bot = q1 - 1.5 * iqr
# effectively remove the outliers, just keeping data inside our `top` and `bottom` range
stroop_df = stroop_df[
    (stroop_df['reaction_time'] <= outlier_threshold_top) & (stroop_df['reaction_time'] >= outlier_threshold_bot)]
stroop_df['stroop_type'] = stroop_df.apply(lambda row: 'congruent' if row['stroop_text'] == row['stroop_color'] else 'incongruent', axis=1)
#print(ggplot(stroop_df) + aes(x='stroop_type', y='reaction_time') + geom_boxplot())
sampled_datasets = []
for participant in participants:
    participant_data = stroop_df[stroop_df['participant'] == participant]
    sampled_data = sample_from_data(participant_data)
    sampled_datasets.append(sampled_data)

# Perform Shapiro-Wilk test for normality on 'reaction_time' column for each 'stroop_type'
normality_results = stroop_df.groupby('stroop_type')['reaction_time'].apply(normality)
print(normality_results)

#print(stroop_df.groupby(['participant', 'stroop_type'])['reaction_time'].count())
# Concatenate the sampled datasets for all participants
stroop_df = pd.concat(sampled_datasets)
# Perform Wilcoxon signed-rank test
wilcoxon_results = wilcoxon(stroop_df['reaction_time'][stroop_df['stroop_type'] == 'congruent'],
                            stroop_df['reaction_time'][stroop_df['stroop_type'] == 'incongruent'],
                            alternative='two-sided')

#Print the results of Wilcoxon signed-rank test
print("\nWilcoxon signed-rank test:")
print(wilcoxon_results)

stroop_df.groupby(['stroop_type'])['is_correct'].mean()
# Create a new dataframe with congruent and incongruent values as columns
mcnemar_df = pd.DataFrame({
    "congruent": stroop_df.loc[
        stroop_df['stroop_type'] == 'congruent', 'is_correct'
    ].values,
    "incongruent": stroop_df.loc[
        stroop_df['stroop_type'] == 'incongruent', 'is_correct'
    ].values})



