import pandas as pd
import random


def stats_by_block(df):
    block_csv_column = df['block_csv']
    block_names = block_csv_column.unique()
    stats = {}

    for block_name in block_names:
        block = df[block_csv_column == block_name]

        # Calculate statistics for the block
        reaction_time_mean = block['Time'].mean()
        reaction_time_std = block['Time'].std()
        accuracy_mean = block['Is_Correct?'].mean()
        accuracy_std = block['Is_Correct?'].std()

        # Create a dictionary to store the block's statistics
        block_stats = {
            'Reaction Time': (reaction_time_mean, reaction_time_std),
            'Accuracy': (accuracy_mean, accuracy_std)
        }

        # Add the block's statistics to the overall stats dictionary
        stats[block_name] = block_stats

    return stats


def positivize(number):
    return abs(number)


def generate_new_responses(block):
    """Generates new values for the 'response' column based on 'is_correct' and 'block'"""

    new_responses = []
    for _, row in block.iterrows():
        block_type = row['Test_Name']
        if block_type == 'FlankerTrial':
            possible_responses = ['Left', 'Right']
        elif block_type == 'StroopTrial':
            possible_responses = ['Red', 'Green', 'Blue']
        else:
            possible_responses = []
        is_correct = row['Is_Correct?']
        correct_response = row['Correct']

        if is_correct:
            response = correct_response
        else:
            response = random.choice([r for r in possible_responses if r != correct_response])

        new_responses.append(response)

    block['Response'] = new_responses
    return block


def generate_random_data(df, stats):
    block_names = df['block_csv'].unique()
    all_blocks = []

    for block_name in block_names:
        # Slice `df` getting rows where 'block_csv' is `block_name`
        block = df[df['block_csv'] == block_name].copy()

        # Generate new reaction times
        mean, std = stats[block_name]['Reaction Time']
        reaction_times = [positivize(random.gauss(mean, std)) for _ in range(len(block))]
        block['Time'] = reaction_times

        # Generate new is correct values
        accuracy = stats[block_name]['Accuracy'][0]
        is_correct_values = [random.choices([True, False], [accuracy, 1 - accuracy])[0] for _ in range(len(block))]
        block['Is_Correct?'] = is_correct_values

        # Generate new response values
        block = generate_new_responses(block)

        # Add the modified block to the list
        all_blocks.append(block)

    # Concatenate all blocks into a single DataFrame
    random_data = pd.concat(all_blocks)

    return random_data


df = pd.read_csv('output.csv')
stats = stats_by_block(df)
random_data = generate_random_data(df, stats)
random_data.to_csv('random.csv', index=False)
