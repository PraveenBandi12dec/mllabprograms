import pandas as pd

def find_s_algorithm(filepath):

    print("Program started\n")

    data = pd.read_csv(filepath)

    print("Training Data:\n")
    print(data)

    attributes = data.columns[:-1]
    class_label = data.columns[-1]

    hypothesis = ['?' for _ in attributes]

    for index, row in data.iterrows():

        if row[class_label] == 'Yes':

            for i, value in enumerate(row[attributes]):

                if hypothesis[i] == '?':
                    hypothesis[i] = value

                elif hypothesis[i] != value:
                    hypothesis[i] = '?'

    print("\nFinal Hypothesis:")
    print(hypothesis)


find_s_algorithm("play_tennis_dataset.csv")