import pandas as pd


def generate_features(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Spending_Score_Age_Ratio'] = df['Spending Score (1-100)'] / df['Age']
    df.to_csv(output_path, index=False)

    print("Новые признаки сгенерированы.")