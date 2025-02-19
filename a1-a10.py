import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Correct file path syntax
file_path = r"C:\Users\Mallikarjuna Rao\Downloads\ML_Assignment02_BL.EN.U4AIE23122\ML_Assignment02_BL.EN.U4AIE23122\Lab Session Data.xlsx"

# Load the Excel file
try:
    xls = pd.ExcelFile(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# A1-A4: Data Loading, Matrix Analysis, Cost Prediction, and Classification
def analyze_purchase_data():
    try:
        df = pd.read_excel(xls, sheet_name="Purchase data")
        purchase_matrix = df.iloc[:, 1:4].values
        purchase_amounts = df.iloc[:, 4].values.reshape(-1, 1)
        dimensionality = purchase_matrix.shape[1]
        num_vectors = purchase_matrix.shape[0]
        rank_A = np.linalg.matrix_rank(purchase_matrix)
        purchase_matrix_pinv = np.linalg.pinv(purchase_matrix)
        product_costs = np.dot(purchase_matrix_pinv, purchase_amounts).flatten()
        
        print("\nA1 Results: Purchase Data Analysis")
        print(f"Dimensionality: {dimensionality}")
        print(f"Number of Vectors: {num_vectors}")
        print(f"Rank of A: {rank_A}")
        print(f"Product Costs: {product_costs}")

    except Exception as e:
        print(f"An error occurred in purchase data analysis: {e}")

def classify_customers():
    try:
        df = pd.read_excel(xls, sheet_name="Purchase data")
        df["Customer Class"] = df.iloc[:, 4].apply(lambda x: "RICH" if x > 200 else "POOR")

        print("\nA3 Results: Customer Classification Completed")
        print(df[["Customer Class"]].head(10))  # Display first 10 classifications

    except Exception as e:
        print(f"An error occurred in customer classification: {e}")

# A5-A10: Data Exploration, Normalization, Similarity Measures, and Heatmap
def analyze_stock_data():
    try:
        df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")
        price_data = df.iloc[:, 3]

        mean_price = statistics.mean(price_data)
        variance_price = statistics.variance(price_data)

        print("\nA5 Results: IRCTC Stock Data Analysis")
        print(f"Mean Price: {mean_price:.2f}")
        print(f"Variance in Price: {variance_price:.2f}")

    except Exception as e:
        print(f"An error occurred in stock data analysis: {e}")

def compute_similarity():
    try:
        df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
        df.replace({'t': 1, 'f': 0}, inplace=True)  # Ensure binary values are numeric

        bin_data = df.iloc[:, :2].values
        f11 = np.sum((bin_data[:, 0] == 1) & (bin_data[:, 1] == 1))
        f00 = np.sum((bin_data[:, 0] == 0) & (bin_data[:, 1] == 0))
        f10 = np.sum((bin_data[:, 0] == 1) & (bin_data[:, 1] == 0))
        f01 = np.sum((bin_data[:, 0] == 0) & (bin_data[:, 1] == 1))

        jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
        smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

        print("\nA8 Results: Similarity Computation")
        print(f"Jaccard Coefficient: {jc:.4f}")
        print(f"Simple Matching Coefficient: {smc:.4f}")

    except Exception as e:
        print(f"An error occurred in similarity computation: {e}")

def plot_heatmap():
    try:
        df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
        df.replace({'t': 1, 'f': 0}, inplace=True)

        # Extract binary columns
        binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
        df_binary = df[binary_cols].astype(int)

        # Extract numeric columns
        df_numeric = df.select_dtypes(include=['int64', 'float64']).fillna(0)
        df_values = df_numeric.to_numpy()

        # Compute similarity matrices
        jc_matrix = np.zeros((20, 20))
        smc_matrix = np.zeros((20, 20))
        cos_matrix = cosine_similarity(df_values[:20])

        def compute_jc_smc(vec1, vec2):
            intersection = np.sum((vec1 == 1) & (vec2 == 1))
            union = np.sum((vec1 == 1) | (vec2 == 1))
            matches = np.sum(vec1 == vec2)
            total_attributes = len(vec1)

            jc = intersection / union if union > 0 else 0
            smc = matches / total_attributes
            return jc, smc

        for i in range(20):
            for j in range(20):
                jc_matrix[i, j], smc_matrix[i, j] = compute_jc_smc(df_binary.iloc[i].values, df_binary.iloc[j].values)

        # Convert to DataFrames
        jc_df = pd.DataFrame(jc_matrix, index=range(1, 21), columns=range(1, 21))
        smc_df = pd.DataFrame(smc_matrix, index=range(1, 21), columns=range(1, 21))
        cos_df = pd.DataFrame(cos_matrix, index=range(1, 21), columns=range(1, 21))

        # Plot heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sns.heatmap(jc_df, ax=axes[0], cmap="coolwarm", annot=False)
        axes[0].set_title("Jaccard Coefficient (JC)")
        sns.heatmap(smc_df, ax=axes[1], cmap="coolwarm", annot=False)
        axes[1].set_title("Simple Matching Coefficient (SMC)")
        sns.heatmap(cos_df, ax=axes[2], cmap="coolwarm", annot=False)
        axes[2].set_title("Cosine Similarity (COS)")

        plt.tight_layout()
        plt.show()

        print("\nA10: Heatmap successfully plotted!")

    except Exception as e:
        print(f"An error occurred in heatmap plotting: {e}")

if __name__ == "__main__":
    analyze_purchase_data()
    classify_customers()
    analyze_stock_data()
    compute_similarity()
    plot_heatmap()
