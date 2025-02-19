import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import seaborn as sns

# Correct file path syntax
file_path = r"C:\Users\Mallikarjuna Rao\Downloads\ML_Assignment02_BL.EN.U4AIE23122\ML_Assignment02_BL.EN.U4AIE23122\Lab Session Data.xlsx"

# Load the Excel file
xls = pd.ExcelFile(file_path)

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
        print("A1 Results:")
        print(f"Dimensionality: {dimensionality}")
        print(f"Number of Vectors: {num_vectors}")
        print(f"Rank of A: {rank_A}")
        print(f"Product Costs: {product_costs}")
        return dimensionality, num_vectors, rank_A, product_costs
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

def classify_customers():
    try:
        df = pd.read_excel(xls, sheet_name="Purchase data")
        df["Customer Class"] = df.iloc[:, 4].apply(lambda x: "RICH" if x > 200 else "POOR")
        print("A3 Results: Customer Classification Done")
        return df[["Customer Class"]]
    except Exception as e:
        print(f"An error occurred in customer classification: {e}")
        return None

# A5-A10: Data Exploration, Normalization, Similarity Measures, and Heatmap
def analyze_stock_data():
    try:
        df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")
        price_data = df.iloc[:, 3]
        mean_price = statistics.mean(price_data)
        variance_price = statistics.variance(price_data)
        print("A5 Results:")
        print(f"Mean Price: {mean_price}")
        print(f"Variance Price: {variance_price}")
    except Exception as e:
        print(f"An error occurred in stock data analysis: {e}")

def compute_similarity():
    try:
        df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
        bin_data = df.iloc[:, :2].values
        f11 = np.sum((bin_data[:, 0] == 1) & (bin_data[:, 1] == 1))
        f00 = np.sum((bin_data[:, 0] == 0) & (bin_data[:, 1] == 0))
        f10 = np.sum((bin_data[:, 0] == 1) & (bin_data[:, 1] == 0))
        f01 = np.sum((bin_data[:, 0] == 0) & (bin_data[:, 1] == 1))
        jc = f11 / (f01 + f10 + f11)
        smc = (f11 + f00) / (f00 + f01 + f10 + f11)
        print("A8 Results:")
        print(f"Jaccard Coefficient: {jc}")
        print(f"Simple Matching Coefficient: {smc}")
    except Exception as e:
        print(f"An error occurred in similarity computation: {e}")

def plot_heatmap():
    try:
        df = pd.read_excel(xls, sheet_name="thyroid0387_UCI")
        data_sample = df.iloc[:20, :].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(data_sample, annot=True, cmap="coolwarm")
        plt.title("A10: Heatmap of Similarity Measures")
        plt.show()
    except Exception as e:
        print(f"An error occurred in heatmap plotting: {e}")

if __name__ == "__main__":
    analyze_purchase_data()
    classify_customers()
    analyze_stock_data()
    compute_similarity()
    plot_heatmap()
