import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)

def load_data(file_path):
    """Loads the dataset."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def remove_duplicate_rows(df):
    """Removes exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")
    return df

def fix_column_spacing(df):
    """Strips leading/trailing spaces from column names."""
    df.columns = df.columns.str.strip()
    return df

def format_hourly_rate(df, column="HourlyRate"):
    """Converts HourlyRate from string format ('$XX.YY ') to a float with two decimal places."""
    if column in df.columns:
        df[column] = df[column].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        df[column] = df[column].round(2)
    return df

def normalize_categorical_values(df, categorical_columns):
    """Fixes inconsistent categorical values using lemmatization and specific rules."""
    lemmatizer = WordNetLemmatizer()

    def normalize(value):
        if isinstance(value, str):
            value = value.replace("_", " ")
            words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z0-9])|[A-Z0-9]+', value)
            lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
            normalized_value = ' '.join(lemmatized_words)
            if "mailed check" in normalized_value:
                return "mail check"
            return normalized_value
        return value

    for col in categorical_columns:
        df[col] = df[col].apply(normalize)
    return df

def handle_missing_values(df):
    """Fills missing values based on column type."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(pd.NA, inplace=True)
        else:
            df[col].fillna(np.nan, inplace=True)
    print("Filled missing values appropriately.")
    return df

def fix_numerical_values(df, numerical_columns):
    """Removes non-numeric characters from numerical columns and converts them to float."""
    for col in numerical_columns:
        df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fix_negative_values(df, columns_to_fix):
    """Converts negative values in certain columns to their absolute values."""
    for col in columns_to_fix:
        df[col] = df[col].abs()
    return df

def detect_and_handle_outliers(df, numerical_columns):
    """Detects and handles outliers using the IQR method."""
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    print("Handled outliers using IQR method.")
    return df

def save_cleaned_data(df, output_file):
    """Saves the cleaned dataset."""
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")

def compare_dataframes(original_file, updated_file, comparison_file="dataframe_comparison.txt"):
    """
    Compares two CSV files, outputs statistics to terminal and a text file.

    Args:
        original_file (str): Path to the original CSV file.
        updated_file (str): Path to the updated CSV file.
        comparison_file (str): Path to the output text file.
    """

    try:
        original_df = pd.read_csv(original_file)
        updated_df = pd.read_csv(updated_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    original_rows, original_cols = original_df.shape
    updated_rows, updated_cols = updated_df.shape

    row_diff = updated_rows - original_rows
    col_diff = updated_cols - original_cols

    with open(comparison_file, "w") as f:
        f.write("DATAFRAME COMPARISON\n")
        f.write("---------------------\n")
        f.write(f"Original: {original_rows} rows, {original_cols} columns\n")
        f.write(f"Updated:  {updated_rows} rows, {updated_cols} columns\n")
        f.write(f"Row Difference: {row_diff}\n")
        f.write(f"Column Difference: {col_diff}\n")

    print("DATAFRAME COMPARISON")
    print("---------------------")
    print(f"Original: {original_rows} rows, {original_cols} columns")
    print(f"Updated:  {updated_rows} rows, {updated_cols} columns")
    print(f"Row Difference: {row_diff}")
    print(f"Column Difference: {col_diff}")
    print(f"Comparison results saved to {comparison_file}")

def output_non_numeric_values(df, output_file="non_numeric_values.txt"):
    """
    Outputs all non-numeric variable possibilities, sorted by column, to a file.
    Handles pd.NA values.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        output_file (str): The path to the output text file.
    """
    non_numeric_values = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].unique()
            unique_values = [val for val in unique_values if pd.notna(val)]
            unique_values.sort()
            non_numeric_values[col] = unique_values

    with open(output_file, "w") as f:
        f.write("NON-NUMERIC VARIABLE POSSIBILITIES\n")
        f.write("-------------------------------------\n")
        for col, values in non_numeric_values.items():
            f.write(f"Column: {col}\n")
            for value in values:
                f.write(f"  - {value}\n")
            f.write("\n")

    print(f"Non-numeric variable possibilities saved to {output_file}")


if __name__ == "__main__":
    file_path = "Employee Turnover Dataset.csv"
    original_file = file_path
    output_file = "Cleaned_Employee_Turnover.csv"
    updated_file = output_file
    
    df = load_data(file_path)
    if df is not None:
        df = fix_column_spacing(df)
        df = handle_missing_values(df)
        original_df = df.copy()
        df = format_hourly_rate(df)
        df = normalize_categorical_values(df, ["JobRoleArea", "PaycheckMethod"])
        df = fix_numerical_values(df, ["NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"])
        df = fix_negative_values(df, ["AnnualSalary", "DrivingCommuterDistance"])
        df = detect_and_handle_outliers(df, ["AnnualSalary", "DrivingCommuterDistance"])
        df = remove_duplicate_rows(df)
        output_non_numeric_values(df)
        save_cleaned_data(df, output_file)