import pandas as pd
import numpy as np
import re
import sys
import enchant
import nltk
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

#### ANALYSIS #######

def check_text_column_variations(df):
    """
    Checks text-based columns for variations of the same letter sets and outputs them.

    Args:
        df (pd.DataFrame): The DataFrame to check.
    """

    text_cols = df.select_dtypes(include=['object']).columns
    variations_found = {}

    for col in text_cols:
        if df[col].astype(str).str.contains('\$').any():
            continue

        unique_values = df[col].unique()
        normalized_values = {}

        for value in unique_values:
            if isinstance(value, str):
                normalized = re.sub(r'[^a-zA-Z]', '', value.lower())
                if normalized:
                    if normalized not in normalized_values:
                        normalized_values[normalized] = []
                    normalized_values[normalized].append(value)

        for normalized, original_values in normalized_values.items():
            if len(original_values) > 1:
                if normalized not in variations_found:
                    variations_found[normalized] = []
                variations_found[normalized].extend(original_values)

    if variations_found:
        print("\nDuplicate variable names found:")
        for normalized, original_values in variations_found.items():
            original_values = list(set(original_values))
            if len(original_values) > 1:
                print(f"  {original_values[0]}: {original_values}")

    else:
        print("No duplicate variable name variations found.")

def check_column_spacing(df):
    """Checks if column names have inconsistent spacing."""
    spaces = [col.endswith(' ') for col in df.columns]
    if all(spaces) or not any(spaces):
        return None
    else:
        inconsistent_cols = [col for col in df.columns if col.endswith(' ')]
        non_space_cols = [col for col in df.columns if not col.endswith(' ')]
        majority_cols_type = "columns with spaces" if len(inconsistent_cols) > len(non_space_cols) else "columns without spaces"
        minority_cols = inconsistent_cols if len(inconsistent_cols) < len(non_space_cols) else non_space_cols
        return f"Warning: Inconsistent spacing in column names. Majority are {majority_cols_type}.\nPossible errored columns are: {minority_cols}"
    
def spell_check_columns(df, ignore_words=None):
    """Checks column names for spelling errors, splitting by capitals or underscores."""
    if ignore_words is None:
        ignore_words = set()
    d = enchant.Dict("en_US")
    errors = []
    for col in df.columns:
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z0-9])|[A-Z0-9]+|\d+|\S', col)
        for word in words:
            word = word.replace('_', '')
            word = word.replace('.', '')
            if word and word.lower() not in ignore_words and not d.check(word):
                errors.append(f"Spelling error in column '{col}': '{word}'")
    if errors:
        output = "Warning: Spelling errors in column names:\n" + "\n".join(errors)
        while "  " in output:
            output = output.replace("  ", " ")
        output = output.rstrip()
        return output
    return None

def normalize_and_split(value):
    """Splits a string by camel case/underscores, converts to lowercase."""
    if isinstance(value, str):
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z0-9])|[A-Z0-9]+|\d+|\S', value)
        words = [word for word in words if word != '_']
        output = ' '.join(word.lower() for word in words)
        while "  " in output:
            output = output.replace("  ", " ")
        output = output.rstrip()
        return output
    return str(value).lower()


def check_formatting_errors(df):
    """Performs comprehensive formatting error checks on the DataFrame."""

    # 1. Data Type Consistency Checks
    for col in df.columns:
        types = df[col].apply(type).unique()
        if len(types) > 1:
            print(f"Warning: Mixed data types in column '{col}': {types}")

        if pd.api.types.is_bool_dtype(df[col]):
            unique_values = df[col].unique()
            if not set(unique_values).issubset({True, False}):
                print(f"Warning: Inconsistent boolean representation in column '{col}': {unique_values}")
        elif df[col].isin([0, 1]).all() and pd.api.types.is_numeric_dtype(df[col]):
            if not set(df[col].unique()).issubset({0,1}):
                print(f"Warning: Inconsistent boolean representation in column '{col}': {df[col].unique()}")

    # 2. String/Text Formatting Checks
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].apply(lambda x: isinstance(x, str) and x != x.strip() if isinstance(x, str) else False).any():
                print(f"Warning: Leading/trailing spaces in column '{col}'")
            
            non_alphanumeric_mask = df[col].apply(lambda x: isinstance(x, str) and not re.match(r'^[A-Za-z0-9\s]*$', x) if isinstance(x,str) else False)
            if non_alphanumeric_mask.any():
                print(f"Warning: Non alphanumeric characters found in column: '{col}'")
                
                # Check for "N/A" or "n/a"
                na_mask = df[col].isin(["N/A", "n/a"]) & non_alphanumeric_mask
                if na_mask.any():
                    print(f"Warning: Expected NULL, 'N/A' used instead in column: '{col}'")
            
            empty_strings = df[df[col] == '']
            if not empty_strings.empty:
                print(f"Warning: Empty strings found in column '{col}': {len(empty_strings)} occurrences")
            
            control_chars = df[df[col].str.contains(r'[\x00-\x1F\x7F]')==True]
            if not control_chars.empty:
                print(f"Warning: Control characters found in column '{col}': {len(control_chars)} occurrences")
            
            email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            url_regex = r"^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/\S*)?$"
            emails = df[df[col].str.contains('@')==True]
            urls = df[df[col].str.contains('http')==True]
            for x in emails[col]:
                if not re.match(email_regex, str(x)):
                    print(f"Warning: email format error in column {col}, value: {x}")
            for x in urls[col]:
                if not re.match(url_regex, str(x)):
                    print(f"Warning: url format error in column {col}, value: {x}")

    # 3. Numerical Formatting Checks
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].astype(str).str.contains(r'[^\d.-]').any():
            print(f"Warning: Non-numeric characters in column '{col}'")
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in column '{col}'")
        if (df[col] == 0).any():
            print(f"Warning: Zero values found in column '{col}'")

    # 4. Date/Time Formatting Checks
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            invalid_dates = df[df[col].isnull()]
            if not invalid_dates.empty:
                print(f"Warning: Invalid dates found in column '{col}'")

def inspect_dataset(df, output_dir):
    
    output_file = f"{output_dir}/data_inspection.txt"
    duplicate_file = f"{output_dir}/data_inspection_duplicates.txt"
    outlier_file = f"{output_dir}/data_inspection_outliers.txt"
    
    """
    Inspects the dataset for duplicate entries, missing values, inconsistent entries, formatting errors, and outliers.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        output_file (str): The path to the main output file.
        duplicate_file (str): The path to the duplicate rows output file.
        outlier_file (str): The path to the outliers output file.
        duplicate_exclude(list): A list of columns to exclude when checking for duplicate rows.
    """
    with open(output_file, "w") as f:
        sys.stdout = f
        print("--- 1. Column Spacing and Spelling Inspection ---")
        
        print("\n--- Column Names with Quotes (Raw) ---")
        for col in df.columns:
            print(f"'{col}'")
        print("\n")
        spacing_warning = check_column_spacing(df)
        spelling_warning = spell_check_columns(df)
        if spacing_warning:
            print(spacing_warning)
            print("\n")
        if spelling_warning:
            print(spelling_warning)
            print("\n")
        
        print("\n--- 2. Missing Values Inspection ---\n")
        missing_values = df.isnull().sum()
        nonzero_missing_values = missing_values[missing_values > 0] if any(missing_values > 0) else pd.Series()
        print("Missing values per column:\n", nonzero_missing_values, sep='')

        missing_percentage = (df.isnull().sum() / len(df)) * 100
        nonzero_missing_percentage = missing_percentage[missing_percentage > 0] if any(missing_percentage > 0) else pd.Series()
        print("\nPercentage of Missing Values per Column:\n", nonzero_missing_percentage, sep='')


        print("\n--- 3. Duplicate Entries Inspection ---")
        cols_to_exclude = []
        cols_to_check = [col for col in df.columns if col not in cols_to_exclude]
        duplicate_rows = df[df.duplicated(subset=cols_to_check, keep=False)]

        if not duplicate_rows.empty:
            print("Duplicate rows found. See data_inspection_duplicates.txt for details.")
            with open(duplicate_file, "w") as df_out:
                sys.stdout = df_out
                print("--- Duplicate Rows (Excluding Specified Columns) ---")

                dr_counter = 0

                grouped = duplicate_rows.groupby(cols_to_check)

                for _, group in grouped:
                    indexes = ", ".join(map(str, group.index.tolist()))
                    print(f"{indexes}:")

                    for _, row in group.iterrows():
                        row_str = ", ".join(map(str, row.tolist()))
                        print(row_str)

                    dr_counter = dr_counter + 1

                    print()

                print("Total duplicate rows:", dr_counter)

                sys.stdout = f
        else:
            print("Findings: No duplicate rows found.")
        

        
        print("\n--- 4. Inconsistent Entries Inspection ---")
        iee = 0
        categorical_cols = df.select_dtypes(include=['object']).columns


        normalized_series = df[col].apply(normalize_and_split)
        duplicates = normalized_series[normalized_series.duplicated(keep=False)]
        if not duplicates.empty:
            print(f"Duplicate values found in:")
        else:
            pass

        for col in categorical_cols:
            if col != 'HourlyRate ':
                normalized_series = df[col].apply(normalize_and_split)
                duplicates = normalized_series[normalized_series.duplicated(keep=False)]
                if not duplicates.empty:
                    print(f"{col} (normalized).")
                    with open(duplicate_file, "a") as df_out:
                        original_stdout = sys.stdout
                        sys.stdout = df_out
                        print(f"\nDuplicate values in {col}:")
                        duplicate_counts = duplicates.value_counts()
                        for value, count in duplicate_counts.items():
                            print(f"Value: '{value}', Count: {count}")
                            iee = iee + 1
                        sys.stdout = original_stdout
                else:
                    print(f"Findings: No logical duplicates found in {col}.")
        
        


        
        check_text_column_variations(df)

        if iee > 0:
            print(f"\nSee {duplicate_file} for details.")



        print("\n--- 5. Formatting Errors Inspection ---")

        check_formatting_errors(df)
        
        print("\n--- 6. Outliers Inspection ---")
        numerical_cols = df.select_dtypes(include=['number']).columns
        outlier_data = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                print(f"Outliers found in {col}. See {outlier_file} for details.")
                outlier_data[col] = outliers[['EmployeeNumber', col]].copy()
                outlier_data[col]['Index'] = outliers.index
        
        if outlier_data:
            with open(outlier_file, "w") as out_out:
                outlier_counter = 0
                
                sys.stdout = out_out

                print("--- Outlier Details ---")
                for col, data in outlier_data.items():
                    print(f"\nOutliers in {col}:")
                    print("------------------------------------------------------------------")
                    print("{:<10} {:<15} {:<15}".format("Index", "EmployeeNumber", "Value"))
                    print("------------------------------------------------------------------")

                    for index, row in data.iterrows():
                        print("{:<10d} {:<15d} {:<15}".format(int(row['Index']), int(row['EmployeeNumber']), row[col]))
                        outlier_counter = outlier_counter + 1

                print("\nTotal Outliers: ", outlier_counter, sep = '')

                sys.stdout = f
        else:
            print("No outliers found during outlier processing.")
        
        sys.stdout = sys.__stdout__
        print(f"Inspection results written to {output_file}")
        
        


#### CLEANUP ####

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

def fix_dataset(file_path):
    """Fixes the dataset by performing various cleaning operations."""
    df = load_data(file_path)
    if df is not None:
        df = fix_column_spacing(df)
        df = handle_missing_values(df)
        df = format_hourly_rate(df)
        df = normalize_categorical_values(df, ["JobRoleArea", "PaycheckMethod"])
        df = fix_numerical_values(df, ["NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"])
        df = fix_negative_values(df, ["AnnualSalary", "DrivingCommuterDistance"])
        df = detect_and_handle_outliers(df, ["AnnualSalary", "DrivingCommuterDistance"])
        df = remove_duplicate_rows(df)
        output_non_numeric_values(df)
        return df
    return None



#### DASHBOARD ####

def browse_input_file():
    """Opens a file dialog to select the input file."""
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    input_entry.delete(0, tk.END)
    input_entry.insert(0, filepath)

def browse_output_dir():
    """Opens a directory dialog to select the output directory."""
    directory = filedialog.askdirectory()
    output_entry.delete(0, tk.END)
    output_entry.insert(0, directory)

def run_analysis_correction():
    input_filepath = input_entry.get()
    output_directory = output_entry.get()
    
    analysis_selected = analysis_var.get()
    correction_selected = correction_var.get()
    view_terminal = terminal_var.get()
    
    analysis_checks = [check for check, var in analysis_checks_vars.items() if var.get()]
    correction_checks = [check for check, var in correction_checks_vars.items() if var.get()]
    
    terminal_output.delete(1.0, tk.END)
    
    if input_filepath and output_directory:
        df = load_data(input_filepath)
        if df is not None:
            if analysis_selected:
                inspect_dataset(df, output_directory)
                with open(f"{output_directory}/data_inspection.txt", "r") as file:
                    terminal_output.insert(tk.END, file.read())
            if correction_selected:
                df = fix_column_spacing(df)
                df = handle_missing_values(df)
                df = format_hourly_rate(df)
                df = normalize_categorical_values(df, ["JobRoleArea", "PaycheckMethod"])
                df = fix_numerical_values(df, ["NumCompaniesPreviouslyWorked", "AnnualProfessionalDevHrs"])
                df = fix_negative_values(df, ["AnnualSalary", "DrivingCommuterDistance"])
                df = detect_and_handle_outliers(df, ["AnnualSalary", "DrivingCommuterDistance"])
                df = remove_duplicate_rows(df)
                save_cleaned_data(df, f"{output_directory}/Cleaned_Employee_Turnover.csv")
                output_non_numeric_values(df, f"{output_directory}/non_numeric_values.txt")
            
            terminal_output.insert(tk.END, "Process completed successfully.\n")
        else:
            terminal_output.insert(tk.END, "Failed to load the dataset.\n")
    else:
        terminal_output.insert(tk.END, "Please select input file and output directory.\n")
    
    if view_terminal:
        terminal_output.see(tk.END)

def toggle_analysis_checks():
    """Toggles all analysis checks on/off based on the 'Run Analysis' checkbox."""
    if analysis_var.get():
        for var in analysis_checks_vars.values():
            var.set(True)
    else:
        for var in analysis_checks_vars.values():
            var.set(False)

def toggle_correction_checks():
    """Toggles all correction checks on/off based on the 'Run Correction' checkbox."""
    if correction_var.get():
        for var in correction_checks_vars.values():
            var.set(True)
    else:
        for var in correction_checks_vars.values():
            var.set(False)

def update_analysis_var(*args):
    """Updates the 'Run Analysis' checkbox based on the state of analysis checks."""
    if any(var.get() for var in analysis_checks_vars.values()):
        analysis_var.set(True)
    elif not all(var.get() for var in analysis_checks_vars.values()):
        analysis_var.set(False)

def update_correction_var(*args):
    """Updates the 'Run Correction' checkbox based on the state of correction checks."""
    if any(var.get() for var in correction_checks_vars.values()):
        correction_var.set(True)
    elif not all(var.get() for var in correction_checks_vars.values()):
        correction_var.set(False)

# --- GUI Setup ---
root = tk.Tk()
root.title("Data Analysis & Correction Dashboard")

# --- Style ---
style = ttk.Style()
style.configure("TButton", padding=5, font=('Arial', 10))
style.configure("TCheckbutton", padding=5, font=('Arial', 10))
style.configure("TLabel", padding=5, font=('Arial', 10))

# --- Input/Output Frames ---
input_frame = ttk.LabelFrame(root, text="Input File")
input_frame.pack(pady=10, padx=10, fill=tk.X)

input_entry = ttk.Entry(input_frame, width=50)
input_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

input_button = ttk.Button(input_frame, text="Browse", command=browse_input_file)
input_button.pack(side=tk.LEFT, padx=5, pady=5)

output_frame = ttk.LabelFrame(root, text="Output Directory")
output_frame.pack(pady=10, padx=10, fill=tk.X)

output_entry = ttk.Entry(output_frame, width=50)
output_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

output_button = ttk.Button(output_frame, text="Browse", command=browse_output_dir)
output_button.pack(side=tk.LEFT, padx=5, pady=5)

# --- Analysis/Correction Options ---
analysis_correction_frame = ttk.LabelFrame(root, text="Process Options")
analysis_correction_frame.pack(pady=10, padx=10, fill=tk.X)

analysis_var = tk.BooleanVar(value=True)
correction_var = tk.BooleanVar(value=True)

analysis_check = ttk.Checkbutton(analysis_correction_frame, text="Run Analysis", variable=analysis_var, command=toggle_analysis_checks)
analysis_check.pack(side=tk.LEFT, padx=5, pady=5)

correction_check = ttk.Checkbutton(analysis_correction_frame, text="Run Correction", variable=correction_var, command=toggle_correction_checks)
correction_check.pack(side=tk.LEFT, padx=5, pady=5)

# --- Checkbox Frames ---
analysis_checks_frame = ttk.LabelFrame(root, text="Analysis Checks")
analysis_checks_frame.pack(pady=10, padx=10, fill=tk.X)

correction_checks_frame = ttk.LabelFrame(root, text="Correction Checks")
correction_checks_frame.pack(pady=10, padx=10, fill=tk.X)

# --- Analysis Checks ---
analysis_checks_vars = {
    "Missing Values": tk.BooleanVar(value=True),
    "Duplicate Rows": tk.BooleanVar(value=True),
    "Outliers": tk.BooleanVar(value=True),
    "Formatting Errors": tk.BooleanVar(value=True),
    "Inconsistent Entries": tk.BooleanVar(value=True)
}

for check, var in analysis_checks_vars.items():
    check_button = ttk.Checkbutton(analysis_checks_frame, text=check, variable=var)
    check_button.pack(side=tk.LEFT, padx=5, pady=5)
    var.trace("w", update_analysis_var)

# --- Correction Checks ---
correction_checks_vars = {
    "Fix Column Spacing": tk.BooleanVar(value=True),
    "Format Hourly Rate": tk.BooleanVar(value=True),
    "Normalize Categorical Values": tk.BooleanVar(value=True),
    "Handle Missing Values": tk.BooleanVar(value=True),
    "Fix Numerical Values": tk.BooleanVar(value=True),
    "Fix Negative Values": tk.BooleanVar(value=True),
    "Handle Outliers": tk.BooleanVar(value=True),
    "Remove Duplicates": tk.BooleanVar(value=True)
}

for check, var in correction_checks_vars.items():
    check_button = ttk.Checkbutton(correction_checks_frame, text=check, variable=var)
    check_button.pack(side=tk.LEFT, padx=5, pady=5)
    var.trace("w", update_correction_var)

# --- Terminal Output ---
terminal_frame = ttk.LabelFrame(root, text="Terminal Output")
terminal_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

terminal_output = scrolledtext.ScrolledText(terminal_frame, wrap=tk.WORD, height=10)
terminal_output.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

terminal_var = tk.BooleanVar(value=True)
terminal_check = ttk.Checkbutton(root, text="View Terminal", variable=terminal_var)
terminal_check.pack(pady=5)

# --- Run Button ---
run_button = ttk.Button(root, text="Run Analysis & Correction", command=run_analysis_correction)
run_button.pack(pady=10)

root.mainloop()
