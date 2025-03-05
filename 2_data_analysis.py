import pandas as pd
import numpy as np
import sys
import re
import enchant


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

def inspect_dataset(df, output_file="data_inspection.txt", duplicate_file="data_inspection_duplicates.txt", outlier_file="data_inspection_outliers.txt"):
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

if __name__ == "__main__":
    try:
        df = pd.read_csv("Employee Turnover Dataset.csv")
        inspect_dataset(df)
    except FileNotFoundError:
        print("Error: 'Employee Turnover Dataset.csv' not found.")
