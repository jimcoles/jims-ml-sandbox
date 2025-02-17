#!/usr/bin/env python3
import sys
import os
import pandas as pd
from pandasgui import show


def main():
    # Check if arguments are provided
    if len(sys.argv) < 2:
        print("Usage: run-pg.py <file_path_or_command>")
        print("Example: run-pg.py data.csv")
        sys.exit(1)

    # Extract the argument (file path or data type params)
    arg = sys.argv[1]

    # Check if the argument is a valid CSV file
    if os.path.isfile(arg) and arg.endswith(".csv"):
        try:
            # Load the CSV file
            df = pd.read_csv(arg)
            print(f"Loaded dataset from {arg}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
    else:
        try:
            # If not a file, treat it as code to create a DataFrame directly
            print("No valid CSV provided. Creating a sample DataFrame...")
            df = pd.DataFrame({
                'Column1': [1, 2, 3],
                'Column2': [4, 5, 6],
                'Column3': [7, 8, 9]
            })
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            sys.exit(1)

    # Launch pandasgui to visualize the DataFrame
    try:
        show(df)
    except Exception as e:
        print(f"Error launching pandasgui: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()