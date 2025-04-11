import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_elexon_data(input_path='elexon_dataset.csv', output_path='../backend/app/data/processed_elexon_data.csv'):
    """
    Preprocess the Elexon energy dataset for use in the energy price prediction model.
    
    Parameters:
    -----------
    input_path : str
        Path to the raw CSV file
    output_path : str
        Path where the processed CSV will be saved
    """
    try:
        print(f"Loading dataset from: {input_path}")
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            return None
        
        # Get file size
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # For large files, use chunking to avoid memory issues
        chunk_size = 100000
        chunks = []
        
        # Try to read the first few rows to understand the structure
        sample_df = pd.read_csv(input_path, nrows=5)
        print("\nSample data:")
        print(sample_df.head())
        
        print("\nDetected columns:")
        print(sample_df.columns.tolist())
        
        # Process the data in chunks
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)):
            print(f"Processing chunk {i+1} with {len(chunk)} rows")
            chunks.append(chunk)
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').replace('.', '_')
        
        # Convert date/time columns to proper datetime format
        time_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in time_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Converted {col} to datetime")
            except Exception as e:
                print(f"Could not convert {col} to datetime: {str(e)}")
        
        # Print data types and missing values after conversion
        print("\nData types after conversion:")
        print(df.dtypes)
        
        print("\nMissing values:")
        missing_values = df.isna().sum()
        print(missing_values[missing_values > 0])  # Only show columns with missing values
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # For numeric columns, fill with median
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Filled {df[col].isna().sum()} missing values in {col} with median: {median_value}")
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_value)
                print(f"Filled {df[col].isna().sum()} missing values in {col} with mode: {mode_value}")
        
        # Identify price column(s)
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        if price_cols:
            print(f"Detected price columns: {price_cols}")
            # Use the first price column if multiple exist
            price_col = price_cols[0]
        else:
            print("Warning: No price column detected. Using first numeric column...")
            price_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        # Add derived features based on date/time if available
        date_col = time_columns[0] if time_columns else None
        if date_col and df[date_col].notna().any():
            print(f"Adding time-based features from {date_col}")
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['month'] = df[date_col].dt.month
            df['year'] = df[date_col].dt.year
            df['hour_of_day'] = df[date_col].dt.hour
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Remove outliers from price column if it exists
        if price_col:
            print(f"Removing outliers from {price_col}")
            # Store original count
            original_count = len(df)
            
            # Calculate bounds
            Q1 = df[price_col].quantile(0.25)
            Q3 = df[price_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter outliers
            df = df[(df[price_col] >= lower_bound) & (df[price_col] <= upper_bound)]
            
            # Report results
            removed_count = original_count - len(df)
            print(f"Removed {removed_count} outliers ({removed_count/original_count:.2%} of data)")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed dataset
        df.to_csv(output_path, index=False)
        print(f"\nProcessed dataset saved to: {output_path}")
        print(f"Final dataset shape: {df.shape}")
        
        return df
    
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_visualizations(df, output_dir='../backend/app/data/visualizations'):
    """
    Create visualizations of the processed data to understand patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed dataframe
    output_dir : str
        Directory where visualization images will be saved
    """
    try:
        if df is None or len(df) == 0:
            print("No data available for visualization")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Identify key columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        
        date_col = date_cols[0] if date_cols and df[date_cols[0]].dtype.kind == 'M' else None
        price_col = price_cols[0] if price_cols else None
        
        if not price_col:
            print("No price column found for visualization")
            return
            
        print(f"Creating visualizations using date column: {date_col} and price column: {price_col}")
        
        # Time series plot of energy prices
        if date_col and price_col:
            plt.figure(figsize=(15, 7))
            # Sort by date for proper time series
            temp_df = df.sort_values(by=date_col)
            plt.plot(temp_df[date_col], temp_df[price_col])
            plt.title(f'{price_col.title()} Over Time')
            plt.xlabel('Date')
            plt.ylabel(price_col.title())
            plt.tight_layout()
            plt.savefig(f"{output_dir}/price_time_series.png")
            plt.close()
            print(f"Saved time series plot to {output_dir}/price_time_series.png")
        
        # Distribution of energy prices
        if price_col:
            plt.figure(figsize=(12, 6))
            sns.histplot(df[price_col], kde=True)
            plt.title(f'Distribution of {price_col.title()}')
            plt.xlabel(price_col.title())
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/price_distribution.png")
            plt.close()
            print(f"Saved price distribution plot to {output_dir}/price_distribution.png")
        
        # Monthly average prices
        if 'month' in df.columns and price_col:
            monthly_avg = df.groupby('month')[price_col].mean().reset_index()
            plt.figure(figsize=(12, 6))
            sns.barplot(x='month', y=price_col, data=monthly_avg)
            plt.title(f'Average {price_col.title()} by Month')
            plt.xlabel('Month')
            plt.ylabel(f'Average {price_col.title()}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/monthly_avg_price.png")
            plt.close()
            print(f"Saved monthly average plot to {output_dir}/monthly_avg_price.png")
        
        # Day of week patterns
        if 'day_of_week' in df.columns and price_col:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_avg = df.groupby('day_of_week')[price_col].mean().reset_index()
            daily_avg['day_name'] = daily_avg['day_of_week'].apply(lambda x: day_names[x])
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='day_name', y=price_col, data=daily_avg)
            plt.title(f'Average {price_col.title()} by Day of Week')
            plt.xlabel('Day')
            plt.ylabel(f'Average {price_col.title()}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/day_of_week_price.png")
            plt.close()
            print(f"Saved day of week plot to {output_dir}/day_of_week_price.png")
        
        print(f"All visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Define the paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'elexon_dataset.csv')
    output_path = os.path.join(current_dir, '../backend/app/data/processed_elexon_data.csv')
    
    # Process the data
    processed_df = preprocess_elexon_data(input_path, output_path)
    
    # Create visualizations
    if processed_df is not None:
        visualizations_dir = os.path.join(current_dir, '../backend/app/data/visualizations')
        create_visualizations(processed_df, visualizations_dir) 