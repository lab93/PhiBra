import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("calculations.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data():
    """
    Load the weather, field properties, and soil measurements data.
    Assumes all CSV files are in the same directory as this script.
    """
    # Define file names
    weather_file = 'weather_data.csv'
    field_properties_file = 'field_properties.csv'
    soil_measurements_file = 'soil_measurements.csv'
    
    # Load weather_data.csv with flexible separator handling
    weather_data = None
    for sep in ['\t', ',', ';']:
        try:
            weather_data = pd.read_csv(weather_file, sep=sep, parse_dates=['Timestamp'])
            logging.info(f"Successfully read '{weather_file}' with separator '{sep}'.")
            break
        except ValueError as e:
            logging.warning(f"Failed to read '{weather_file}' with separator '{sep}': {e}")
    if weather_data is None:
        raise ValueError(f"Unable to read '{weather_file}'. Please check the file format and separators.")
    
    # Load field_properties.csv
    try:
        field_properties = pd.read_csv(field_properties_file)
        logging.info(f"Successfully read '{field_properties_file}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{field_properties_file}' does not exist.")
    except Exception as e:
        raise ValueError(f"Error reading '{field_properties_file}': {e}")
    
    # Load soil_measurements.csv with flexible separator handling
    soil_measurements = None
    for sep in ['\t', ',', ';']:
        try:
            soil_measurements = pd.read_csv(soil_measurements_file, sep=sep, parse_dates=['Timestamp'])
            logging.info(f"Successfully read '{soil_measurements_file}' with separator '{sep}'.")
            break
        except ValueError as e:
            logging.warning(f"Failed to read '{soil_measurements_file}' with separator '{sep}': {e}")
    if soil_measurements is None:
        raise ValueError(f"Unable to read '{soil_measurements_file}'. Please check the file format and separators.")
    
    # Clean column names: strip whitespace and replace special characters
    # Also remove double quotes
    field_properties.columns = (
        field_properties.columns
        .str.strip()
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace('/', '_', regex=False)
        .str.replace('-', '_', regex=False)
        .str.replace('"', '', regex=False)  # Remove double quotes
    )
    weather_data.columns = (
        weather_data.columns
        .str.strip()
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace('/', '_', regex=False)
        .str.replace('-', '_', regex=False)
        .str.replace('"', '', regex=False)  # Remove double quotes
    )
    soil_measurements.columns = (
        soil_measurements.columns
        .str.strip()
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace('/', '_', regex=False)
        .str.replace('-', '_', regex=False)
        .str.replace('"', '', regex=False)  # Remove double quotes
    )
    
    # Verify 'Timestamp' columns are correctly parsed
    for df, name in zip([weather_data, soil_measurements], ['weather_data', 'soil_measurements']):
        if 'Timestamp' not in df.columns:
            raise ValueError(f"'Timestamp' column is missing in '{name}.csv'. Please check the file.")
        if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            if df['Timestamp'].isnull().any():
                raise ValueError(f"Some 'Timestamp' entries in '{name}.csv' could not be parsed. Please check the date formats.")
            logging.info(f"'Timestamp' column in '{name}.csv' successfully parsed as datetime.")
    
    # Display column names for verification
    logging.info("Weather Data Columns: %s", weather_data.columns.tolist())
    logging.info("Field Properties Columns: %s", field_properties.columns.tolist())
    logging.info("Soil Measurements Columns: %s", soil_measurements.columns.tolist())
    
    return field_properties, weather_data, soil_measurements

def preprocess_data(weather_data, soil_measurements):
    """
    Preprocess the weather data and soil measurements:
    - Set 'Timestamp' as index.
    - Resample soil measurements to daily averages.
    """
    # Set Timestamp as index for soil_measurements
    soil_measurements.set_index('Timestamp', inplace=True)
    weather_data.set_index('Timestamp', inplace=True)
    
    # Resample soil measurements to daily averages
    soil_measurements_daily = soil_measurements.resample('D').mean()
    
    # Reset index to have Timestamp as a column again
    soil_measurements_daily.reset_index(inplace=True)
    weather_data.reset_index(inplace=True)
    
    logging.info("Data preprocessing completed: Resampled soil measurements to daily averages.")
    return weather_data, soil_measurements_daily

def calculate_water_stress(weather_data, soil_measurements_daily, field_properties):
    """
    Calculate Water Stress Index (WSI) based on soil moisture and field capacity.
    """
    # Define root zone depths after cleaning (e.g., 'M4', 'M8', etc.)
    root_zone_depths = ['M4', 'M8', 'M12', 'M16', 'M20', 'M24']
    
    # Identify soil moisture columns (e.g., 'M4', 'M8', etc.)
    moisture_columns = [col for col in soil_measurements_daily.columns if col in root_zone_depths]
    logging.info(f"Identified moisture columns: {moisture_columns}")
    
    if not moisture_columns:
        logging.warning("No root zone soil moisture columns found in the data.")
        soil_moisture = pd.Series([np.nan] * len(soil_measurements_daily), index=soil_measurements_daily.index)
        # Define field_capacity to a default value
        field_capacity = 0.20
    else:
        # Calculate volumetric water content using soil moisture percentages and field capacity
        # Assume field capacity based on soil texture from field_properties
        if 'Textural_Class' in field_properties.columns:
            soil_texture = field_properties['Textural_Class'].iloc[0]
            field_capacity = estimate_field_capacity(soil_texture)
            logging.info(f"Estimated field capacity based on soil texture '{soil_texture}': {field_capacity}")
        else:
            logging.warning("'Textural_Class' not found in field_properties. Using default field capacity of 0.20.")
            field_capacity = 0.20  # Default value
        
        # Ensure moisture_columns contain numeric data
        soil_measurements_daily[moisture_columns] = soil_measurements_daily[moisture_columns].apply(pd.to_numeric, errors='coerce')
        
        # Check for NaNs after conversion
        if soil_measurements_daily[moisture_columns].isnull().any().any():
            logging.warning("NaN values found in soil moisture columns after conversion. These will affect WSI calculations.")
        
        soil_moisture_percent = soil_measurements_daily[moisture_columns].mean(axis=1)
        soil_moisture = soil_moisture_percent / 100 * field_capacity
        logging.debug(f"Soil moisture (volumetric water content):\n{soil_moisture.head()}")
    
    # Calculate water stress index (WSI)
    permanent_wilting_point = field_capacity * 0.5  # Simplified assumption
    denominator = field_capacity - permanent_wilting_point
    logging.debug(f"Field Capacity: {field_capacity}, Permanent Wilting Point: {permanent_wilting_point}, Denominator: {denominator}")
    
    # Remove the problematic line below
    # denominator = denominator.replace(0, np.nan)  # Avoid division by zero
    
    # Since denominator is a scalar and unlikely to be zero, proceed without replacement
    if denominator == 0:
        logging.error("Denominator is zero in WSI calculation. Setting WSI to NaN to avoid division by zero.")
        wsi = pd.Series([np.nan] * len(soil_measurements_daily), index=soil_measurements_daily.index)
    else:
        wsi = (field_capacity - soil_moisture) / denominator
        wsi = wsi.clip(0, 1)  # Ensure WSI is between 0 and 1
        logging.debug(f"Water Stress Index (WSI):\n{wsi.head()}")
    
    # Merge WSI with soil_measurements_daily
    water_stress_data = pd.DataFrame({
        'Timestamp': soil_measurements_daily['Timestamp'],
        'Water_Stress_Index': wsi
    })
    
    # Check if WSI has valid values
    if water_stress_data['Water_Stress_Index'].isnull().all():
        logging.error("All WSI values are NaN. Check soil moisture data and field capacity calculations.")
    else:
        logging.info("Calculated Water Stress Index.")
    
    return water_stress_data

def estimate_field_capacity(soil_texture):
    """
    Estimate field capacity based on soil texture.
    """
    # Estimate field capacity based on soil texture
    texture_field_capacity = {
        'Sand': 0.10,
        'Loamy Sand': 0.12,
        'Sandy Loam': 0.15,
        'Loam': 0.20,
        'Silt Loam': 0.25,
        'Silt': 0.30,
        'Sandy Clay Loam': 0.20,
        'Clay Loam': 0.25,
        'Silty Clay Loam': 0.30,
        'Sandy Clay': 0.25,
        'Silty Clay': 0.35,
        'Clay': 0.40
    }
    field_capacity = texture_field_capacity.get(soil_texture, 0.20)  # Default to 0.20 if texture not found
    logging.info(f"Field capacity for soil texture '{soil_texture}': {field_capacity}")
    return field_capacity

def calculate_irrigation_needs(weather_data, soil_measurements_daily, field_properties):
    """
    Calculate irrigation requirements based on evapotranspiration (ET) and crop coefficients.
    """
    # Advanced irrigation scheduling using ET and crop coefficients
    # Assume crop coefficient (Kc) is provided or use typical values based on growth stage
    crop_coefficients = estimate_crop_coefficients(weather_data)
    logging.info("Estimated crop coefficients based on accumulated GDD.")
    
    # Calculate ETc (Crop Evapotranspiration)
    if 'Arable_Field_Evapotranspiration_mm' in weather_data.columns:
        eto = weather_data['Arable_Field_Evapotranspiration_mm']
        logging.info("Using 'Arable_Field_Evapotranspiration_mm' for ET calculation.")
    else:
        eto = pd.Series([np.nan] * len(weather_data), index=weather_data.index)
        logging.warning("'Arable_Field_Evapotranspiration_mm' not found. Using NaN for ETc.")
    
    etc = eto * crop_coefficients
    logging.info("Calculated Crop Evapotranspiration (ETc).")
    
    # Calculate net irrigation requirement
    if 'Precipitation_mm' in weather_data.columns:
        precipitation_mm = weather_data['Precipitation_mm']
        logging.info("Using 'Precipitation_mm' for irrigation calculation.")
    else:
        precipitation_mm = pd.Series([0] * len(weather_data), index=weather_data.index)
        logging.warning("'Precipitation_mm' not found. Assuming 0 mm precipitation.")
    
    net_irrigation = etc - precipitation_mm
    net_irrigation = net_irrigation.clip(lower=0)  # No negative irrigation
    
    # Adjust for irrigation efficiency
    irrigation_efficiency = 0.8  # Assume 80% efficiency
    gross_irrigation = net_irrigation / irrigation_efficiency
    gross_irrigation = gross_irrigation.clip(lower=0)
    
    # Calculate cumulative irrigation need
    cumulative_irrigation = gross_irrigation.cumsum()
    
    # Calculate cost of irrigation
    # Assume cost per mm of irrigation is based on energy and water costs
    cost_per_mm = 0.10  # Adjust as necessary
    irrigation_cost = gross_irrigation * cost_per_mm
    
    irrigation_data = pd.DataFrame({
        'Timestamp': weather_data['Timestamp'],
        'Net_Irrigation_Requirement_mm': net_irrigation,
        'Gross_Irrigation_Requirement_mm': gross_irrigation,
        'Cumulative_Irrigation_mm': cumulative_irrigation,
        'Irrigation_Cost_$': irrigation_cost
    })
    
    # Check if irrigation data has valid values
    if irrigation_data['Gross_Irrigation_Requirement_mm'].isnull().all():
        logging.error("All Gross Irrigation Requirement values are NaN. Check ETc and precipitation data.")
    else:
        logging.info("Calculated Irrigation Needs.")
    
    return irrigation_data

def estimate_crop_coefficients(weather_data):
    """
    Estimate crop coefficient (Kc) based on accumulated Growing Degree Days (GDD) and growth stages.
    """
    # Estimate crop coefficient (Kc) based on accumulated GDD and crop growth stages
    if 'Accumulated_Growing_Degree_Days' in weather_data.columns:
        accumulated_gdd = weather_data['Accumulated_Growing_Degree_Days']
        logging.info("Using 'Accumulated_Growing_Degree_Days' for Kc estimation.")
    else:
        accumulated_gdd = pd.Series([np.nan] * len(weather_data), index=weather_data.index)
        logging.warning("'Accumulated_Growing_Degree_Days' not found. Using default Kc = 0.5.")
    
    # Define crop growth stages and corresponding Kc values (example for corn)
    growth_stages = {
        0: 0.3,    # Planting
        200: 0.5,  # Early growth
        500: 0.8,  # Mid-season
        800: 1.15, # Peak growth
        1100: 0.9, # Late season
        1400: 0.6  # Maturity
    }
    
    # Sort the GDD values
    gdd_values = sorted(growth_stages.keys())
    kc_values = [growth_stages[gdd] for gdd in gdd_values]
    
    # Interpolate Kc values based on GDD
    # Replace NaN in accumulated_gdd with the last valid value to avoid NaNs in interpolation
    accumulated_gdd_filled = accumulated_gdd.ffill().fillna(0)  # Changed from fillna(method='ffill')
    crop_coefficients = np.interp(accumulated_gdd_filled, gdd_values, kc_values)
    
    # Handle any remaining NaN values if 'Accumulated_Growing_Degree_Days' was missing
    crop_coefficients = np.where(np.isnan(crop_coefficients), 0.5, crop_coefficients)  # Default Kc
    
    logging.debug(f"Crop Coefficients Sample:\n{crop_coefficients[:5]}")
    return crop_coefficients

def calculate_disease_risk(weather_data):
    """
    Calculate disease risk based on adjusted thresholds and add a 'Disease_Risk' column.
    """
    # Define the correct column names after cleaning
    min_rel_humidity_col = 'Minimum_Relative_Humidity'
    mean_temp_col = 'Mean_Temp'
    leaf_wetness_col = 'Leaf_Wetness_Hours'
    
    # Check if necessary columns are available
    required_columns = [min_rel_humidity_col, mean_temp_col, leaf_wetness_col]
    missing_columns = [col for col in required_columns if col not in weather_data.columns]
    if missing_columns:
        logging.warning(f"Missing columns for disease risk calculation: {missing_columns}")
        disease_risk_data = pd.DataFrame({
            'Timestamp': weather_data['Timestamp'],
            'Disease_Risk': 'Data Not Available'
        })
    else:
        # Adjusted thresholds for less stringent classification
        high_risk_conditions = (
            (weather_data[min_rel_humidity_col] >= 75) &
            (weather_data[leaf_wetness_col] >= 6) &
            (weather_data[mean_temp_col] >= 60) &
            (weather_data[mean_temp_col] <= 90)
        )
        
        moderate_risk_conditions = (
            (weather_data[min_rel_humidity_col] >= 65) &
            (weather_data[leaf_wetness_col] >= 4) &
            (weather_data[mean_temp_col] >= 50) &
            (weather_data[mean_temp_col] <= 95)
        )
        
        # Initialize Disease Risk as 'Low Risk'
        disease_risk = pd.Series('Low Risk', index=weather_data.index)
        disease_risk[moderate_risk_conditions] = 'Moderate Risk'
        disease_risk[high_risk_conditions] = 'High Risk'
        
        # Add debugging statements
        high_risk_entries = weather_data.loc[high_risk_conditions, [min_rel_humidity_col, leaf_wetness_col, mean_temp_col]]
        moderate_risk_entries = weather_data.loc[moderate_risk_conditions, [min_rel_humidity_col, leaf_wetness_col, mean_temp_col]]
        
        logging.info(f"High Risk Conditions Met: {len(high_risk_entries)} entries")
        logging.debug(f"High Risk Entries:\n{high_risk_entries.head()}")
        
        logging.info(f"Moderate Risk Conditions Met: {len(moderate_risk_entries)} entries")
        logging.debug(f"Moderate Risk Entries:\n{moderate_risk_entries.head()}")
        
        # Create Disease Risk DataFrame
        disease_risk_data = pd.DataFrame({
            'Timestamp': weather_data['Timestamp'],
            'Minimum_Relative_Humidity': weather_data[min_rel_humidity_col],
            'Mean_Temp': weather_data[mean_temp_col],
            'Leaf_Wetness_Hours': weather_data[leaf_wetness_col],
            'Disease_Risk': disease_risk
        })
        
        logging.info("Calculated Disease Risk.")
    
    return disease_risk_data

def soil_health_assessment(soil_measurements_daily, field_properties):
    """
    Assess soil health based on Electrical Conductivity (EC) and soil pH.
    """
    # Define EC root zone depths without 'M' prefix
    ec_depths = [4, 8, 12, 16, 20, 24]
    
    # Correct EC column naming to match cleaned column names (e.g., 'EC4')
    ec_columns = [f'EC{depth}' for depth in ec_depths]
    
    # Ensure columns exist in soil_measurements_daily
    ec_columns_existing = [col for col in ec_columns if col in soil_measurements_daily.columns]
    logging.info(f"Identified EC columns: {ec_columns_existing}")
    
    # Average EC profile
    if ec_columns_existing:
        avg_ec = soil_measurements_daily[ec_columns_existing].mean()
        ec_profile = pd.DataFrame({'Depth': ec_columns_existing, 'EC': avg_ec.values})
        ec_profile['Depth'] = ec_profile['Depth'].str.replace('EC', '', regex=False)  # Extract depth part
        
        # Interpret EC levels
        ec_profile['Salinity_Interpretation'] = ec_profile['EC'].apply(interpret_salinity)
        logging.info("Assessed Soil Health based on EC.")
        logging.debug(f"EC Profile Sample:\n{ec_profile.head()}")
    else:
        logging.warning("No EC data available. Please check the column names in 'soil_measurements_daily.csv'.")
        ec_profile = pd.DataFrame(columns=['Depth', 'EC', 'Salinity_Interpretation'])
    
    # Include soil pH from field_properties
    if 'Soil_pH' in field_properties.columns:
        soil_pH = field_properties['Soil_pH'].iloc[0]
        logging.info(f"Soil pH: {soil_pH}")
    else:
        soil_pH = np.nan
        logging.warning("Soil pH data not available.")
    
    # Add soil pH to the EC profile DataFrame
    ec_profile['Soil_pH'] = soil_pH
    
    return ec_profile

def interpret_salinity(ec_value):
    """
    Interpret salinity levels based on EC values.
    """
    if pd.isna(ec_value):
        return 'No data'
    elif ec_value < 1.7:
        return 'No yield reduction expected'
    elif 1.7 <= ec_value <= 3.8:
        return 'Mild to moderate yield reduction'
    else:
        return 'Severe yield reduction'

def growth_stage_monitoring(weather_data):
    """
    Monitor crop growth stages based on accumulated Growing Degree Days (GDD).
    """
    # Map accumulated GDD to growth stages for corn (example)
    if 'Accumulated_Growing_Degree_Days' in weather_data.columns:
        accumulated_gdd = weather_data['Accumulated_Growing_Degree_Days']
        logging.info("Using 'Accumulated_Growing_Degree_Days' for Growth Stage monitoring.")
    else:
        accumulated_gdd = pd.Series([np.nan] * len(weather_data), index=weather_data.index)
        logging.warning("'Accumulated_Growing_Degree_Days' not found. Growth Stage data will be NaN.")
    
    # Define growth stages based on GDD
    growth_stages = {
        0: 'Emergence',
        200: 'Vegetative',
        500: 'Tasseling',
        800: 'Silking',
        1100: 'Dough',
        1400: 'Maturity'
    }
    
    # Sort the GDD values
    gdd_values = sorted(growth_stages.keys())
    stage_names = [growth_stages[gdd] for gdd in gdd_values]
    
    # Interpolate Growth Stage based on GDD
    growth_stage = pd.cut(
        accumulated_gdd,
        bins=gdd_values + [float('inf')],
        labels=stage_names,
        right=False,
        include_lowest=True
    )
    
    growth_stage_data = pd.DataFrame({
        'Timestamp': weather_data['Timestamp'],
        'Accumulated_GDD': accumulated_gdd,
        'Growth_Stage': growth_stage
    })
    
    logging.info("Monitored Growth Stages based on GDD.")
    logging.debug(f"Growth Stage Data Sample:\n{growth_stage_data.head()}")
    return growth_stage_data

def microclimate_analysis(weather_data):
    """
    Analyze microclimate conditions and provide recommendations.
    """
    # Define the required columns based on cleaned column names
    required_columns = [
        'Mean_Temp',
        'Minimum_Relative_Humidity',
        'Wind_Speed',
        'Shortwave_Downwelling_Radiation'  # Corrected column name
    ]
    
    # Verify and adjust column names
    available_columns = weather_data.columns.tolist()
    for col in required_columns:
        if col not in available_columns:
            # Attempt to find a similar column
            similar_cols = [c for c in available_columns if col.lower() in c.lower()]
            if similar_cols:
                logging.info(f"Adjusted column '{col}' to '{similar_cols[0]}' based on similarity.")
                required_columns[required_columns.index(col)] = similar_cols[0]
            else:
                logging.warning(f"Required column '{col}' not found in weather_data.")
                weather_data[col] = np.nan  # Fill with NaN if not found
    
    # Extract necessary data and include 'Timestamp'
    try:
        microclimate_data = weather_data[['Timestamp'] + required_columns].copy()
        logging.debug(f"Microclimate Data Columns: {microclimate_data.columns.tolist()}")
    except KeyError as e:
        logging.error(f"Missing columns for microclimate analysis: {e}")
        microclimate_data = pd.DataFrame()
    
    # Provide recommendations based on thresholds
    recommendations = []
    
    for index, row in microclimate_data.iterrows():
        recommendation = ''
        
        # High temperature stress
        if pd.notna(row['Mean_Temp']) and row['Mean_Temp'] > 95:
            recommendation += 'High temperature stress. Consider irrigation or shading techniques. '
        
        # Low humidity
        if pd.notna(row['Minimum_Relative_Humidity']) and row['Minimum_Relative_Humidity'] < 30:
            recommendation += 'Low humidity may increase evapotranspiration. Monitor soil moisture closely. '
        
        # High wind speed
        if pd.notna(row['Wind_Speed']) and row['Wind_Speed'] > 15:
            recommendation += 'High winds detected. Secure loose equipment and monitor for wind damage. '
        
        # High solar radiation
        if pd.notna(row['Shortwave_Downwelling_Radiation']) and row['Shortwave_Downwelling_Radiation'] > 800:
            recommendation += 'High solar radiation may increase water demand. '
        
        recommendations.append(recommendation.strip())
    
    microclimate_data['Recommendations'] = recommendations
    
    # Check if recommendations have been added
    if microclimate_data['Recommendations'].isnull().all():
        logging.error("All Recommendations are empty. Check if required columns have valid data.")
    else:
        logging.info("Completed Microclimate Analysis.")
    
    # Log a sample of the microclimate data
    logging.debug(f"Microclimate Data Sample:\n{microclimate_data.head()}")
    
    return microclimate_data

def generate_trends(df, timestamp_col, value_cols, output_dir, dataset_name):
    """
    Generates trend plots for specified value columns over time.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        timestamp_col (str): The name of the timestamp column.
        value_cols (list): List of column names to plot.
        output_dir (str): Directory to save the plots.
        dataset_name (str): Name of the dataset (for logging purposes).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if df is None or df.empty:
        logging.warning(f"'{dataset_name}' DataFrame is None or empty. Skipping trend generation.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory '{output_dir}' for saving plots.")
    
    for col in value_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=timestamp_col, y=col, data=df)
            plt.title(f"Trend of {col} over Time for {dataset_name}")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{dataset_name}_{col}_trend.png")
            try:
                plt.savefig(plot_file)
                plt.close()
                logging.info(f"Saved trend plot for '{col}' in '{dataset_name}' to '{plot_file}'.")
            except Exception as e:
                logging.error(f"Error saving plot '{plot_file}': {e}")
        else:
            logging.warning(f"Column '{col}' not found or not numeric in '{dataset_name}'. Skipping plot.")

def save_processed_data(
    water_stress_data,
    irrigation_data,
    ec_profile,
    growth_stage_data,
    microclimate_data,
    disease_risk_data
):
    """
    Save all processed data to CSV files in the current directory.
    """
    # Define output file names
    water_stress_file = 'water_stress_data.csv'
    irrigation_file = 'irrigation_data.csv'
    ec_profile_file = 'ec_profile.csv'
    growth_stage_file = 'growth_stage_data.csv'
    microclimate_file = 'microclimate_data.csv'
    disease_risk_file = 'disease_risk_data.csv'
    
    # Save to CSV with error handling
    for df, filename in zip(
        [water_stress_data, irrigation_data, ec_profile, growth_stage_data, microclimate_data, disease_risk_data],
        [water_stress_file, irrigation_file, ec_profile_file, growth_stage_file, microclimate_file, disease_risk_file]
    ):
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Successfully saved '{filename}'.")
        except Exception as e:
            logging.error(f"Error saving '{filename}': {e}")
    
    logging.info("All processed data saved successfully.")

def main():
    """
    Main function to execute the data loading, cleaning, preprocessing, calculations, and saving.
    """
    try:
        # Load data
        field_properties, weather_data, soil_measurements = load_data()
        
        # Preprocess data
        weather_data, soil_measurements_daily = preprocess_data(weather_data, soil_measurements)
        
        # Convert Precipitation from inches to millimeters
        if 'Precipitation' in weather_data.columns:
            weather_data['Precipitation_mm'] = weather_data['Precipitation'] * 25.4
            logging.info("Converted 'Precipitation' from inches to millimeters and created 'Precipitation_mm' column.")
        else:
            # Handle cases where 'Precipitation_mm' is already present or 'Precipitation' is missing
            if 'Precipitation_mm' not in weather_data.columns:
                weather_data['Precipitation_mm'] = 0
                logging.warning("'Precipitation' column not found. Initialized 'Precipitation_mm' with 0 mm.")
            else:
                logging.info("'Precipitation_mm' column already exists in weather_data.")
        
        # Calculate water stress
        water_stress_data = calculate_water_stress(weather_data, soil_measurements_daily, field_properties)
        
        # Calculate irrigation needs
        irrigation_data = calculate_irrigation_needs(weather_data, soil_measurements_daily, field_properties)
        
        # Soil health assessment
        ec_profile = soil_health_assessment(soil_measurements_daily, field_properties)
        
        # Growth stage monitoring
        growth_stage_data = growth_stage_monitoring(weather_data)
        
        # Microclimate analysis
        microclimate_data = microclimate_analysis(weather_data)
        
        # Calculate disease risk
        disease_risk_data = calculate_disease_risk(weather_data)
        
        # Save all processed data
        save_processed_data(
            water_stress_data,
            irrigation_data,
            ec_profile,
            growth_stage_data,
            microclimate_data,
            disease_risk_data
        )
        
        # Generate trend plots (optional)
        # Define output directories
        output_plots_dir = 'plots'
        
        # Define timestamp and value columns for each dataset
        soil_timestamp_col = 'Timestamp'
        soil_value_cols = ['Water_Stress_Index']  # Only plotting WSI for soil data
        
        micro_timestamp_col = 'Timestamp'
        micro_value_cols = ['Mean_Temp', 'Minimum_Relative_Humidity', 'Wind_Speed', 'Shortwave_Downwelling_Radiation']
        
        # Generate plots
        generate_trends(water_stress_data, soil_timestamp_col, soil_value_cols, output_plots_dir, 'Water_Stress')
        generate_trends(microclimate_data, micro_timestamp_col, micro_value_cols, output_plots_dir, 'Microclimate_Analysis')
        
        # Print summaries for verification
        logging.info("\nWater Stress Data Summary:")
        logging.info("%s", water_stress_data.describe())
        
        logging.info("\nIrrigation Data Summary:")
        logging.info("%s", irrigation_data.describe())
        
        logging.info("\nEC Profile Summary:")
        logging.info("%s", ec_profile.head())
        
        logging.info("\nGrowth Stage Data Summary:")
        logging.info("%s", growth_stage_data.head())
        
        logging.info("\nMicroclimate Data Summary:")
        logging.info("%s", microclimate_data.head())
        
        logging.info("\nDisease Risk Data Summary:")
        logging.info("%s", disease_risk_data['Disease_Risk'].value_counts())
        
        logging.info("All calculations completed successfully.")
        
    except Exception as e:
        logging.error("An error occurred during calculations:", exc_info=True)

if __name__ == '__main__':
    main()
