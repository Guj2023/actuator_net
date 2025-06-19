#
#   Extracts the data used for actuator net training from a given .mat file for experiments.
#   From .mat to .pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def extract_data(
    datapath, data_name, variable_name=None, data_start=None, data_end=None
):
    """
    Extracts data from a .mat file for actuator net training.

    Parameters
    ----------
    datapath : str
        Path to the directory containing the .mat file.
    data_name : str
        Name of the .mat file (with or without .mat extension).
    variable_name : str or list, optional
        Name(s) of the variable(s) to extract from the .mat file.
        If None, all numeric arrays will be extracted.
    data_start : int, optional
        Starting index for valid data extraction. If None, all data is used.
    data_end : int, optional
        Ending index for valid data extraction. If None, all data is used.

    Returns
    -------
    dict
        A dictionary containing the extracted data.
    """
    # Ensure the data_name has .mat extension
    if not data_name.endswith(".mat"):
        data_name += ".mat"

    # Load the .mat file
    file_path = os.path.join(datapath, data_name)
    try:
        data = sio.loadmat(file_path)
        print(f"Successfully loaded file: {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

    # Extract specified variables or all numeric arrays if none specified
    if variable_name is None:
        extracted_data = {
            k: v
            for k, v in data.items()
            if not k.startswith("__") and isinstance(v, np.ndarray)
        }
    else:
        if isinstance(variable_name, str):
            variable_name = [variable_name]
        extracted_data = {}
        for var in variable_name:
            if var in data:
                extracted_data[var] = data[var]
            else:
                print(f"Warning: Variable '{var}' not found in the data file.")

    # Apply data start/end filtering if specified
    if data_start is not None or data_end is not None:
        for key in extracted_data:
            if (
                extracted_data[key].ndim >= 1
            ):  # Check if the array has at least one dimension
                start_idx = data_start if data_start is not None else 0
                end_idx = (
                    data_end if data_end is not None else extracted_data[key].shape[0]
                )
                extracted_data[key] = extracted_data[key][start_idx:end_idx]

    return extracted_data


def process_for_training(
    datapath,
    data_name,
    output_dir=None,
    data_start=None,
    data_end=None,
    scaler="standard",
):
    """
    Process .mat file data for training an actuator network.
    Extracts force feedback, force command, and position data,
    creates features, and saves to pickle format.

    Parameters
    ----------
    datapath : str
        Path to the directory containing the .mat file.
    data_name : str
        Name of the .mat file (with or without .mat extension).
    output_dir : str, optional
        Directory to save the processed data. If None, uses the input directory.
    data_start : int, optional
        Starting index for valid data extraction.
    data_end : int, optional
        Ending index for valid data extraction.
    scaler : str, default="standard"
        Type of scaling to apply to features. Options: "standard", "minmax", "robust", "none".

    Returns
    -------
    dict
        A dictionary containing the processed data.
    """
    # Set output directory
    if output_dir is None:
        output_dir = datapath

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define required variables for extraction
    required_vars = ["Loa2", "F2_des", "q2_act"]

    # Extract data
    data = extract_data(datapath, data_name, required_vars, data_start, data_end)
    if data is None:
        return None

    # Process the data
    processed_data = {}

    # Check if all required variables are present
    if not all(var in data for var in required_vars):
        missing = [var for var in required_vars if var not in data]
        print(f"Error: Missing required variables: {missing}")
        return None

    # Process force feedback and command
    force_feedback = np.squeeze(data["Loa2"])
    force_command = np.squeeze(data["F2_des"])

    # Process position data (convert from degrees to radians if needed)
    position = np.squeeze(data["q2_act"])
    position_rad = np.radians(position)  # Convert to radians

    # Process force data
    force_desired = np.squeeze(data["F2_des"])  # Desired force
    force_feedback = np.squeeze(data["Loa2"])  # Actual force feedback

    # Create time-delayed features for force desired
    force_last_desired = np.zeros_like(force_desired)
    force_last_desired[1:] = force_desired[:-1]

    force_last_last_desired = np.zeros_like(force_desired)
    force_last_last_desired[2:] = force_desired[:-2]

    # Create time-delayed features for position
    position_last = np.zeros_like(position_rad)
    position_last[1:] = position_rad[:-1]

    position_last_last = np.zeros_like(position_rad)
    position_last_last[2:] = position_rad[:-2]

    # Combine features according to requirements:
    # force_desired, force_last_desired, force_last_last_desired,
    # position, position_last, position_last_last
    features = np.column_stack(
        [
            force_desired,  # Current desired force
            force_last_desired,  # Desired force from previous timestep
            force_last_last_desired,  # Desired force from two timesteps ago
            position_rad,  # Current position (in radians)
            position_last,  # Position from previous timestep
            position_last_last,  # Position from two timesteps ago
        ]
    )

    # Create label (output to predict)
    labels = force_feedback.reshape(-1, 1)

    # Scale features
    if scaler.lower() != "none":
        scaler_obj = None
        if scaler.lower() == "standard":
            scaler_obj = StandardScaler()
        elif scaler.lower() == "minmax":
            scaler_obj = MinMaxScaler()
        elif scaler.lower() == "robust":
            scaler_obj = RobustScaler()

        if scaler_obj is not None:
            scaled_features = scaler_obj.fit_transform(
                features
            )  # Save scaler for later use
            scaler_dict = {"use_scale": True, "scaler": scaler_obj}
            scaler_file = os.path.join(output_dir, "scaler.pkl")
            save_numpy_compatible_pickle(scaler_dict, scaler_file)

            print(f"Scaler saved to: {scaler_file}")
        else:
            scaled_features = features
            print("No scaling applied to features.")
    else:
        scaled_features = features
        scaler_dict = {"use_scale": False, "scaler": None}
        scaler_file = os.path.join(output_dir, "scaler.pkl")
        save_numpy_compatible_pickle(scaler_dict, scaler_file)

    # Prepare final data structure
    processed_data = {
        "input_data": features,  # Original unscaled features
        "scaled_input_data": scaled_features,  # Scaled features
        "output_data": labels,  # Labels (force feedback)
        "force_command": force_command.reshape(-1, 1),  # Force command
        "position": position_rad.reshape(-1, 1),  # Position in radians
    }

    # Save processed data
    output_file = os.path.join(
        output_dir, f"{os.path.splitext(data_name)[0]}_processed.pkl"
    )
    save_numpy_compatible_pickle(processed_data, output_file)

    print(f"Processed data saved to: {output_file}")
    print(f"Input data shape: {features.shape}")
    print(f"Output data shape: {labels.shape}")

    return processed_data


def process_multiple_files(datapath, file_config, output_dir=None, scaler="standard"):
    """
    Process multiple .mat files from the same directory for training.
    Each file can have its own data_start and data_end parameters.

    Parameters
    ----------
    datapath : str
        Path to the directory containing the .mat files.
    file_config : list of dict
        List of configuration dictionaries for each file.
        Each dictionary should contain:
        - 'data_name': Name of the .mat file
        - 'data_start': (optional) Starting index for valid data
        - 'data_end': (optional) Ending index for valid data
    output_dir : str, optional
        Directory to save the processed data. If None, uses the input directory.
    scaler : str, default="standard"
        Type of scaling to apply to features. Options: "standard", "minmax", "robust", "none".

    Returns
    -------
    dict
        A dictionary containing the combined processed data.
    """
    # Set output directory
    if output_dir is None:
        output_dir = datapath

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define required variables for extraction
    required_vars = ["Loa2", "F2_des", "q2_act"]

    # Initialize lists to store combined data
    combined_features = []
    combined_labels = []
    combined_force_command = []
    combined_position = []

    # Process each file
    for config in file_config:
        data_name = config["data_name"]
        data_start = config.get("data_start", None)
        data_end = config.get("data_end", None)

        print(f"\nProcessing file: {data_name}")
        print(
            f"Data range: {data_start if data_start is not None else 'start'} to "
            f"{data_end if data_end is not None else 'end'}"
        )

        # Extract data for this file
        data = extract_data(datapath, data_name, required_vars, data_start, data_end)
        if data is None:
            print(f"Skipping file {data_name} due to extraction errors.")
            continue

        # Check if all required variables are present
        if not all(var in data for var in required_vars):
            missing = [var for var in required_vars if var not in data]
            print(f"Skipping file {data_name}. Missing required variables: {missing}")
            continue

        # Process force feedback and command
        force_feedback = np.squeeze(data["Loa2"])
        force_desired = np.squeeze(data["F2_des"])

        # Process position data (convert from degrees to radians if needed)
        position = np.squeeze(data["q2_act"])
        position_rad = np.radians(position)  # Convert to radians

        # Create time-delayed features for force desired
        force_last_desired = np.zeros_like(force_desired)
        force_last_desired[1:] = force_desired[:-1]

        force_last_last_desired = np.zeros_like(force_desired)
        force_last_last_desired[2:] = force_desired[:-2]

        # Create time-delayed features for position
        position_last = np.zeros_like(position_rad)
        position_last[1:] = position_rad[:-1]

        position_last_last = np.zeros_like(position_rad)
        position_last_last[2:] = position_rad[:-2]

        # Combine features
        features = np.column_stack(
            [
                force_desired,  # Current desired force
                force_last_desired,  # Desired force from previous timestep
                force_last_last_desired,  # Desired force from two timesteps ago
                position_rad,  # Current position (in radians)
                position_last,  # Position from previous timestep
                position_last_last,  # Position from two timesteps ago
            ]
        )

        # Create label (output to predict)
        labels = force_feedback.reshape(-1, 1)

        # Append to combined data
        combined_features.append(features)
        combined_labels.append(labels)
        combined_force_command.append(force_desired.reshape(-1, 1))
        combined_position.append(position_rad.reshape(-1, 1))

        print(f"Extracted {features.shape[0]} samples from {data_name}")

    # Combine all data
    if not combined_features:
        print("No valid data found in any of the files.")
        return None

    all_features = np.vstack(combined_features)
    all_labels = np.vstack(combined_labels)
    all_force_command = np.vstack(combined_force_command)
    all_position = np.vstack(combined_position)

    print(f"\nTotal combined samples: {all_features.shape[0]}")

    # Scale features
    if scaler.lower() != "none":
        scaler_obj = None
        if scaler.lower() == "standard":
            scaler_obj = StandardScaler()
        elif scaler.lower() == "minmax":
            scaler_obj = MinMaxScaler()
        elif scaler.lower() == "robust":
            scaler_obj = RobustScaler()

        if scaler_obj is not None:
            scaled_features = scaler_obj.fit_transform(all_features)

            # Save scaler for later use
            scaler_dict = {"use_scale": True, "scaler": scaler_obj}
            scaler_file = os.path.join(output_dir, "scaler.pkl")
            save_numpy_compatible_pickle(scaler_dict, scaler_file)

            print(f"Scaler saved to: {scaler_file}")
        else:
            scaled_features = all_features
            print("No scaling applied to features.")
    else:
        scaled_features = all_features
        scaler_dict = {"use_scale": False, "scaler": None}
        scaler_file = os.path.join(output_dir, "scaler.pkl")
        save_numpy_compatible_pickle(scaler_dict, scaler_file)

    # Prepare final data structure
    processed_data = {
        "input_data": all_features,  # Original unscaled features
        "scaled_input_data": scaled_features,  # Scaled features
        "output_data": all_labels,  # Labels (force feedback)
        "force_command": all_force_command,  # Force command
        "position": all_position,  # Position in radians
        "file_config": file_config,  # Keep track of which files were used
    }

    # Save processed data
    output_file = os.path.join(output_dir, "combined_processed_data.pkl")
    save_numpy_compatible_pickle(processed_data, output_file)

    print(f"Combined processed data saved to: {output_file}")
    print(f"Combined input data shape: {all_features.shape}")
    print(f"Combined output data shape: {all_labels.shape}")

    return processed_data


def visualize_processed_data(processed_data, figsize=(15, 10), save_path=None):
    """
    Visualize the processed data to check features and labels.

    Parameters
    ----------
    processed_data : dict
        Dictionary containing the processed data.
    figsize : tuple, default=(15, 10)
        Figure size for the plots.
    save_path : str, optional
        Path to save the visualization plots.
    """
    if processed_data is None:
        print("No data to visualize.")
        return

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    # Time axis
    n_samples = processed_data["output_data"].shape[0]
    dt = 1 / 500  # Based on the standard timestep
    time = np.arange(0, n_samples * dt, dt)[:n_samples]

    # Plot 1: Force feedback vs Force command
    axs[0].plot(time, processed_data["output_data"], label="Force Feedback")
    axs[0].plot(
        time, processed_data["force_command"], label="Force Command", linestyle="--"
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Force")
    axs[0].set_title("Force Feedback vs Command")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Position and Velocity
    axs[1].plot(time, processed_data["position"], label="Position (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Position (rad)")
    axs[1].set_title("Position")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3: Features
    feature_names = [
        "Force Desired",
        "Force Last Desired",
        "Force Last-Last Desired",
        "Position",
        "Position Last",
        "Position Last-Last",
    ]

    for i in range(min(processed_data["input_data"].shape[1], len(feature_names))):
        axs[2].plot(time, processed_data["input_data"][:, i], label=feature_names[i])
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Feature Value")
    axs[2].set_title("Input Features")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to: {save_path}")

    plt.show()


def save_numpy_compatible_pickle(data, file_path, target_numpy_version="1.24.4"):
    """
    Save a dictionary containing NumPy arrays to a pickle file ensuring compatibility
    with a specific NumPy version.

    Parameters
    ----------
    data : dict
        Dictionary containing data to save (potentially including NumPy arrays)
    file_path : str
        Path where the pickle file will be saved
    target_numpy_version : str, default="1.24.4"
        Target NumPy version for compatibility
    """
    current_numpy_version = np.__version__
    print(f"Current NumPy version: {current_numpy_version}")
    print(f"Ensuring compatibility with NumPy {target_numpy_version}")
    # For NumPy 1.24.4 compatibility:
    # 1. Use protocol 4 (supported in Python 3.4+)
    # 2. Convert arrays to base ndarray type without subclasses
    # 3. Make sure arrays are contiguous in memory

    # Function to process numpy arrays for compatibility
    def process_arrays(obj):
        if isinstance(obj, np.ndarray):
            # Convert to a contiguous array of the base ndarray type (no subclasses)
            return np.ascontiguousarray(obj)
        elif isinstance(obj, dict):
            # Process dictionaries recursively
            return {k: process_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Process lists recursively
            return [process_arrays(item) for item in obj]
        elif isinstance(obj, tuple):
            # Process tuples recursively
            return tuple(process_arrays(item) for item in obj)
        else:
            # Return other objects unchanged
            return obj

    # Process all arrays in the data
    compatible_data = process_arrays(data)

    # Use protocol 4 for better compatibility
    with open(file_path, "wb") as f:
        pickle.dump(compatible_data, f, protocol=4)

    print(f"Data saved to {file_path} with NumPy {target_numpy_version} compatibility")


def extract_to_csv(
    datapath, data_name, output_dir=None, data_start=None, data_end=None
):
    """
    Extract data from a .mat file and save it as CSV for testing purposes.

    Parameters
    ----------
    datapath : str
        Path to the directory containing the .mat file.
    data_name : str
        Name of the .mat file (with or without .mat extension).
    output_dir : str, optional
        Directory to save the CSV file. If None, uses the input directory.
    data_start : int, optional
        Starting index for valid data extraction.
    data_end : int, optional
        Ending index for valid data extraction.

    Returns
    -------
    str
        Path to the created CSV file, or None if extraction failed.
    """
    import pandas as pd

    # Set output directory
    if output_dir is None:
        output_dir = datapath

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define required variables for extraction
    required_vars = ["Loa2", "F2_des", "q2_act"]

    # Extract data
    data = extract_data(datapath, data_name, required_vars, data_start, data_end)
    if data is None:
        return None

    # Check if all required variables are present
    if not all(var in data for var in required_vars):
        missing = [var for var in required_vars if var not in data]
        print(f"Error: Missing required variables: {missing}")
        return None

    # Process the data
    force_feedback = np.squeeze(data["Loa2"])
    force_desired = np.squeeze(data["F2_des"])
    position = np.squeeze(data["q2_act"])
    position_rad = np.radians(position)  # Convert to radians
    # Plot and save the extracted data as a quick check
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(force_feedback, label="Force Feedback")
    plt.plot(force_desired, label="Force Desired", linestyle="--")
    plt.legend()
    plt.title("Force Feedback vs Desired")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(position, label="Position (deg)")
    plt.plot(position_rad, label="Position (rad)", linestyle="--")
    plt.legend()
    plt.title("Position")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    # Compute time-delayed features for plotting
    force_last_desired = np.zeros_like(force_desired)
    force_last_desired[1:] = force_desired[:-1]
    force_last_last_desired = np.zeros_like(force_desired)
    force_last_last_desired[2:] = force_desired[:-2]
    position_last = np.zeros_like(position_rad)
    position_last[1:] = position_rad[:-1]
    position_last_last = np.zeros_like(position_rad)
    position_last_last[2:] = position_rad[:-2]

    plt.plot(force_last_desired, label="Force Last Desired")
    plt.plot(force_last_last_desired, label="Force Last-Last Desired")
    plt.plot(position_last, label="Position Last")
    plt.plot(position_last_last, label="Position Last-Last")
    plt.legend()
    plt.title("Time-Delayed Features")
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, f"{os.path.splitext(data_name)[0]}_quickplot.png"
    )
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Quick plot saved to: {plot_path}")
    # Create time array
    n_samples = len(force_feedback)
    dt = 1 / 500  # Based on the standard timestep
    time = np.arange(0, n_samples * dt, dt)[:n_samples]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "time": time,
            "force_feedback": force_feedback,
            "force_desired": force_desired,
            "position_degrees": position,
            "position_radians": position_rad,
        }
    )

    # Create time-delayed features (same as in the training data)
    df["force_last_desired"] = df["force_desired"].shift(1).fillna(0)
    df["force_last_last_desired"] = df["force_desired"].shift(2).fillna(0)
    df["position_last"] = df["position_radians"].shift(1).fillna(0)
    df["position_last_last"] = df["position_radians"].shift(2).fillna(0)

    # Save to CSV
    csv_filename = f"{os.path.splitext(data_name)[0]}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    print(f"Data extracted and saved to CSV: {csv_path}")
    print(f"Number of samples: {n_samples}")

    return csv_path


# Example usage
if __name__ == "__main__":
    # # Example 1: Extract and process data from a .mat file
    datapath = "resources/20250618"
    data_name = "policy1_sin1Hz"
    output_dir = "resources/20250618/processed/pos"

    # # Process data for training
    # processed_data = process_for_training(
    #     datapath=datapath,
    #     data_name=data_name,
    #     output_dir=output_dir,
    #     data_start=100,  # Skip initial transient
    #     data_end=None,  # Use all remaining data
    #     scaler="standard",
    # )

    # # Visualize the processed data
    # if processed_data:
    #     visualize_processed_data(
    #         processed_data,
    #         save_path=os.path.join(output_dir, f"{data_name}_visualization.png"),
    #     )

    # # Example 3: Extract data to CSV for testing
    # csv_file = extract_to_csv(
    #     datapath=datapath,
    #     data_name=data_name,
    #     output_dir=output_dir,
    #     data_start=5000,  # Skip initial transient
    #     data_end=25000,  # Use all remaining data
    # )

    # if csv_file:
    #     print(f"CSV extraction complete. File saved at: {csv_file}")

    # # You can also extract multiple files to CSV
    # for config in file_config if "file_config" in locals() else []:
    #     csv_file = extract_to_csv(
    #         datapath=datapath,
    #         data_name=config["data_name"],
    #         output_dir=output_dir,
    #         data_start=config.get("data_start", None),
    #         data_end=config.get("data_end", None),
    #     )

    # Example 2: Process multiple .mat files
    file_config = [
        {"data_name": "policy1_sin1Hz", "data_start": 3500, "data_end": 25000},
        {"data_name": "policy1_sin_sweep", "data_start": 5000, "data_end": 17500},
        {"data_name": "policy2_sin1Hz", "data_start": 3600, "data_end": 26600},
        {"data_name": "policy2_sin_sweep", "data_start": 3000, "data_end": 15500},
    ]

    combined_processed_data = process_multiple_files(
        datapath=datapath,
        file_config=file_config,
        output_dir=output_dir,
        scaler="standard",
    )

    # Visualize combined processed data
    if combined_processed_data:
        visualize_processed_data(
            combined_processed_data,
            save_path=os.path.join(output_dir, "combined_visualization.png"),
        )
