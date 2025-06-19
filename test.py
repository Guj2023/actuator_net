import torch
import pandas as pd
import os
import os.path as osp
import pickle as pkl
from train import ActuatorNet
from sklearn.metrics import r2_score
import numpy as np
import time
import matplotlib.pyplot as plt


def load_model(model_path, scaler, use_scale=False, device="cpu"):
    print(f"Loading model from: {model_path}")
    print(f"Use scaling: {use_scale}")

    actuator_net = None

    if "pth" == model_path[-3:]:
        try:
            actuator_net = torch.jit.load(model_path, map_location=torch.device(device))
            print("Loaded TorchScript model")
        except RuntimeError as e:
            print(f"Failed to load TorchScript model: {e}")
            print("Attempting to load as state dict instead...")
            try:
                # Load the saved model info
                model_data = torch.load(model_path, map_location=device)

                if isinstance(model_data, dict) and "state_dict" in model_data:
                    # New format with model parameters
                    params = model_data["model_params"]
                    actuator_net = ActuatorNet(**params)
                    actuator_net.load_state_dict(model_data["state_dict"])
                    print("Successfully loaded model with saved parameters")
                else:
                    # Old format - try to load as state dict directly
                    actuator_net = ActuatorNet(
                        in_dim=6, out_dim=1, units=100, layers=4, act="softsign"
                    )
                    actuator_net.load_state_dict(model_data)
                    print("Successfully loaded as state dict with default parameters")

                actuator_net.to(device)
            except Exception as e2:
                print(f"Failed to load as state dict: {e2}")
                raise RuntimeError(
                    f"Could not load model as TorchScript or state dict: {e}, {e2}"
                )
    else:
        actuator_net = ActuatorNet(in_dim=6, out_dim=1)
        state_dict = torch.load(model_path, map_location=device)
        actuator_net.load_state_dict(state_dict)
        actuator_net.to(device)
        print("Loaded state dict model")

    actuator_net.eval()  # Ensure eval mode
    print(f"Model is in training mode: {actuator_net.training}")

    def eval_actuator_network(
        force_desired,
        force_desired_last,
        force_desired_last_last,
        pos,
        pos_last,
        pos_last_last,
    ):
        inputs = torch.tensor(
            [
                force_desired,
                force_desired_last,
                force_desired_last_last,
                pos,
                pos_last,
                pos_last_last,
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        if use_scale:
            scaled_inputs = scaler.transform(inputs.cpu().numpy())
            inputs = torch.tensor(scaled_inputs, dtype=torch.float32, device=device)

        # Debug: Print first few inputs to compare with training
        if hasattr(eval_actuator_network, "debug_count"):
            eval_actuator_network.debug_count += 1
        else:
            eval_actuator_network.debug_count = 1

        if eval_actuator_network.debug_count <= 3:
            print(f"Input {eval_actuator_network.debug_count}: {inputs.cpu().numpy()}")

        return actuator_net(inputs).squeeze()

    return eval_actuator_network


def load_data(datafile_dir):
    data_path = os.path.join(datafile_dir, "policy1_sin1Hz.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file does not exist: {data_path}")

    rawdata = pd.read_csv(data_path)
    scaler_file = os.path.join(datafile_dir, "scaler.pkl")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler file does not exist: {scaler_file}")

    with open(scaler_file, "rb") as fd:
        scaler_dict = pkl.load(fd)
        scaler = scaler_dict["scaler"]
        use_scale = scaler_dict["use_scale"]

    return rawdata, scaler, use_scale


if __name__ == "__main__":
    # Configuration
    model_path = osp.join("./app/resources", "actuator_eval_model.pth")
    device = "cpu"
    print(f"Using device: {device}")

    # Load data and model
    rawdata, scaler, use_scale = load_data(os.path.dirname(model_path))
    print(f"Data shape: {len(rawdata)}")
    print(f"Scaler type: {type(scaler)}")

    actuator_network = load_model(model_path, scaler, use_scale, device)

    # Run inference
    actual_force = []
    estimated_force = []
    desired_force = []
    calculation_force = []
    height = []

    start_time = time.time()

    for idx in range(min(2000, len(rawdata))):
        with torch.no_grad():
            # Get input values
            force_desired = rawdata["force_desired"][idx]
            force_desired_last = rawdata["force_last_desired"][idx]
            force_desired_last_last = rawdata["force_last_last_desired"][idx]
            pos = rawdata["position_radians"][idx]
            pos_last = rawdata["position_last"][idx]
            pos_last_last = rawdata["position_last_last"][idx]

            # Store height for plotting
            height.append(1 - 0.38 - np.cos(pos) * 0.38)

            # Store actual force and desired force
            actual_force.append(rawdata["force_feedback"][idx])
            desired_force.append(force_desired)

            # Calculate force using physics equation
            calculation = force_desired - 8.6 * (pos - pos_last) / 0.002
            calculation_force.append(calculation)

            # Predict force
            torque = actuator_network(
                force_desired,
                force_desired_last,
                force_desired_last_last,
                pos,
                pos_last,
                pos_last_last,
            )
            estimated_force.append(torque.item())

    end_time = time.time()
    print(f"Loop time cost: {end_time - start_time:.4f} seconds")

    # Convert to numpy arrays
    actual = np.array(actual_force)
    estimation = np.array(estimated_force)
    desired = np.array(desired_force)
    calculation = np.array(calculation_force)

    # Calculate R2 scores
    r2 = r2_score(actual, estimation)
    cal_r2 = r2_score(actual, calculation)
    print(f"Test R2 score: {r2:.4f}")
    print(f"Calculation R2 score: {cal_r2:.4f}")

    # Plot results with height subplot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Force predictions
    axs[0].plot(actual, label="Actual", color="black")
    axs[0].plot(estimation, label="Estimated", color="red")
    axs[0].plot(calculation, label="Calculation", color="green")
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title(
        f"Actuator Force Prediction (Est R2: {r2:.4f}, Calc R2: {cal_r2:.4f})"
    )
    axs[0].set_ylabel("Force")

    # Height subplot
    axs[1].plot(height, label="Height", color="blue")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title("Height Over Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Height")

    plt.tight_layout()
    plt.show()
