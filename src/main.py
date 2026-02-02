# src/main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_models import SimulationInput, OrbitalElements, SpringParameters
from src.simulation import run_simulation, _calculate_spring_force

def process_results(solution: object, sim_input: SimulationInput) -> pd.DataFrame:
    """
    Processes simulation results from 'solve_ivp'.

    1. Extracts state vectors from the solution for each time step.
    2. Calculates relative parameters and stores full state vectors.
    3. Generates and returns a pandas DataFrame with the results.
    """
    results = []
    
    for i, t in enumerate(solution.t):
        state = solution.y[:, i]
        r_sc, v_sc = state[0:3], state[3:6]
        r_lv, v_lv = state[6:9], state[9:12]

        relative_distance = np.linalg.norm(r_sc - r_lv)
        relative_speed = np.linalg.norm(v_sc - v_lv)
        
        spring_force_vec = _calculate_spring_force(r_sc, r_lv, sim_input.spring)
        spring_force_mag = np.linalg.norm(spring_force_vec)

        results.append({
            "time_s": t,
            "r_sc_x": r_sc[0], "r_sc_y": r_sc[1], "r_sc_z": r_sc[2],
            "v_sc_x": v_sc[0], "v_sc_y": v_sc[1], "v_sc_z": v_sc[2],
            "r_lv_x": r_lv[0], "r_lv_y": r_lv[1], "r_lv_z": r_lv[2],
            "v_lv_x": v_lv[0], "v_lv_y": v_lv[1], "v_lv_z": v_lv[2],
            "relative_distance_m": relative_distance,
            "relative_speed_mps": relative_speed,
            "spring_force_N": spring_force_mag
        })

    return pd.DataFrame(results)

def main():
    """
    The main executable function.
    1. Defines the input data for the simulation.
    2. Starts the simulation.
    3. Processes the results.
    4. Saves the DataFrame to .csv and builds graphs.
    """
    print("Starting simulation...")

    orbit = OrbitalElements(
        semi_major_axis_km=6700.0,
        eccentricity=0.003,
        inclination_deg=80.0,
        raan_deg=-15.0,
        argument_of_periapsis_deg=30.0,
        mean_anomaly_deg=0.0
    )
    spring = SpringParameters(
        stiffness_Npm=30000.0,
        initial_length_mm=110.0,
        natural_length_mm=220.0,
        final_length_mm=195.0
    )
    sim_input = SimulationInput(
        orbit=orbit,
        spring=spring,
        spacecraft_mass_kg=3100.0,
        launch_vehicle_mass_kg=3500.0,
        simulation_time_s=5.0
    )

    start_time = time.time()
    solution = run_simulation(sim_input)
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    results_df = process_results(solution, sim_input)
    print("Processing results...")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simulation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # --- File 1: Relative values ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig1.suptitle('Relative Kinematics and Forces', fontsize=16)

    axes1[0].plot(results_df['time_s'], results_df['relative_distance_m'], label='Relative Distance (m)')
    axes1[0].plot(results_df['time_s'], results_df['relative_speed_mps'], label='Relative Speed (m/s)', linestyle='--')
    axes1[0].set_ylabel('Relative Values')
    axes1[0].legend()
    axes1[0].grid(True)

    axes1[1].plot(results_df['time_s'], results_df['spring_force_N'])
    axes1[1].set_ylabel('Spring Force (N)')
    axes1[1].set_xlabel('Time (s)')
    axes1[1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path1 = os.path.join(output_dir, "simulation_plots_relative.png")
    plt.savefig(plot_path1)
    plt.close(fig1)
    print(f"Relative plots saved to {plot_path1}")

    # --- File 2: Absolute positions ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig2.suptitle('Absolute Position (ECI)', fontsize=16)

    axes2[0].plot(results_df['time_s'], results_df['r_sc_x'], label='Spacecraft')
    axes2[0].plot(results_df['time_s'], results_df['r_lv_x'], label='Launch Vehicle', linestyle='--')
    axes2[0].set_ylabel('X Position (m)')
    axes2[0].legend()
    axes2[0].grid(True)

    axes2[1].plot(results_df['time_s'], results_df['r_sc_y'], label='Spacecraft')
    axes2[1].plot(results_df['time_s'], results_df['r_lv_y'], label='Launch Vehicle', linestyle='--')
    axes2[1].set_ylabel('Y Position (m)')
    axes2[1].legend()
    axes2[1].grid(True)

    axes2[2].plot(results_df['time_s'], results_df['r_sc_z'], label='Spacecraft')
    axes2[2].plot(results_df['time_s'], results_df['r_lv_z'], label='Launch Vehicle', linestyle='--')
    axes2[2].set_ylabel('Z Position (m)')
    axes2[2].set_xlabel('Time (s)')
    axes2[2].legend()
    axes2[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path2 = os.path.join(output_dir, "simulation_plots_position.png")
    plt.savefig(plot_path2)
    plt.close(fig2)
    print(f"Position plots saved to {plot_path2}")

    # --- File 3: Absolute Speeds ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig3.suptitle('Absolute Velocity (ECI)', fontsize=16)

    axes3[0].plot(results_df['time_s'], results_df['v_sc_x'], label='Spacecraft')
    axes3[0].plot(results_df['time_s'], results_df['v_lv_x'], label='Launch Vehicle', linestyle='--')
    axes3[0].set_ylabel('X Velocity (m/s)')
    axes3[0].legend()
    axes3[0].grid(True)

    axes3[1].plot(results_df['time_s'], results_df['v_sc_y'], label='Spacecraft')
    axes3[1].plot(results_df['time_s'], results_df['v_lv_y'], label='Launch Vehicle', linestyle='--')
    axes3[1].set_ylabel('Y Velocity (m/s)')
    axes3[1].legend()
    axes3[1].grid(True)

    axes3[2].plot(results_df['time_s'], results_df['v_sc_z'], label='Spacecraft')
    axes3[2].plot(results_df['time_s'], results_df['v_lv_z'], label='Launch Vehicle', linestyle='--')
    axes3[2].set_ylabel('Z Velocity (m/s)')
    axes3[2].set_xlabel('Time (s)')
    axes3[2].legend()
    axes3[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path3 = os.path.join(output_dir, "simulation_plots_velocity.png")
    plt.savefig(plot_path3)
    plt.close(fig3)
    print(f"Velocity plots saved to {plot_path3}")
    

if __name__ == "__main__":
    main()
