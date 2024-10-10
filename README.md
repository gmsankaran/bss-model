
# Coordinated Dual-Station Bike Share Simulation

This repository contains a Python script for simulating a bike-sharing scenario using the PySD library to interact with a Vensim model. The simulation is set up for two stations with complementary usage patterns.

## Prerequisites

- Python 3.x
- PySD library for Python (`pip install pysd`)
- Vensim (to create or modify simulation models)
- Any additional libraries used in the script (e.g., NumPy, Pandas)

## File Structure

- `coordinated_dual_station_simulation.py`: The main Python script that runs the simulation.
- Data and model files (not included in the script) should be placed in the following folders:
  - `Data/`: Folder containing input data for the simulation.
  - `Models/`: Folder for forecasting models.
  - `Vensim models/`: Folder containing Vensim simulation models.

## Running the Simulation

1. Ensure all dependencies are installed (listed in Prerequisites).
2. Place the necessary data and models in the appropriate folders.
3. Run the script using:

   ```bash
   python coordinated_dual_station_simulation.py
   ```

4. The script will load the models and data, perform the simulation, and output the results.

## Dependencies

- PySD: A Python interface for Vensim models.
- Other libraries (to be specified in the script or requirements file).

## Notes

Make sure to adjust the folder paths in the script to match your local setup if not using the default structure.
