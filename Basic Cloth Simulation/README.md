**Basic Cloth Simulation Demo**

This folder contains several basic implementation of cloth simulation.

For detailed parameter setup, checkout `metadata.py`. 

For simulation implementation, read through the following:

- To try out explicit Euler simulation, run `python3 explicit.py`.
This script construct the cloth simulation using:
  1. explicit euler simulation step
  2. Hook's law for spring force computation
  3. simplified impulse collision simulation


- To try out implicit Euler simulation, run `python3 implicit.py`.
This script construct the cloth simulation using: (has BUG now)
  1. implicit euler simulation step based on GAMES 103 HW2 process
  2. Hook's law for spring force computation

- To try out implicit Euler simulation, run `python3 PBD.py`.
This script construct the cloth simulation using



*Note: different file might require different setting to run properly*