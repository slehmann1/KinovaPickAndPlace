# Franka Pick and Place


## What is in this repository

- A custom `TargetEnvironment` with a table, target zone, and one manipulable object.
- Simple PID-style arm control helpers for Cartesian motion and gripper commands.
- Dataset utilities for loading GraspFactory grasp annotations and generating MJCF wrappers for OBJ meshes.
- Imitation learning pipeline using the GraspFactory dataset and associated simulatiion functionality.
- Obstacle avoidance routine executing a pick-and-place task.
- Integration pipeline combining simulation of  imitation-based grasping with pick-and-place task in an obstacle-laden environment.

## Setup

This project currently targets a Windows PowerShell workflow.

1. Create the environment and install dependencies:

   ```powershell
   .\build.ps1
   ```

2. Activate the virtual environment when you want an interactive shell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

The build script installs local dependencies, clones the upstream GraspFactory repository into `data/graspfactory_repo`, and copies its sample dataset into `data/graspfactory`.

This provides a sample dataset, and includes a helper script to download the entire dataset used for training and evaluation.
The full dataset is required for execution of the grasping and integrated routines. Note that the directory for this is hard coded and should be updated for your own system.

## Running the scripts

From the project root:

```powershell
.\.venv\Scripts\python.exe .\main.py
.\.venv\Scripts\python.exe .\obstacle_navigation_rrt.py
.\.venv\Scripts\python.exe .\pick_place_obstacle_rrt.py
.\.venv\Scripts\python.exe .\evaluate_obstacle_navigation.py
.\.venv\Scripts\python.exe .\imitation_grasp.py
.\.venv\Scripts\python.exe .\evaluate_imitation_grasp.py --num-objects 10
.\.venv\Scripts\python.exe .\imitation_grasp_obstacle_rrt.py
.\.venv\Scripts\python.exe -m src.grasp.grasp_playground
.\.venv\Scripts\python.exe -m src.utils.mesh_debug
```

## Notes

- `main.py` is the simplest end-to-end scripted demo.
- `obstacle_navigation_rrt.py` runs the obstacle-navigation RRT demo.
- `pick_place_obstacle_rrt.py` runs the obstacle-aware pick-and-place pipeline.
- `evaluate_obstacle_navigation.py` runs randomized obstacle-scene evaluation and writes a report.
- `imitation_grasp.py` trains a grasping policy using imitation learning.
- `evaluate_imitation_grasp.py` evaluates imitation grasping policies across multiple objects and logs success rates.
- `imitation_grasp_obstacle_rrt.py` runs the integrated pipeline: it tests an imitation-based grasp on one object, and if that grasp succeeds, retries it in an obstacle scene and completes the pick-and-place task.
- `python -m src.grasp.grasp_playground` is the current sandbox for loading one dataset grasp and inspecting it in the scene.
- `python -m src.utils.mesh_debug` is useful when the loaded mesh appears offset or rotated incorrectly.

## Demos
Demonstrations of the various routines are available in the ```demos``` folder.