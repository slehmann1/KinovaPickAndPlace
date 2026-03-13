# Franka Pick and Place

Robosuite-based pick-and-place experiments for loading GraspFactory meshes, visualizing candidate grasps, and testing simple scripted manipulation routines.

## What is in this repository

- A custom `TargetEnvironment` with a table, target zone, and one manipulable object.
- Simple PID-style arm control helpers for Cartesian motion and gripper commands.
- Dataset utilities for loading GraspFactory grasp annotations and generating MJCF wrappers for OBJ meshes.
- Debug scripts for grasp inspection and mesh alignment checks.

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

## Running the scripts

From the project root:

```powershell
.\.venv\Scripts\python.exe .\src\main.py
.\.venv\Scripts\python.exe .\src\grasp_playground.py
.\.venv\Scripts\python.exe .\src\mesh_debug.py
```

## Notes

- `src/main.py` is the simplest end-to-end scripted demo.
- `src/grasp_playground.py` is the current sandbox for loading one dataset grasp and inspecting it in the scene.
- `src/mesh_debug.py` is useful when the loaded mesh appears offset or rotated incorrectly.
- Dataset files, simulator logs, and generated local artifacts are ignored by `.gitignore` so the commit surface stays focused on source code and documentation.

## Current limitations

- The scripted motions in `src/main.py` are hardcoded for a narrow demo scenario.
- The environment reward is still a placeholder and is not suitable for learning experiments yet.
- Dataset mesh placement is still being validated for more varied object geometries and orientations.
