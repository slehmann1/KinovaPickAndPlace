from __future__ import annotations

from pathlib import Path

from src.grasp_pipeline.config import ExecutionConfig, SceneConfig
from src.grasp_pipeline.execution import append_trial_record, execute_grasp_candidate
from src.grasp_pipeline.models import TrialRecord
from src.grasp_pipeline.sources import GraspSource


def run_grasp_trials(
    source: GraspSource,
    object_index: int,
    max_grasps: int,
    output_path: str | Path,
    scene_config: SceneConfig,
    execution_config: ExecutionConfig,
) -> list[TrialRecord]:
    """Execute a fixed number of candidate grasps and log the resulting outcomes."""
    sample = source.get_object_sample(object_index=object_index, max_grasps=max_grasps)
    results: list[TrialRecord] = []

    for candidate in sample.candidates:
        record = execute_grasp_candidate(
            sample=sample,
            candidate=candidate,
            scene_config=scene_config,
            execution_config=execution_config,
        )
        append_trial_record(output_path=output_path, record=record)
        results.append(record)

    return results
