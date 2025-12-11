"""
MCP Job Resources

Resources for accessing training job status.
"""

from typing import Any
from datetime import datetime


class JobResources:
    """Training job resources."""

    async def get_active_jobs(self) -> dict[str, Any]:
        """Get active and recent training jobs."""
        # Import job storage from training tools
        try:
            from mcp.tools.training import _jobs
        except ImportError:
            _jobs = {}

        active_jobs = []
        completed_jobs = []
        failed_jobs = []

        for job_id, job in _jobs.items():
            job_summary = {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "model_name": job["config"]["model_name"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"]
            }

            if job["status"] == "running":
                active_jobs.append(job_summary)
            elif job["status"] == "completed":
                completed_jobs.append(job_summary)
            elif job["status"] == "failed":
                job_summary["error"] = job.get("error")
                failed_jobs.append(job_summary)

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "active": len(active_jobs),
                "completed": len(completed_jobs),
                "failed": len(failed_jobs),
                "total": len(_jobs)
            },
            "active_jobs": active_jobs,
            "recent_completed": completed_jobs[-5:],  # Last 5 completed
            "recent_failed": failed_jobs[-5:]  # Last 5 failed
        }
