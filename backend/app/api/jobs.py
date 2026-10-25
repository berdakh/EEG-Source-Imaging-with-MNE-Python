from fastapi import APIRouter, Depends, HTTPException

from app.jobs import JobQueue

from .deps import get_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(jobs: JobQueue = Depends(get_jobs)):
    return jobs.list()


@router.get("/{job_id}")
async def get_job(job_id: str, jobs: JobQueue = Depends(get_jobs)):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job
