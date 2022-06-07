from pydantic import BaseModel 

class Candidate(BaseModel):
    gender: int
    bachelor_score: float
    work_experience: int
    experience_test: int
    master_score: float
    