from fastapi import APIRouter
from models.Candidate import Candidate
import pickle



api_router = APIRouter()

model = pickle.load(open('../model/applicants_hireable.pkl', 'rb'))

# Setting up the home route
@api_router.get("/")
def read_root():
    return {"data": "Welcome to online employee hireability prediction model"}

# Setting up the prediction route
@api_router.post("/prediction/")
async def get_predict(data: Candidate):
    sample = [[
        data.gender,
        data.bachelor_score,
        data.work_experience,
        data.experience_test,
        data.master_score
    ]]

    hired = model.predict(sample).tolist()[0]

    return {
        "data": {
            'prediction': hired,
            'interpretation': 'Candidate can be hired.' if hired == 1 else 'Candidate can not be hired.'
        }
    }
