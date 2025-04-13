from fastapi import FastAPI, Request
from pydantic import BaseModel
from model_utils import evaluate_college
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class CollegeRequest(BaseModel):
    colleges: list
    major: str
    income_tier: int
    user_state: str
    parent_loans: bool
    weight_qol: float
    weight_roi: float

@app.post("/recommend")
async def recommend_college(request: CollegeRequest):
    print("Received payload:", request)
    result = evaluate_college(
        college_list=request.colleges,
        major=request.major,
        income_tier=request.income_tier,
        user_state=request.user_state,
        parent_loans=request.parent_loans,
        weight_qol=request.weight_qol,
        weight_roi=request.weight_roi
    )
    return result


@app.get("/")
def home():
    return {"message": "College Advisor API is running!"}
