from typing import List, Dict
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class FeedbackModel(BaseModel):
    main_skills: List[str] = Field(description="Main skills from the candidate CV")
    fit_skills: List[str] = Field(description="Skills from the candidate that fit the job position")
    match_score: int = Field(description="Score from 0 to 100 indicating how the candidate fits the job position")

    def to_dict(self) -> Dict[str, any]:
        return {"main_skills": self.main_skills, "fit_skills": self.fit_skills, "match_score": self.match_score}
    
feedback_parser = PydanticOutputParser(pydantic_object=FeedbackModel)