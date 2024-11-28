from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class RoleMatch(BaseModel):
    rol_name: str = Field(description="Name of the role")
    fit_skills: List[str] = Field(description="Skills from the candidate that fit the role")
    match_score: int = Field(description="Score from 0 to 100 indicating how the candidate fits the role")

    def to_dict(self) -> Dict[str, Any]:
        return {"rol_name": self.rol_name, "fit_skills": self.fit_skills, "match_score": self.match_score}

class FeedbackModel(BaseModel):
    main_skills: List[str] = Field(description="Main skills from the candidate CV")
    role_matches: List[RoleMatch] = Field(description="List of role matches with the candidate")

    def to_dict(self) -> Dict[str, Any]:
        return {"main_skills": self.main_skills, "role_matches": [role_match.to_dict() for role_match in self.role_matches]}

feedback_parser = PydanticOutputParser(pydantic_object=FeedbackModel)
