import asyncio
import json
import re
from typing import Annotated, Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import PyPDF2
import io
import openai
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

# Configuration
OPENAI_API_KEY = "your-openai-api-key"  # Replace with your actual API key
openai.api_key = OPENAI_API_KEY

# Data structures for CV and job information
@dataclass
class PersonalInfo:
    full_name: str
    email: str
    phone: str
    location: str
    linkedin_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    github_url: Optional[str] = None

@dataclass
class Experience:
    job_title: str
    company: str
    start_date: str
    end_date: Optional[str]
    description: str
    achievements: List[str]
    technologies: List[str]

@dataclass
class Education:
    degree: str
    institution: str
    graduation_date: str
    gpa: Optional[str] = None
    relevant_coursework: List[str] = None

@dataclass
class Project:
    name: str
    description: str
    technologies: List[str]
    url: Optional[str] = None
    achievements: List[str] = None

@dataclass
class CVData:
    personal_info: PersonalInfo
    summary: str
    experience: List[Experience]
    education: List[Education]
    skills: List[str]
    projects: List[Project]
    certifications: List[str]
    languages: List[str]
    last_updated: str

@dataclass
class JobPosting:
    job_title: str
    company: str
    location: str
    job_description: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_level: str
    salary_range: Optional[str] = None
    job_url: Optional[str] = None

@dataclass
class JobMatch:
    job_posting: JobPosting
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    tailored_cv: Optional[str] = None

class ChatGPTService:
    """Service for ChatGPT API interactions"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def parse_cv_from_text(self, cv_text: str) -> Dict:
        """Use ChatGPT to parse CV text into structured data"""
        prompt = f"""
        Parse the following CV text and extract structured information. Return a JSON object with the following structure:

        {{
            "personal_info": {{
                "full_name": "string",
                "email": "string",
                "phone": "string",
                "location": "string",
                "linkedin_url": "string or null",
                "portfolio_url": "string or null",
                "github_url": "string or null"
            }},
            "summary": "string",
            "experience": [
                {{
                    "job_title": "string",
                    "company": "string",
                    "start_date": "string",
                    "end_date": "string or null",
                    "description": "string",
                    "achievements": ["string"],
                    "technologies": ["string"]
                }}
            ],
            "education": [
                {{
                    "degree": "string",
                    "institution": "string",
                    "graduation_date": "string",
                    "gpa": "string or null",
                    "relevant_coursework": ["string"]
                }}
            ],
            "skills": ["string"],
            "projects": [
                {{
                    "name": "string",
                    "description": "string",
                    "technologies": ["string"],
                    "url": "string or null",
                    "achievements": ["string"]
                }}
            ],
            "certifications": ["string"],
            "languages": ["string"]
        }}

        CV Text:
        {cv_text}

        Please extract all available information and return only the JSON object:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert CV parser. Extract structured information from CV text and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            json_response = response.choices[0].message.content
            # Clean the response to ensure it's valid JSON
            json_response = json_response.strip()
            if json_response.startswith("```json"):
                json_response = json_response[7:-3]
            elif json_response.startswith("```"):
                json_response = json_response[3:-3]
            
            return json.loads(json_response)
        
        except Exception as e:
            raise Exception(f"Failed to parse CV with ChatGPT: {str(e)}")
    
    async def parse_job_posting(self, job_text: str) -> Dict:
        """Use ChatGPT to parse job posting text"""
        prompt = f"""
        Parse the following job posting and extract structured information. Return a JSON object with this structure:

        {{
            "job_title": "string",
            "company": "string",
            "location": "string",
            "job_description": "string",
            "required_skills": ["string"],
            "preferred_skills": ["string"],
            "experience_level": "string",
            "salary_range": "string or null",
            "job_url": "string or null"
        }}

        Job Posting:
        {job_text}

        Return only the JSON object:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert job posting parser. Extract structured information and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            json_response = response.choices[0].message.content.strip()
            if json_response.startswith("```json"):
                json_response = json_response[7:-3]
            elif json_response.startswith("```"):
                json_response = json_response[3:-3]
            
            return json.loads(json_response)
        
        except Exception as e:
            raise Exception(f"Failed to parse job posting with ChatGPT: {str(e)}")
    
    async def tailor_cv_for_job(self, cv_data: CVData, job_posting: JobPosting) -> str:
        """Use ChatGPT to tailor CV for specific job"""
        prompt = f"""
        Tailor the following CV for the specific job posting. Focus on:
        1. Highlighting relevant experience and skills
        2. Adjusting the professional summary
        3. Prioritizing relevant achievements
        4. Using keywords from the job description

        CV Data:
        {json.dumps(asdict(cv_data), indent=2)}

        Job Posting:
        {json.dumps(asdict(job_posting), indent=2)}

        Create a tailored CV that emphasizes the most relevant aspects for this role:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert CV writer. Create tailored CVs that highlight the most relevant experience and skills for specific job postings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"Failed to tailor CV with ChatGPT: {str(e)}")
    
    async def generate_cv_summary(self, cv_data: CVData) -> str:
        """Generate a comprehensive CV summary using ChatGPT"""
        prompt = f"""
        Create a comprehensive professional summary of this CV:

        {json.dumps(asdict(cv_data), indent=2)}

        Include:
        - Key professional highlights
        - Core competencies
        - Career progression
        - Notable achievements
        - Technical skills summary

        Make it engaging and professional:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert career consultant. Create compelling professional summaries that highlight a candidate's strengths."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"Failed to generate CV summary with ChatGPT: {str(e)}")

class CVMemoryManager:
    """Manages CV data in memory"""
    
    def __init__(self):
        self.cv_data: Optional[CVData] = None
        self.job_matches: List[JobMatch] = []
        self.job_postings: List[JobPosting] = []
        self.conversation_history: List[Dict] = []
    
    def store_cv(self, cv_data: CVData):
        """Store CV data in memory"""
        self.cv_data = cv_data
        self.cv_data.last_updated = datetime.now().isoformat()
    
    def store_job_posting(self, job_posting: JobPosting):
        """Store job posting in memory"""
        self.job_postings.append(job_posting)
    
    def update_cv_section(self, section: str, data: Any):
        """Update specific section of CV"""
        if self.cv_data:
            setattr(self.cv_data, section, data)
            self.cv_data.last_updated = datetime.now().isoformat()
    
    def get_cv_summary(self) -> str:
        """Get formatted CV summary"""
        if not self.cv_data:
            return "No CV data stored"
        
        return f"""
        CV Summary for {self.cv_data.personal_info.full_name}:
        - Email: {self.cv_data.personal_info.email}
        - Location: {self.cv_data.personal_info.location}
        - Experience: {len(self.cv_data.experience)} positions
        - Skills: {', '.join(self.cv_data.skills[:5])}{'...' if len(self.cv_data.skills) > 5 else ''}
        - Last Updated: {self.cv_data.last_updated}
        """

class JobMatcher:
    """Handles job matching logic"""
    
    def calculate_match_score(self, cv_data: CVData, job_posting: JobPosting) -> JobMatch:
        """Calculate match score between CV and job posting"""
        cv_skills = set(skill.lower() for skill in cv_data.skills)
        required_skills = set(skill.lower() for skill in job_posting.required_skills)
        preferred_skills = set(skill.lower() for skill in job_posting.preferred_skills)
        
        # Calculate matching and missing skills
        matching_required = cv_skills.intersection(required_skills)
        matching_preferred = cv_skills.intersection(preferred_skills)
        missing_required = required_skills - cv_skills
        missing_preferred = preferred_skills - cv_skills
        
        # Calculate score (weighted: required skills worth more)
        required_match_ratio = len(matching_required) / len(required_skills) if required_skills else 0
        preferred_match_ratio = len(matching_preferred) / len(preferred_skills) if preferred_skills else 0
        
        # Score calculation: 70% weight for required skills, 30% for preferred
        match_score = (required_match_ratio * 0.7) + (preferred_match_ratio * 0.3)
        
        return JobMatch(
            job_posting=job_posting,
            match_score=match_score,
            matching_skills=list(matching_required.union(matching_preferred)),
            missing_skills=list(missing_required.union(missing_preferred))
        )
    
    def find_job_matches(self, cv_data: CVData, job_postings: List[JobPosting]) -> List[JobMatch]:
        """Find and rank job matches"""
        matches = []
        for job in job_postings:
            match = self.calculate_match_score(cv_data, job)
            matches.append(match)
        
        # Sort by match score (highest first)
        return sorted(matches, key=lambda x: x.match_score, reverse=True)

class CVToolkit:
    """Main toolkit containing all CV-related functions"""
    
    def __init__(self):
        self.chatgpt_service = ChatGPTService(OPENAI_API_KEY)
        self.memory_manager = CVMemoryManager()
        self.job_matcher = JobMatcher()
    
    async def process_cv_pdf(self, file_info: Dict) -> str:
        """Process CV PDF and store data"""
        try:
            # Extract text from PDF
            pdf_content = await self.extract_pdf_content(file_info)
            
            # Parse CV data using ChatGPT
            cv_dict = await self.chatgpt_service.parse_cv_from_text(pdf_content)
            
            # Convert to CVData object
            cv_data = self.dict_to_cv_data(cv_dict)
            
            # Store in memory
            self.memory_manager.store_cv(cv_data)
            
            return f"✅ CV successfully processed and stored for {cv_data.personal_info.full_name}"
        
        except Exception as e:
            return f"❌ Error processing CV: {str(e)}"
    
    async def process_job_posting(self, job_text: str) -> str:
        """Process job posting text"""
        try:
            # Parse job posting using ChatGPT
            job_dict = await self.chatgpt_service.parse_job_posting(job_text)
            
            # Convert to JobPosting object
            job_posting = self.dict_to_job_posting(job_dict)
            
            # Store in memory
            self.memory_manager.store_job_posting(job_posting)
            
            return f"✅ Job posting for {job_posting.job_title} at {job_posting.company} processed and stored"
        
        except Exception as e:
            return f"❌ Error processing job posting: {str(e)}"
    
    async def generate_cv_summary(self) -> str:
        """Generate CV summary"""
        if not self.memory_manager.cv_data:
            return "❌ No CV data available. Please upload your CV first."
        
        try:
            summary = await self.chatgpt_service.generate_cv_summary(self.memory_manager.cv_data)
            return f"📄 **CV Summary**\n\n{summary}"
        
        except Exception as e:
            return f"❌ Error generating CV summary: {str(e)}"
    
    async def find_job_matches(self) -> str:
        """Find job matches for stored CV"""
        if not self.memory_manager.cv_data:
            return "❌ No CV data available. Please upload your CV first."
        
        if not self.memory_manager.job_postings:
            return "❌ No job postings available. Please add some job postings first."
        
        try:
            matches = self.job_matcher.find_job_matches(
                self.memory_manager.cv_data, 
                self.memory_manager.job_postings
            )
            
            result = "🎯 **Job Matches Found**\n\n"
            for i, match in enumerate(matches[:5], 1):  # Show top 5 matches
                result += f"**{i}. {match.job_posting.job_title} at {match.job_posting.company}**\n"
                result += f"   Match Score: {match.match_score:.2%}\n"
                result += f"   Matching Skills: {', '.join(match.matching_skills[:5])}\n"
                result += f"   Missing Skills: {', '.join(match.missing_skills[:3])}\n\n"
            
            return result
        
        except Exception as e:
            return f"❌ Error finding job matches: {str(e)}"
    
    async def tailor_cv_for_job(self, job_index: int = 0) -> str:
        """Tailor CV for specific job"""
        if not self.memory_manager.cv_data:
            return "❌ No CV data available. Please upload your CV first."
        
        if not self.memory_manager.job_postings:
            return "❌ No job postings available. Please add some job postings first."
        
        if job_index >= len(self.memory_manager.job_postings):
            return f"❌ Job index {job_index} not found. Available jobs: {len(self.memory_manager.job_postings)}"
        
        try:
            job_posting = self.memory_manager.job_postings[job_index]
            tailored_cv = await self.chatgpt_service.tailor_cv_for_job(
                self.memory_manager.cv_data,
                job_posting
            )
            
            return f"✏️ **Tailored CV for {job_posting.job_title} at {job_posting.company}**\n\n{tailored_cv}"
        
        except Exception as e:
            return f"❌ Error tailoring CV: {str(e)}"
    
    def get_stored_data_info(self) -> str:
        """Get information about stored data"""
        cv_info = "✅ CV stored" if self.memory_manager.cv_data else "❌ No CV stored"
        job_info = f"✅ {len(self.memory_manager.job_postings)} job posting(s) stored" if self.memory_manager.job_postings else "❌ No job postings stored"
        
        return f"📊 **Data Status**\n{cv_info}\n{job_info}"
    
    async def extract_pdf_content(self, file_info: Dict) -> str:
        """Extract text content from PDF file"""
        try:
            file_content = file_info.get('content', b'')
            
            if isinstance(file_content, str):
                return file_content
            
            # Use PyPDF2 to extract text
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Failed to extract PDF content: {str(e)}")
    
    def dict_to_cv_data(self, cv_dict: Dict) -> CVData:
        """Convert dictionary to CVData object"""
        personal_info = PersonalInfo(**cv_dict['personal_info'])
        
        experience = [Experience(**exp) for exp in cv_dict['experience']]
        education = [Education(**edu) for edu in cv_dict['education']]
        projects = [Project(**proj) for proj in cv_dict['projects']]
        
        return CVData(
            personal_info=personal_info,
            summary=cv_dict['summary'],
            experience=experience,
            education=education,
            skills=cv_dict['skills'],
            projects=projects,
            certifications=cv_dict['certifications'],
            languages=cv_dict['languages'],
            last_updated=datetime.now().isoformat()
        )
    
    def dict_to_job_posting(self, job_dict: Dict) -> JobPosting:
        """Convert dictionary to JobPosting object"""
        return JobPosting(**job_dict)

# Global toolkit instance
cv_toolkit = CVToolkit()

# Agent configuration
AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0MTNlMzdiOS00YTUyLTQwYzMtYjg0NS0zNTdmMzEyMDNiMzAiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6Ijg2NzRkNzNmLTlkOTYtNGFhOC1hYmVjLThiMGFjNTk0YWExMCJ9.Raph28nCcP95HVd66Zvxlfs17XCxR4-wb0mA16kXlOM"
session = GenAISession(jwt_token=AGENT_JWT)

@session.bind(
    name="job_search_agent",
    description="AI-powered job search assistant that helps with CV processing, job matching, and career optimization"
)
async def job_search_agent(
    agent_context: GenAIContext,
    user_input: Annotated[str, "User input for job search assistance"]
):
    """
    AI-powered job search assistant that provides:
    - CV processing and analysis
    - Job posting parsing
    - Intelligent job matching
    - CV tailoring for specific roles
    - Career insights and recommendations
    """
    
    try:
        # Check for attached files
        attached_files = getattr(agent_context, 'files', []) or []
        
        # Handle PDF CV uploads
        if attached_files and any(file.get('name', '').lower().endswith('.pdf') for file in attached_files):
            for file in attached_files:
                if file.get('name', '').lower().endswith('.pdf'):
                    return await cv_toolkit.process_cv_pdf(file)
        
        # The master agent will handle tool selection, so we provide simple responses
        # and let the master agent decide what to do
        
        return """
        🤖 **Job Search Assistant Ready**
        
        I'm your AI-powered job search assistant! Here's what I can help you with:
        
        **📄 CV Management:**
        - Upload and parse PDF CVs
        - Generate professional summaries
        - Update and optimize CV sections
        
        **🎯 Job Matching:**
        - Parse job postings from text or URLs
        - Calculate compatibility scores
        - Find the best job matches
        
        **✏️ CV Tailoring:**
        - Customize CVs for specific roles
        - Highlight relevant experience
        - Optimize for applicant tracking systems
        
        **📊 Career Insights:**
        - Skill gap analysis
        - Career progression recommendations
        - Market trend insights
        
        **Current Status:**
        """ + cv_toolkit.get_stored_data_info() + """
        
        What would you like to do today?
        """
    
    except Exception as e:
        return f"❌ I encountered an error: {str(e)}. Please try again or contact support."

# Tool functions for master agent integration
async def process_cv_pdf_tool(file_info: Dict) -> str:
    """Tool for processing CV PDFs"""
    return await cv_toolkit.process_cv_pdf(file_info)

async def process_job_posting_tool(job_text: str) -> str:
    """Tool for processing job postings"""
    return await cv_toolkit.process_job_posting(job_text)

async def generate_cv_summary_tool() -> str:
    """Tool for generating CV summaries"""
    return await cv_toolkit.generate_cv_summary()

async def find_job_matches_tool() -> str:
    """Tool for finding job matches"""
    return await cv_toolkit.find_job_matches()

async def tailor_cv_tool(job_index: int = 0) -> str:
    """Tool for tailoring CVs"""
    return await cv_toolkit.tailor_cv_for_job(job_index)

async def get_data_status_tool() -> str:
    """Tool for getting data status"""
    return cv_toolkit.get_stored_data_info()

# Export tools for master agent
AVAILABLE_TOOLS = {
    "process_cv_pdf": process_cv_pdf_tool,
    "process_job_posting": process_job_posting_tool,
    "generate_cv_summary": generate_cv_summary_tool,
    "find_job_matches": find_job_matches_tool,
    "tailor_cv": tailor_cv_tool,
    "get_data_status": get_data_status_tool
}

async def main():
    """Main function to run the job search agent"""
    print("🚀 Job Search Agent with ChatGPT Integration started!")
    print("Features:")
    print("- Intelligent CV parsing with ChatGPT")
    print("- Smart job posting analysis")
    print("- AI-powered CV tailoring")
    print("- Advanced job matching algorithms")
    print("- Career insights and recommendations")
    print("\nListening for requests...")
    
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())