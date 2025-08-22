# ResumeMatch

**ResumeMatch** is a tool to help recruiters and job seekers analyze resume-to-job description matching.  

## Features
- Upload job descriptions and candidate resumes
- Compute match scores for Hard Skills, Soft Skills, Education, and Experience
- Visualize results with progress bars and score cards
- Identify top missing skills per candidate

## How to Use
1. Go to the **Job Seeker** tab to see your resume match against a job.
2. Go to the **Recruiter** tab to upload multiple resumes and rank candidates.
3. Adjust the **threshold slider** to change matching sensitivity.

## Gemini API Configuration
ResumeMatch uses the **Gemini API** for advanced embedding-based similarity scoring.  
To configure:

1. Sign up or log in to your Gemini account to obtain your **API key**.
2. In your environment (local `.env` or Colab), set the key:

or in Python:
```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```
3. ResumeMatch automatically reads this key when computing embeddings.
4. Ensure your network allows outbound API calls to Gemini.                        


                 
	
                                                                                      

                                                                                     
