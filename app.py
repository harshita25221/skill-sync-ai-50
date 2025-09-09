import docx
import spacy 
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from keybert import KeyBERT
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from flask_cors import CORS
from rapidfuzz import process, fuzz  

app = Flask(__name__)
CORS(app)

# Load NLP models only when needed to reduce memory usage
nlp = None
kw_model = None

def load_nlp_models():
    global nlp, kw_model
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    if kw_model is None:
        kw_model = KeyBERT()

import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")


skills_df = pd.read_csv("merged_skills.csv")
GLOBAL_SKILLS = set(skills_df["skill"].dropna().str.lower().str.strip())



def extract_text_from_docx(file):
    try:
        # Limit memory usage by processing in chunks
        doc = docx.Document(file)
        # Only process a reasonable number of paragraphs
        max_paragraphs = 500
        paragraphs = [para.text for para in doc.paragraphs[:max_paragraphs]]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        raise

def extract_text_from_pdf(file):
    try:
        text = ""
        file_content = BytesIO(file.read())
        with pdfplumber.open(file_content) as pdf:
            # Limit to a reasonable number of pages
            max_pages = 30
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    text += "\n[Document truncated due to length]\n"
                    break
                extracted = page.extract_text() or ""
                # Limit text per page to prevent memory issues
                if len(extracted) > 5000:
                    extracted = extracted[:5000] + "\n[Page content truncated]\n"
                text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+"," ", text)
    return text



def get_keywords(text, num_keywords=20):
    load_nlp_models()  # Ensure model is loaded
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), 
        stop_words='english',
        top_n=num_keywords
    )
    return [kw[0] for kw in keywords]

def extract_spacy_skills(text):
    load_nlp_models()  # Ensure model is loaded
    doc = nlp(text)
    skills = set()
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
            word = token.text.strip()
            if word.lower() not in STOP_WORDS:
                if word[0].isupper() or re.search(r"[A-Za-z0-9\+\#]", word):
                    skills.add(word)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE']:
            skills.add(ent.text.lower())        
    return list(skills)

def normalize_skills_with_fuzzy(extracted_skills, global_skills, threshold=85):
    """ ✅ Use rapidfuzz to map extracted skills to closest taxonomy skills """
    normalized = set()
    for skill in extracted_skills:
        match = process.extractOne(skill, global_skills, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:  
            normalized.add(match[0])  
    return normalized

def extract_multiword_skills(text, global_skills):
    found = set()
    for skill in global_skills:
        if " " in skill and skill in text.lower():
            found.add(skill)
    return found



def get_combined_skills(text):
    # Limit the number of keywords to extract to reduce processing time
    kw_skills = set(get_keywords(text, num_keywords=10))
    spacy_skills = set(extract_spacy_skills(text))
    multiword_skills = extract_multiword_skills(text, GLOBAL_SKILLS)

    # Limit the total number of skills to process
    all_extracted = {s.lower().strip() for s in kw_skills.union(spacy_skills, multiword_skills)}
    if len(all_extracted) > 50:
        all_extracted = set(list(all_extracted)[:50])
   
    filtered = normalize_skills_with_fuzzy(all_extracted, GLOBAL_SKILLS)

    return filtered



# -------------------- Scoring --------------------
def get_skills_and_score(resume_text, job_description, alpha=0.3):
    # Limit text length to reduce processing time
    resume_text = resume_text[:10000] if len(resume_text) > 10000 else resume_text
    job_description = job_description[:10000] if len(job_description) > 10000 else job_description
    
    resume_skills = set(get_combined_skills(resume_text))
    job_req_skills = set(get_combined_skills(job_description))

    # Limit to most important skills
    resume_skills = {s for s in resume_skills if s not in STOP_WORDS and len(s) > 2}
    job_req_skills = {s for s in job_req_skills if s not in STOP_WORDS and len(s) > 2}
    
    # Limit the number of skills to process
    if len(resume_skills) > 30:
        resume_skills = set(list(resume_skills)[:30])
    if len(job_req_skills) > 30:
        job_req_skills = set(list(job_req_skills)[:30])

    if not resume_skills or not job_req_skills:
        return 0.0, [], [], 0.0

    overlap = len(resume_skills & job_req_skills) / len(job_req_skills)

    # Only compute cosine similarity if there are enough skills
    if len(resume_skills) > 3 and len(job_req_skills) > 3:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(
            [" ".join(resume_skills), " ".join(job_req_skills)]
        )
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    else:
        cosine_sim = overlap  # Fallback to overlap score

    final_score = (alpha * cosine_sim + (1 - alpha) * overlap) * 100

    missing_skills = sorted(list(job_req_skills - resume_skills))
    highlighted_skills = sorted(list(job_req_skills & resume_skills))

    return final_score, missing_skills, highlighted_skills, cosine_sim



def generate_ai_text(prompt: str) -> str:
    # Only initialize OpenAI API key when needed
    if not openai.api_key and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Use a smaller model to reduce resource usage
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model to reduce resource usage
            messages=[
                {"role":"system","content":"You are an AI-powered career coach that analyzes resumes and job descriptions, rewrites resumes for better alignment, crafts tailored cover letters, and provides suggestions to maximize a candidate's chances of getting hired."},
                {"role": "user", "content": prompt}
            ], 
            max_tokens=400,  # Reduced token count
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "AI text generation is currently unavailable. Please try again later."

def generate_tailored_resume(resume_text, job_description):
    prompt = f"""
    Resume: \n{resume_text}\n
    Job description: \n{job_description}\n
    Rewrite the resume so it better matches the job description.
    Focus on aligning skills, experience, and phrasing with the job description while keeping authenticity.
    """
    return generate_ai_text(prompt)

def generate_cover_letter(resume_text, job_description):
    prompt = f"""
    Write a professional cover letter tailored to the following job description: \n{job_description}\n
    Resume content: \n{resume_text}\n
    Make it concise, skill-focused, and role-specific."""
    return generate_ai_text(prompt)

import re 

def generate_suggestions(resume_text, job_description, cosine_sim, missing_skills):
    prompt = f"""
    You are an expert career coach. 
    A candidate has a resume and is applying for this job description.
    Their resume-Job description match score is {round(cosine_sim*100,2)}%
    Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}

    Provide a numbered list of 3-5 clear, practical suggestions.
    Each suggestion must be on a new line.
    """
    
    response_text = generate_ai_text(prompt)
    
    
    suggestions_list = [
        re.sub(r'^\d+\.\s*', '', line).strip() 
        for line in response_text.split('\n') 
        if line.strip()
    ]
    
    return suggestions_list



@app.route("/")
def index():  
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])  
def analyze():  
    try:
        resume_file = request.files.get("resume")  
        jd_file = request.files.get("job_description")
        if not resume_file or not jd_file:
            return jsonify({"error": "Please upload both resume and job description files."})  

        # Check file size to prevent large file processing
        max_file_size = 5 * 1024 * 1024  # 5MB limit
        if resume_file.content_length and resume_file.content_length > max_file_size:
            return jsonify({"error": "Resume file size exceeds 5MB limit."})
        if jd_file.content_length and jd_file.content_length > max_file_size:
            return jsonify({"error": "Job description file size exceeds 5MB limit."})

        # Process resume file
        try:
            if resume_file.filename.endswith(".docx"):
                resume_raw = extract_text_from_docx(resume_file)
            elif resume_file.filename.endswith(".pdf"):
                resume_raw = extract_text_from_pdf(resume_file)
            else:
                return jsonify({"error": "Unsupported resume file format. Please upload .docx or .pdf files only."})  
        except Exception as e:
            print(f"Error processing resume file: {e}")
            return jsonify({"error": "Could not process resume file. Please check the file format and try again."})

        # Process job description file
        try:
            if jd_file.filename.endswith(".docx"):
                jd_raw = extract_text_from_docx(jd_file)
            elif jd_file.filename.endswith(".pdf"):
                jd_raw = extract_text_from_pdf(jd_file)
            else:
                return jsonify({"error": "Unsupported job description file format. Please upload .docx or .pdf files only."})  
        except Exception as e:
            print(f"Error processing job description file: {e}")
            return jsonify({"error": "Could not process job description file. Please check the file format and try again."})

        # Limit text length to prevent memory issues
        max_text_length = 50000  # Limit text to 50K characters
        resume_raw = resume_raw[:max_text_length] if len(resume_raw) > max_text_length else resume_raw
        jd_raw = jd_raw[:max_text_length] if len(jd_raw) > max_text_length else jd_raw

        resume_clean = clean_text(resume_raw)
        jd_clean = clean_text(jd_raw)

        final_score, missing_skills, highlighted_skills, cosine_sim = get_skills_and_score(resume_clean, jd_clean)

        # Limit the number of skills returned to prevent large responses
        if len(missing_skills) > 20:
            missing_skills = missing_skills[:20]
        if len(highlighted_skills) > 20:
            highlighted_skills = highlighted_skills[:20]

        tailored_resume = generate_tailored_resume(resume_clean, jd_clean)
        cover_letter = generate_cover_letter(resume_clean, jd_clean)
        suggestions = generate_suggestions(resume_clean, jd_clean, cosine_sim, missing_skills)

        return jsonify({  
            "match_score": round(final_score,2),
            "missing_skills": missing_skills,
            "highlighted_skills": highlighted_skills,
            "tailored_resume": tailored_resume,
            "cover_letter": cover_letter,
            "suggestions": suggestions
        })
    except Exception as e:
        print(f"Unexpected error in analyze endpoint: {e}")
        return jsonify({"error": "An unexpected error occurred. Please try again later."})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
