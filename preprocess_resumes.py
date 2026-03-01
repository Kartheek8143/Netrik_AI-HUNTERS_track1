import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import sys

def get_stopwords_and_lemmatizer():
    # Attempt to use NLTK if corpora is already available locally
    try:
        nltk.data.find('corpora/stopwords')
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        print(f"NLTK stopwords issue: {e}. Using fallback static list.")
        stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                      "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
                      "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
                      "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                      "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
                      "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                      "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
                      "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
                      "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
                      "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
                      "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
                      "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

    try:
        nltk.data.find('corpora/wordnet')
        nltk_lemmatizer = WordNetLemmatizer()
        def lemmatize_func(word):
            return nltk_lemmatizer.lemmatize(word)
    except Exception as e:
        print(f"NLTK wordnet issue: {e}. Using fallback simple lemmatizer.")
        def lemmatize_func(word):
            # Extremely simple fallback lemmatization (removes trailing 's' except for specific words)
            if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
                return word[:-1]
            return word

    return stop_words, lemmatize_func


hr_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HR agent'))
if hr_agent_path not in sys.path:
    sys.path.append(hr_agent_path)

try:
    from track2_hr_agent_template import Candidate
except ImportError:
    print(f"Warning: Could not import Candidate from {hr_agent_path}. Defining it locally.")
    from dataclasses import dataclass, field
    from typing import List
    @dataclass
    class Candidate:
        candidate_id: str
        name: str
        email: str
        resume_text: str
        skills: List[str] = field(default_factory=list)
        experience_years: float = 0.0
        match_score: float = 0.0
        status: str = "applied"

PRESERVED_MAPPINGS = {
    # Variations -> Standardized
    "restful apis": "rest_api_preserved",
    "restful api": "rest_api_preserved",
    "rest apis": "rest_api_preserved",
    "rest api": "rest_api_preserved",
    "machine learning": "machine_learning_preserved",
    "ml": "machine_learning_preserved",
    "artificial intelligence": "ai_preserved",
    "ai": "ai_preserved",
    "deep learning": "deep_learning_preserved",
    "dl": "deep_learning_preserved",
    "natural language processing": "nlp_preserved",
    "nlp": "nlp_preserved",
    "ui/ux": "uiux_preserved",
    
    # Preserved Tokens
    "c++": "cpp_preserved",
    "c#": "csharp_preserved",
    ".net": "dotnet_preserved",
    "node.js": "nodejs_preserved",
    "vue.js": "vuejs_preserved",
    "react.js": "react_preserved",
    "ci/cd": "cicd_preserved",
    "aws": "aws_preserved",
    "sql": "sql_preserved",
    "docker": "docker_preserved",
    "python": "python_preserved",
}

REVERSE_MAPPINGS = {
    "cpp_preserved": "c++",
    "csharp_preserved": "c#",
    "dotnet_preserved": ".net",
    "nodejs_preserved": "node.js",
    "vuejs_preserved": "vue.js",
    "react_preserved": "react.js",
    "cicd_preserved": "ci/cd",
    "rest_api_preserved": "rest api",
    "machine_learning_preserved": "machine learning",
    "ai_preserved": "ai",
    "deep_learning_preserved": "deep learning",
    "nlp_preserved": "nlp",
    "uiux_preserved": "ui/ux",
    "aws_preserved": "aws",
    "sql_preserved": "sql",
    "docker_preserved": "docker",
    "python_preserved": "python",
}

def mask_preserved(text: str) -> str:
    # Sort keys by length descending to match longest phrases first
    sorted_keys = sorted(PRESERVED_MAPPINGS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        val = PRESERVED_MAPPINGS[key]
        if bool(re.search(r'[a-z0-9]$', key)):
            pattern = r'\b' + re.escape(key) + r'\b'
        else:
            pattern = r'(?<!\w)' + re.escape(key) + r'(?!\w)'
        text = re.sub(pattern, val, text)
    return text

def unmask_preserved(text: str) -> str:
    for masked, original in REVERSE_MAPPINGS.items():
        text = text.replace(masked, original)
    return text

def clean_text(text: str, stop_words, lemmatize_func) -> str:
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Mask variations and preserved terms
    text = mask_preserved(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove Email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove Phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numeric-only tokens
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Token processing
    tokens = text.split()
    
    clean_tokens = []
    for t in tokens:
        if t not in stop_words:
            if t in REVERSE_MAPPINGS:
                # If it's a preserved token, do not lemmatize
                clean_tokens.append(t)
            else:
                lemma = lemmatize_func(t)
                clean_tokens.append(lemma)
                
    cleaned = " ".join(clean_tokens)
    
    # Unmask preserved terms
    cleaned = unmask_preserved(cleaned)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def preprocess_resume_dataset(csv_path: str):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    stop_words, lemmatize_func = get_stopwords_and_lemmatizer()
    
    print("Processing resumes...")
    candidates = []
    
    for idx, row in df.iterrows():
        # Identify target job description and other relevant text columns
        parts = []
        if pd.notna(row.get('Target_Job_Description')):
            parts.append(str(row['Target_Job_Description']))
            
        if pd.notna(row.get('Skills')):
            parts.append(str(row['Skills']))
            
        if pd.notna(row.get('Experience_Years')):
            parts.append(f"{row['Experience_Years']} years experience")
            
        if pd.notna(row.get('Certifications')):
            parts.append(str(row['Certifications']))
            
        if pd.notna(row.get('Current_Job_Title')):
            parts.append(str(row['Current_Job_Title']))
            
        if pd.notna(row.get('Degrees')):
            parts.append(str(row['Degrees']))
            
        raw_resume = " ".join(parts)
        cleaned_text = clean_text(raw_resume, stop_words, lemmatize_func)
        
        # Store in dataframe
        df.at[idx, 'cleaned_resume_text'] = cleaned_text
        
        # Candidate object creation
        candidate_id = f"C{idx+1:04d}"
        name = str(row['Name']) if pd.notna(row.get('Name')) else f"Candidate {idx+1}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        
        c = Candidate(
            candidate_id=candidate_id,
            name=name,
            email=email,
            resume_text=cleaned_text
        )
        
        if pd.notna(row.get('Experience_Years')):
            try:
                c.experience_years = float(row['Experience_Years'])
            except ValueError:
                pass

        candidates.append(c)
        
    # Save the modified dataframe
    output_path = "cleaned_" + os.path.basename(csv_path)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset with 'cleaned_resume_text' column to {output_path}")
        
    print(f"\nTotal resumes processed: {len(candidates)}")
    print("\nSample Cleaned Resume (Candidate 1):")
    print(df.iloc[0]['cleaned_resume_text'])
    
    return candidates

if __name__ == "__main__":
    csv_file = "resume_dataset_1200.csv"
    if os.path.exists(csv_file):
        preprocess_resume_dataset(csv_file)
    else:
        print(f"File {csv_file} not found locally.")
