""" 
Uses Ollama and Gwen3+RAG for automatic resume personalization given a job posting
Could use other frameworks but im on AMD GPU and windows so lack of support.
"""

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
from retrieval import retrieve_chunks_for_query
import fitz
import argparse
from ollama import chat
from ollama import ChatResponse
from playwright.sync_api import sync_playwright

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "Qwen/Qwen3-4B-Instruct-2507"
# tokenizer_llm = AutoTokenizer.from_pretrained(model_name)
# tokenizer_llm.pad_token = tokenizer_llm.eos_token
# model_llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device)
# model_llm.eval()

def scrape_with_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        # Wait for content to load if needed: page.wait_for_selector('selector')
        text = page.content()
        browser.close()
        return text
    
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def build_requirement_extraction_prompt(full_text):
    prompt = f"""
    You are an expert job analyst. Extract and list the job requirements, qualifications and anything relevant from the following job posting text.

    Job Posting Text:
    {full_text}

    """
    return prompt.strip()

def build_resume_prompt(job_description, retrieved_chunks, transcript):
    context = "\n\n---\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    prompt = f"""
    You are an expert resume writer. Using the provided job description and relevant project, code snippets, and transcript, create a personalized resume that highlights the 
    candidate’s skills and experience tailored to the role. Structure each bullet point using the format: Action verb + task/project + outcome, emphasizing quantifiable outcomes. 
    Make sure the points are have relevant keywords from the job description and importantly that they align with the candidate's experience.
    The points should be concise, impactful, and demonstrate the candidate's fit for the role.
    
    Job Description:
    {job_description}

    Relevant Projects and Code Snippets:
    {context}
    
    Transcript:
    {transcript}

    Generate the resume below:
    """
    return prompt.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume generator using Gwen3 and RAG via StarEnCoder embeddings")
    parser.add_argument("--url", help="URL of the job posting")
    parser.add_argument("--transcript", help="Path to the transcript PDF")
    parser.add_argument("--retrieve", help="Number of relevant code snippets to retrieve", type=int, default=5)
    args = parser.parse_args()
    
    # Extract job description using LLM
    print("Extracting job description...")
    job_description = scrape_with_playwright(args.url)
    print("Generating job description...")
    job_description_response: ChatResponse = chat(model='gwen3-4b-instruct-ctx-8k', messages=[
    {
        'role': 'user',
        'content': job_description,
    },
    ])

    # Extract text from transcript PDF
    transcript_pdf_path = args.transcript
    transcript = extract_text_from_pdf(transcript_pdf_path)

    # Build resume prompt and receive resume
    print("Building resume prompt...")
    prompt = build_resume_prompt(job_description_response.message.content, 
                                 retrieve_chunks_for_query(job_description_response.message.content, 
                                                           './embeds/faiss.index', 
                                                           './embeds/metadata.txt', 
                                                           k=args.retrieve), 
                                 transcript)
    print("Generating personalized resume...")
    response: ChatResponse = chat(model='gwen3-4b-instruct-ctx-8k', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    # inputs = tokenizer_llm.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer_llm([inputs], return_tensors="pt").to(device)
    #
    # # conduct text completion
    # generated_ids = model_llm.generate(
    #     **model_inputs,
    #     max_new_tokens=16384
    # )
    #
    # resume = tokenizer_llm.decode(generated_ids[0][len(model_inputs.input_ids[0]):].tolist(), skip_special_tokens=True)

    print("Generated Resume:")
    print(response.message.content)