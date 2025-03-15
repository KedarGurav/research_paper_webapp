import arxiv
import nltk
from nltk.corpus import stopwords
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

nltk.download('punkt')
nltk.download('stopwords')

def get_papers(topic):
    try:
        search = arxiv.Search(
            query=topic,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = {}
        for result in arxiv.Client().results(search):
            papers[result.title] = {
                'entry_id': result.entry_id,
                'summary': result.summary,
                'pdf_url': result.pdf_url
            }
        return papers
    except Exception as e:
        print(f"Error fetching papers: {str(e)}")
        return None

def generate_summary(text):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        template = """You give concise, easy to understand summaries of research papers. Keep it under 200 words.
        Text: {text}
        Summary:"""
        prompt = PromptTemplate(input_variables=["text"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(text)
    except Exception as e:
        print(f"Summary generation error: {str(e)}")
        return "Summary unavailable at this time."

def chat_with_ai(message, papers):
    try:
        # Validate inputs
        if not message.strip() or len(message) > 500:
            return "Please provide a valid question (max 500 characters)."
            
        if not papers or len(papers) == 0:
            return "No research papers available. Please start a new search."

        # Check API key
        if not os.getenv('GOOGLE_API_KEY'):
            print("Error: Google API key not configured")
            return "Service configuration error. Please contact support."

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,  # More focused responses
            max_retries=3,    # Add retry logic
            request_timeout=30
        )

        # Create context with truncation to avoid token limits
        summaries = " ".join([paper['summary'][:1500] for paper in papers.values()])
        
        # Enhanced prompt template
        template = """You are a research assistant analyzing these papers. Follow these rules:
        1. Answer ONLY based on the provided context
        2. If unsure, say "Based on the available research, I cannot determine..."
        3. Keep answers under 300 words
        4. Use markdown for formatting when helpful
        
        Context: {summaries}
        
        Question: {message}
        
        Answer in this structure:
        - **Key Findings**: [main points]
        - **Relevant Papers**: [paper titles]
        - **Conclusion**: [summary]"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["summaries", "message"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({'summaries': summaries, 'message': message})
        
        # Post-process response
        return response.strip() or "I couldn't generate a valid response. Please try rephrasing your question."

    except Exception as e:
        print(f"Chat error: {str(e)}")
        # Return more specific error messages
        if "SAFETY" in str(e):
            return "My response was blocked for safety concerns. Please try a different question."
        if "TIMEOUT" in str(e):
            return "Request timed out. Please try again."
        return "Technical difficulty. Please try again later."