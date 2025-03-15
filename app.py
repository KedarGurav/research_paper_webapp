from flask import Flask, render_template, request, session, redirect, url_for
from services import get_papers, generate_summary, chat_with_ai
import os
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


load_dotenv()

import secrets



app = Flask(__name__)
app.secret_key = secrets.token_hex(24)

@app.context_processor
def utility_processor():
    return dict(generate_summary=generate_summary)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.clear()
        topic = request.form.get('topic')
        session['topic'] = topic  # Store topic in session
        papers = get_papers(topic)
        if not papers:
            return render_template('index.html', error="No papers found. Please try a different topic.")
        session['papers'] = papers
        return redirect(url_for('results'))
    return render_template('index.html')

@app.route('/results')
def results():
    papers = session.get('papers', {})
    return render_template('results.html', papers=papers)


# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

# Apply rate limiting to chat endpoint
@app.route('/chat', methods=['POST'])
@limiter.limit("3 per minute")
def chat():
    user_message = request.form.get('message')
    papers = session.get('papers', {})
    
    if not user_message or not papers:
        return redirect(url_for('index'))
    
    chat_history = session.get('chat_history', [])
    chat_history.append({'role': 'user', 'content': user_message})
    
    try:
        ai_response = chat_with_ai(user_message, papers)
        chat_history.append({'role': 'assistant', 'content': ai_response})
        session['chat_history'] = chat_history
    except Exception as e:
        print(f"Error in AI chat: {str(e)}")
        chat_history.append({'role': 'assistant', 'content': "Sorry, I'm having trouble processing that request."})
        session['chat_history'] = chat_history
    
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)