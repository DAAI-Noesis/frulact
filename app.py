from flask import Flask, render_template, request, jsonify, send_from_directory, session
import os
from openai import AzureOpenAI
import PyPDF2
from dotenv import load_dotenv
import pdfplumber  
import pytesseract
from pdf2image import convert_from_path
from shutil import which
import uuid 
import tiktoken
load_dotenv()

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Changed from temp_uploads to permanent uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Azure OpenAI Configuration
client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-12-01-preview",  # Updated to a stable API version
    azure_endpoint="https://occazureopenai.openai.azure.com/",
)

cv_context = []

# Function to count tokens for a given text
def count_tokens(text, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

# Function to print session token usage
def print_session_token_usage():
    if 'token_usage' in session:
        # Calculate total CV tokens from cv_context
        total_cv_tokens = sum(cv.get('tokens', 0) for cv in cv_context)
        
        print("\n===== TOKEN USAGE SUMMARY =====")
        print(f"Total CV Tokens (from session): {session['token_usage'].get('cv_tokens', 0)}")
        print(f"Total CV Tokens (sum of all CVs): {total_cv_tokens}")
        print(f"User Question Tokens: {session['token_usage'].get('user_question_tokens', 0)}")
        print(f"AI Answer Tokens: {session['token_usage'].get('ai_answer_tokens', 0)}")
        print(f"Total Tokens: {session['token_usage'].get('total_tokens', 0)}")
        print("===============================\n")
        
        # Print individual CV token counts
        if cv_context:
            print("===== INDIVIDUAL CV TOKEN COUNTS =====")
            for i, cv in enumerate(cv_context):
                print(f"CV {i+1}: {cv.get('filename', 'Unknown')} - {cv.get('tokens', 0)} tokens")
            print(f"Total CV Tokens: {total_cv_tokens}")
            print("=====================================\n")
    else:
        print("\n===== TOKEN USAGE SUMMARY =====")
        print("No token usage data available for this session")
        print("===============================\n")

# Function to update session token usage
def update_session_token_usage(category, tokens):
    if 'token_usage' not in session:
        session['token_usage'] = {
            'cv_tokens': 0,
            'user_question_tokens': 0,
            'ai_answer_tokens': 0,
            'total_tokens': 0
        }
    
    # Only update if tokens is a positive number
    if tokens > 0:
        session['token_usage'][category] += tokens
        session['token_usage']['total_tokens'] += tokens
        session.modified = True
        return True
    return False

def extract_text_with_ocr(pdf_path):
    try:
        print("Starting OCR process...")
        
        # Check if poppler is accessible
        if which('pdftoppm') is None:
            print("Error: poppler-utils is not found in PATH")
            print("Current PATH:", os.environ['PATH'])
            print("\nInstallation instructions:")
            if os.name == 'nt':  # Windows
                print("1. Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/")
                print("2. Extract to C:\\Program Files\\poppler")
                print("3. Add C:\\Program Files\\poppler\\Library\\bin to your system PATH")
            else:  # Unix-like
                print("For Linux: sudo apt-get install poppler-utils")
                print("For macOS: brew install poppler")
            return ""

        # Test poppler installation
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            pdfinfo_from_path(pdf_path)
        except Exception as e:
            print(f"Error testing poppler installation: {str(e)}")
            return ""

        print("Converting PDF to images...")
        images = convert_from_path(
            pdf_path,
            dpi=300,  # Higher DPI for better quality
            fmt='jpeg',  # Use JPEG format
            thread_count=2,  # Use multiple threads
            poppler_path=None  # Will use PATH environment variable
        )
        print(f"Converted {len(images)} pages to images")
        
        text = ""
        # Process each page
        for i, image in enumerate(images):
            print(f"Processing page {i+1} with OCR")
            # Extract text from image
            page_text = pytesseract.image_to_string(image, lang='eng')
            if page_text.strip():
                print(f"Successfully extracted text from page {i+1}, length: {len(page_text)}")
                text += page_text + "\n"
            else:
                print(f"No text extracted from page {i+1}")
            
        if not text.strip():
            print("Warning: No text was extracted from any page")
        else:
            print(f"Total extracted text length: {len(text)}")
            
        return text
    except Exception as e:
        print(f"OCR failed with error: {str(e)}")
        print("Full error details:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        return ""

@app.route('/')
def index():
    # Clear the CV context when loading the index page
    global cv_context
    cv_context = []
    
    # Reset session token usage completely
    session['token_usage'] = {
        'cv_tokens': 0,
        'user_question_tokens': 0,
        'ai_answer_tokens': 0,
        'total_tokens': 0
    }
    session.modified = True
    
    print("\n===== NEW SESSION STARTED =====")
    print("Token usage has been reset to zero")
    print("===============================\n")
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        print(f"Received file: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            file.save(file_path)
            
            # Generate file URL - Use absolute URL with host
            file_url = f"{request.scheme}://{request.host}/uploads/{unique_filename}"
            print(f"Generated URL: {file_url}")  # Debug print
            
            cv_text = ""
            if file.filename.endswith('.pdf'):
                try:
                    print(f"Attempting to read PDF with pdfplumber: {file.filename}")
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    cv_text += page_text
                            except Exception as page_error:
                                print(f"Error on page {page_num + 1}: {str(page_error)}")
                except Exception as e:
                    print(f"pdfplumber failed: {str(e)}, trying PyPDF2")
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        if pdf_reader.is_encrypted:
                            os.remove(file_path)
                            return jsonify({'error': 'PDF is encrypted and cannot be read'}), 400
                        for page in pdf_reader.pages:
                            cv_text += page.extract_text()
                
                if not cv_text.strip():
                    cv_text = extract_text_with_ocr(file_path)
            else:  # .txt file
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    cv_text = txt_file.read()

            # --- Token Counting for CV Text ---
            cv_tokens = count_tokens(cv_text)
            print(f"--- Uploaded {file.filename}: Approx. {cv_tokens} tokens extracted. ---")
            
            # Update session token usage
            update_session_token_usage('cv_tokens', cv_tokens)
            
            # Store the CV text with metadata
            cv_data = {
                'filename': file.filename,
                'unique_filename': unique_filename,
                'content': cv_text,
                'url': file_url,  # Store the complete URL
                'tokens': cv_tokens  # Store token count
            }
            
            # Debug print to verify CV data
            print("\nStored CV Data:")
            print(f"Filename: {cv_data['filename']}")
            print(f"URL: {cv_data['url']}")
            print(f"Content length: {len(cv_data['content'])}")
            print(f"Token count: {cv_data['tokens']}")
            
            cv_context.append(cv_data)
            
            # Calculate and print total CV tokens
            total_cv_tokens = sum(cv.get('tokens', 0) for cv in cv_context)
            print(f"\n--- Total CV Tokens (sum of all CVs): {total_cv_tokens} ---")
            
            # Print session token usage
            print_session_token_usage()

            return jsonify({
                'status': 'success',
                'message': 'CV uploaded successfully',
                'totalCVs': len(cv_context),
                'fileUrl': file_url,
                'tokenCount': cv_tokens,
                'totalCVTokens': total_cv_tokens
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF or TXT file'}), 400
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        question = request.json.get('question', '')

        # Check if we have any CVs to chat about
        if not cv_context:
            return jsonify({'error': 'No CVs uploaded to chat about'}), 400

        # Count tokens for the user's question
        question_tokens = count_tokens(question)
        if question_tokens > 0:
            print(f"--- User Question Tokens: {question_tokens} ---")
            update_session_token_usage('user_question_tokens', question_tokens)
        else:
            print("--- User Question Tokens: 0 (empty or invalid question) ---")

        # Format CVs with their filenames and URLs
        cv_text_with_info = ""
        for cv in cv_context:
            # Ensure URL exists, otherwise provide a placeholder or skip
            cv_url = cv.get('url', 'URL Indisponível')
            cv_text_with_info += f"\n=== CV from file: {cv['filename']} | URL: {cv_url} ===\n"
            cv_text_with_info += f"{cv['content']}\n"
            cv_text_with_info += f"=== End of CV: {cv['filename']} ===\n"

        chat_prompt = f"""
        Conteúdo dos CVs (incluindo nome do ficheiro e URL):
        {cv_text_with_info}

        Pergunta sobre os CVs:
        {question}

        Por favor, responda à pergunta com base na informação dos CVs fornecidos.
        IMPORTANTE: Sempre que referir informação de um CV específico, mencione o nome do ficheiro E o URL completo desse CV.
        O URL deve ser formatado como um link HTML clicável usando a tag <a href="URL">nome do ficheiro</a>.
        """

        # Call Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # Your Azure OpenAI deployment name
            messages=[
                {"role": "system", "content": """You are a highly skilled recruitment assistant working for the Human Resources department of Frulact. Your primary purpose is to meticulously analyze the provided CVs and assist the HR team in the talent selection process.

                Your core tasks involve:
                1.  **In-depth CV Analysis:** Carefully examine the content of each uploaded CV.
                2.  **Candidate Comparison & Role Matching:** When asked (implicitly or explicitly) about suitability for a specific job role, compare the candidates based on their CVs against the requirements of that role. Identify the strengths and weaknesses of each candidate concerning the position.
                3.  **Recommendation & Justification:** Clearly articulate which candidate(s) appear most suitable for a given role and provide specific reasons based *only* on the information present in their CVs. Highlight key qualifications, experiences, or skills that align with the job.
                4.  **Answering Queries:** Respond accurately and professionally to any questions asked about the content of the CVs, always within the context of recruitment for Frulact.

                CRITICAL INSTRUCTIONS:
                1.  You MUST answer exclusively in Portuguese language (Portugal variant).
                2.  When referring to information from a specific CV, you MUST mention both the filename AND include the URL as a clickable HTML link using the format: <a href="URL">filename</a>. Adhere strictly to this format.
                3.  FORMAT YOUR RESPONSES IN A CLEAN, STRUCTURED WAY:
                   - Use proper Markdown formatting for headings, lists, and emphasis
                   - Organize information in clear sections with appropriate headings
                   - Use bullet points or numbered lists for multiple items
                   - Avoid unnecessary line breaks or excessive spacing
                   - Use bold text for important information or conclusions
                   - Format links properly as clickable HTML elements
                4.  NEVER include raw HTML tags in your response text except for the required link format
                5.  NEVER include unnecessary formatting characters or symbols
                """},
                {"role": "user", "content": chat_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        answer = response.choices[0].message.content

        # --- Log Chat Token Usage ---
        if response.usage:
            # Count tokens for the AI's answer
            answer_tokens = count_tokens(answer)
            if answer_tokens > 0:
                print(f"--- AI Answer Tokens: {answer_tokens} ---")
                update_session_token_usage('ai_answer_tokens', answer_tokens)
            else:
                print("--- AI Answer Tokens: 0 (empty or invalid answer) ---")
            
            print(f"--- Chat Call Token Usage ---")
            print(f"   Question: {question[:50]}...") # Log part of the question for context
            print(f"   Question Tokens: {question_tokens}")
            print(f"   Answer Tokens: {answer_tokens}")
            print(f"   Total Tokens: {question_tokens + answer_tokens}")
            print(f"---------------------------")
            
            # Print current session token usage
            print_session_token_usage()
        else:
            print("--- Chat Call: Usage data not available in response. ---")
        # ---------------------------

        return jsonify({'answer': answer})
    
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({'error': f'Error processing chat: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
