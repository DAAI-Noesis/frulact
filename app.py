from flask import Flask, render_template, request, jsonify, send_from_directory
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
    api_key="cbaedf7c466348a88acd1fa8a8cfb698",
    api_version="2024-12-01-preview",  # Updated to a stable API version
    azure_endpoint="https://csa-openai-dev.openai.azure.com/",
)

cv_context = []

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
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o") # Or your specific model
                num_tokens = len(encoding.encode(cv_text))
                print(f"--- Uploaded {file.filename}: Approx. {num_tokens} tokens extracted. ---")
            except Exception as token_error:
                print(f"Could not count tokens for {file.filename}: {token_error}")
            # ------------------------------------

            # Store the CV text with metadata
            cv_data = {
                'filename': file.filename,
                'unique_filename': unique_filename,
                'content': cv_text,
                'url': file_url  # Store the complete URL
            }
            
            # Debug print to verify CV data
            print("\nStored CV Data:")
            print(f"Filename: {cv_data['filename']}")
            print(f"URL: {cv_data['url']}")
            print(f"Content length: {len(cv_data['content'])}")
            
            cv_context.append(cv_data)

            return jsonify({
                'status': 'analyzing',
                'message': 'Analyzing CVs...',
                'totalCVs': len(cv_context),
                'fileUrl': file_url
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF or TXT file'}), 400
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_role = request.json.get('requirements', '')
        
        if not cv_context:
            return jsonify({'error': 'No CVs uploaded to analyze'}), 400
        
        # Verify that we have URLs for all CVs
        missing_url_files = [cv['filename'] for cv in cv_context if not cv.get('url')]
        if missing_url_files:
            print(f"Error: Missing URLs for files: {missing_url_files}")
            return jsonify({'error': f'Missing URLs for some files: {", ".join(missing_url_files)}'}), 500
            
        # Debug prints
        print("\n=== Debug: CV Context ===")
        for cv in cv_context:
            print(f"Filename: {cv['filename']}")
            print(f"Unique Filename: {cv['unique_filename']}")
            print(f"URL: {cv['url']}")
            print(f"Content length: {len(cv['content'])}")
            print("------------------------")
        
        # Create a clear and prominent CV summary with URLs
        cv_summary = "LISTA DE CVs E URLs (IMPORTANTE - INCLUIR NA ANÁLISE):\n\n"
        for idx, cv in enumerate(cv_context, 1):
            cv_summary += f"{idx}. {cv['filename']}\n   URL: {cv['url']}\n\n"
        
        # Create the detailed CV text list with prominent URL placement
        cv_list_text = "\n=== DETALHES DOS CVs ===\n\n"
        for cv in cv_context:
            cv_list_text += f"### CV: {cv['filename']} ###\n"
            cv_list_text += f"URL DO CV: {cv['url']}\n\n"
            cv_list_text += f"Conteúdo do CV:\n{cv['content']}\n"
            cv_list_text += "### Fim do CV ###\n\n"
        
        analysis_prompt = f"""
        INSTRUÇÕES IMPORTANTES:
        - Para cada CV analisado, você DEVE incluir o URL completo do CV no início da análise
        - Os URLs abaixo são essenciais e devem ser copiados exatamente como apresentados
        
        {cv_summary}

        Requisitos da Vaga:
        {job_role}

        Por favor, analise os seguintes CVs e:
        1. Avalie a percentagem de correspondência de cada candidato para a vaga
        2. Liste as principais competências e experiências correspondentes
        3. Identifique lacunas de competências
        4. Forneça uma classificação final dos candidatos
        5. Dê uma breve justificação para a classificação

        FORMATO OBRIGATÓRIO para cada análise:

        =====================================
        Análise para: [nome do arquivo]
        Link para Download: [url completo do CV]
        Percentagem de Correspondência: XX%
        
        Competências Correspondentes:
        - ...
        
        Lacunas Identificadas:
        - ...
        
        Justificação:
        ...
        =====================================

        {cv_list_text}
        """
        
        print("\n=== Debug: URLs in Context ===")
        for cv in cv_context:
            print(f"URL for {cv['filename']}: {cv['url']}")
        
        # Call Azure OpenAI with improved system message
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Your Azure OpenAI deployment name
            messages=[
                {"role": "system", "content": """You are an expert HR analyst specializing in CV evaluation and job matching.
                CRITICAL REQUIREMENTS:
                1. You MUST answer in Portuguese language (Portugal variant)
                2. You MUST include the COMPLETE URL at the start of each CV analysis
                3. You MUST follow the exact format provided
                4. You MUST copy and paste the URLs exactly as they appear in the input
                5. DO NOT modify, shorten, or omit any URLs
                
                For each CV analysis, start with:
                =====================================
                Análise para: [nome do arquivo]
                Link para Download: [url completo do CV]
                """},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        analysis = response.choices[0].message.content
        
        # Verify URLs are in the response
        print("\n=== Debug: Checking URLs in Response ===")
        missing_urls = []
        for cv in cv_context:
            if cv['url'] not in analysis:
                missing_urls.append(cv['filename'])
                print(f"WARNING: URL for {cv['filename']} not found in response")
        
        if missing_urls:
            print(f"Some URLs are missing from the response: {missing_urls}")
            # Attempt to fix missing URLs by regenerating the response
            return jsonify({'error': 'Algumas URLs estão faltando na análise. Por favor, tente novamente.'}), 500
        
        # --- Log Analyze Token Usage ---
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            print(f"--- Analyze Call Token Usage ---")
            print(f"   Prompt Tokens (Input): {prompt_tokens}")
            print(f"   Completion Tokens (Output): {completion_tokens}")
            print(f"   Total Tokens: {total_tokens}")
            print(f"------------------------------")
        else:
            print("--- Analyze Call: Usage data not available in response. ---")
        # -------------------------------

        # Check if the model is incorrectly claiming no URLs exist
        if "não contêm urls" in analysis.lower() or "sem urls" in analysis.lower() or "sem quaisquer urls" in analysis.lower():
            print("Model incorrectly claiming no URLs exist")
            return jsonify({'error': 'O modelo está ignorando os URLs fornecidos. Por favor, tente novamente.'}), 500
        
        return jsonify({'analysis': analysis})
    
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return jsonify({'error': f'Error analyzing CVs: {str(e)}'}), 500

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
            model="gpt-4o-mini",  # Your Azure OpenAI deployment name
            messages=[
                {"role": "system", "content": """You are a helpful assistant that answers questions about CV content. Provide accurate and concise answers based on the CV information provided.
                CRITICAL INSTRUCTIONS:
                1. You MUST answer in Portuguese language (Portugal variant).
                2. When referring to information from a specific CV, you MUST mention both the filename AND include the URL as a clickable HTML link.
                3. Format URLs as HTML links using: <a href="URL">nome do ficheiro</a>
                4. Example: "According to the CV <a href="http://example.com/cv.pdf">example.pdf</a>, the candidate has experience in..."
                5. NEVER modify or shorten URLs, use them exactly as provided in the context.
                """},
                {"role": "user", "content": chat_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        # --- Log Chat Token Usage ---
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            print(f"--- Chat Call Token Usage ---")
            print(f"   Question: {question[:50]}...") # Log part of the question for context
            print(f"   Prompt Tokens (Input): {prompt_tokens}")
            print(f"   Completion Tokens (Output): {completion_tokens}")
            print(f"   Total Tokens: {total_tokens}")
            print(f"---------------------------")
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
