from flask import Flask, render_template, request
import requests
import json
import os
from nbconvert import ScriptExporter
import shutil
from transformers import GPT2LMHeadModel, GPT2Tokenizer


app = Flask(__name__)

MAX_SEQUENCE_LENGTH = 900
MODEL_NAME = 'gpt2'
TOKENIZER_NAME = 'gpt2'
PAD_TOKEN = '[PAD]'
GENERATED_OUTPUT_MAX_LENGTH = 1024
CHUNK_SIZE = 1024
ALLOWED_EXTENSIONS = ['.py', '.ipynb', '. java', '. c', '. cpp', '.js', '. cs', '.php', '.rb', ' .go','.rs', '.kt', '.swift', '.m', '.h', ' .scala', '.hs', '.sh', '.bat',' .pl','.lua', '.tel','jl', 'f90', '.f95', '.f03','. sol', '. clj', '.ex','exs','.elm', '.erl', '. fs', '.fsx',' .groovy', '.lisp', '.scm', 'ml', 'mli', '.nim',' .pas','.pascal''.pp' ,' .purs', '.re', '.rei','.ts ','.tsx','v', '. vhdl', ' . vhd']


def fetch_repositories(user_url):
    username = user_url.split('/')[-1]
    response = requests.get(f"https://api.github.com/users/{username}/repos")

    if response.status_code == 200:
        repositories = json.loads(response.text)
        return repositories
    else:
        raise Exception("Failed to fetch repositories")


def preprocess_code(repository):
    repo_name = repository['name']
    clone_url = repository['clone_url']
    repo_dir = os.path.join(os.getcwd(), repo_name)

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    os.system(f"git clone {clone_url}")

    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1]

            if is_allowed_extension(file_extension):
                if file_extension == '.ipynb':
                    preprocess_jupyter_notebook(file_path)
                else:
                    preprocess_regular_file(file_path)

    return repo_dir


def preprocess_jupyter_notebook(notebook_path):
    exporter = ScriptExporter()
    output, _ = exporter.from_filename(notebook_path)

    script_path = os.path.splitext(notebook_path)[0] + '.py'
    with open(script_path, 'w') as file:
        file.write(output)

    os.remove(notebook_path)

def preprocess_regular_file(file_path):
    pass


def is_allowed_extension(file_extension):
    return file_extension in ALLOWED_EXTENSIONS


def evaluate_complexity(code, model, tokenizer):
    complexity_score = 0

    if len(code) > MAX_SEQUENCE_LENGTH:
        code = code[:MAX_SEQUENCE_LENGTH]

    print(f"Code: {code}")

    inputs = tokenizer.encode_plus(code, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print(f"Input IDs: {input_ids}")
    print(f"Attention Mask: {attention_mask}")

    try:
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=GENERATED_OUTPUT_MAX_LENGTH,pad_token_id=tokenizer.eos_token_id,padding=True)

        print(f"Output: {output}")

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Decoded Output: {decoded_output}")

        complexity_score += len(decoded_output)

    except Exception as e:
        print(f"Error occurred during code evaluation: {e}")

    return complexity_score


def collect_code_files(directory):
    code_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1]

            if is_allowed_extension(file_extension):
                code_files.append(file_path)
    return code_files


def find_most_complex_repository(user_url):
    repositories = fetch_repositories(user_url)
    most_complex_repo = None
    max_complexity_score = float('-inf')

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    for repository in repositories:
        try:
            repo_dir = preprocess_code(repository)
            code_files = collect_code_files(repo_dir)

            complexity_score = 0

            for file_path in code_files:
                with open(file_path, 'r') as file:
                    code = file.read()
                    complexity_score += evaluate_complexity(code, model, tokenizer)

            if complexity_score > max_complexity_score:
                max_complexity_score = complexity_score
                most_complex_repo = repository

            shutil.rmtree(repo_dir)

        except Exception as e:
            print(f"Error occurred during repository processing: {e}")

    return most_complex_repo


def generate_gpt_analysis(repository, generated_output):
    analysis = f"This repository has been identified as the most technically complex based on its code complexity and structure. The code within the repository demonstrates advanced techniques and algorithms, requiring a deep understanding of the subject matter. It contains intricate implementations, optimal performance optimizations, and well-designed architecture. Overall, this repository showcases the author's exceptional coding skills and expertise in the field.\n\nGenerated Output:\n{generated_output}"
    return analysis


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_url = request.form['user_url']
        most_complex_repo = find_most_complex_repository(user_url)
        analysis = generate_gpt_analysis(most_complex_repo, generated_output='')
        return render_template('result.html', repository=most_complex_repo, analysis=analysis)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')

