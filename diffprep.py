# Here I used Greykode's gitparser.py as a starting point and guide:
# https://github.com/graykode/commit-autosuggestions/blob/master/gitparser.py

# IMPORTS
import os
import re
import json
import random
import jsonlines
import spacy
from tqdm import tqdm 
from functools import partial
from multiprocessing.pool import Pool
import errno
from pydriller import RepositoryMining
from transformers import RobertaTokenizer
from nltk.tokenize import word_tokenize
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# CONSTANTS
repositories="repositories/python-15.txt"
repos_dir="data/raw/python"
output_dir="data/parsed/python"
output_file = os.path.join(output_dir, 'dataset.jsonl')
cb_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# LANGUAGE DETECTION
@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

detect_lang = spacy.load("en_core_web_sm")
detect_lang.add_pipe('language_detector')
english_not_detected = []


# GENERAL FUNCTIONS
def process_commits(repos_dir, repo, codebert_tokenize="code_only"):
    """
    Pre-processes commit messages in a repo.
    Tokenizes the commit messages and the diff code for each commit in a repository. 
    Determines whether diff code has been added or deleted.

    :param repo: A pre-downloaded Git repository to process.
    :param codebert_tokenize: Specifies whether to tokenize the message text and/or code in the commit message. 
                     Set to "all" (default), "code_only" or None. 
    """
    repo_path = os.path.join(repos_dir, repo)
    # Basic error handling for input params
    if codebert_tokenize not in ["code_only", "all", None]:
        raise Exception("Input parameter codebert_tokenize must be 'code_only', 'all' or None. ")
    if not os.path.exists(repo_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), repo_path)

    # Loops over all commits in the repository (Python code only)
    for commit in RepositoryMining(repo_path, only_modifications_with_file_types=".py").traverse_commits():
        # Get the commit message text
        msg = commit.msg.split("\n")[0]
        # Remove unecessary characters from the message
        cleaned_msg = re.sub(r"(\(|)#([0-9])+(\)|)", "", msg)
        # Only keep commits with commit messages in English
        langdetect_result = detect_lang(cleaned_msg)._.language
        if langdetect_result["language"] != "en":
            english_not_detected.append(cleaned_msg)
            continue
        # Tokenizing the commit message text
        if codebert_tokenize == "all":
            # Tokenize the commit message with a CodeBERT tokenizer
            msg_tokens = cb_tokenizer.tokenize(cleaned_msg)
        else: 
            # Only perform simple NLTK tokenization
            msg_tokens = word_tokenize(cleaned_msg)

        # Only keep messages that are over a certain length, so that the model has enough text to train on
        if len(msg_tokens) < 3:
            continue
        # Only keep commits with less than four file changes
        if len(commit.modifications) > 3:
            continue
        # Discard commits that reference an issue, e.g. "#454123" or "gh-24558"
        filter_expr = re.compile(r'(#|gh-)[0-9]{3,}')
        matches = list(filter(filter_expr.match, msg_tokens)) # Read Note
        if any(matches):
            continue
        
        # Looping over the file modifications in the commit
        for mod in commit.modifications:
            # FILTERING
            if not (mod.old_path and mod.new_path):
                continue
            if os.path.splitext(mod.new_path)[1] != ".py": # Python code only (excluding notebooks)
                continue
            if (not mod.diff_parsed["added"]) or (not mod.diff_parsed["deleted"]): # Only added and deleted code
                continue
            
            # Keep track of added and deleted code in the file
            added, deleted = [], []

            # Looping over added lines of code
            for _, code in mod.diff_parsed["added"]:
                if codebert_tokenize == "all" or codebert_tokenize == "code_only":
                    # Tokenizing the added line of code with a CodeBERT tokenizer
                    code = cb_tokenizer.tokenize(code)
                added.extend(code)

            # Looping over deleted lines of code
            for _, code in mod.diff_parsed["deleted"]:
                if codebert_tokenize == "all" or codebert_tokenize == "code_only":
                    # Tokenizing the deleted line of code with a CodeBERT tokenizer
                    code = cb_tokenizer.tokenize(code)
                deleted.extend(code)
            
            # Check to see if the source length is a max of 256 tokens. 
            # (Long messages make fine-tuning difficult)
            if added and deleted and (len(added) + len(deleted) <= 256):
                with jsonlines.open(output_file, mode="a") as writer:
                    writer.write({
                            "commit_message": msg_tokens,
                            "added": added,
                            "deleted": deleted
                            })

# Saves parsed (tokenized) code to json lines (.jsonl)
def create_jsonl_dataset(lines, purpose):
    saved_path = os.path.join(output_dir, purpose)
    for line in lines:
        with jsonlines.open(f"{saved_path}.jsonl", mode="a") as dataset:
            dataset.write(line)


# RUNNING THE PRE-PROCESSING STEPS ON THE DOWNLOADED REPOSITORIES
repos = set()
with open(repositories, encoding="utf-8") as f:
    for _, line in enumerate(f):
        line = line.strip()
        repos.add(line.replace('https://github.com/', ''))

# Ensure that the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process the Git repositories with multiprocessing
func = partial(process_commits, repos_dir=repos_dir, codebert_tokenize="code_only")
print(repos)
with Pool(processes=8) as pool:
    with tqdm(total=len(repos)) as pbar:
        for i, _ in tqdm(enumerate(pool.imap_unordered(func, repos))):
            pbar.update()

# # Process the Git repositories sequentially
# for i, repo in enumerate(repos): 
#     print(f"Processing repo number {i + 1} out of {len(repos)} ({repo})")
#     try:
#         process_commits(repos_dir, repo, codebert_tokenize="code_only")
#     except FileNotFoundError as err:
#         print(err)
#         continue

# Printing some negative samples from language detection task
print(f"Some examples of words that were labelled as not English: ", english_not_detected[:30])
print(f"Number of words labelled as not English: {len(english_not_detected)}\n\n")

# Load the processed commits from the dataset.jsonl file to a single list
data = []
with open(output_file, encoding="utf-8") as dataset_file:
    for line in dataset_file:
        line = line.strip()  # Remove unecessary whitespace
        data.append(json.loads(line))

# Creating train, validation and test datasets by shuffling and splicing the list
# Adding a seed to the randomizer keeps the split consistent over multiple trials
random.seed(42)
random.shuffle(data)
create_jsonl_dataset(data[:int(len(data) * 0.9)], purpose='train')
create_jsonl_dataset(data[int(len(data) * 0.9):int(len(data) * 0.95)], purpose='valid')
create_jsonl_dataset(data[int(len(data) * 0.95):], purpose='test')

print("\nShowing some examples of training data: ")
for data_dict in data[:int(len(data) * 0.9)][:10]:
    print(data_dict)