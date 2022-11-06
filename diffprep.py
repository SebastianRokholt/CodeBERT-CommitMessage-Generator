# Here I used Greykode's gitparser.py as a starting point and guide:
# https://github.com/graykode/commit-autosuggestions/blob/master/gitparser.py

from pydriller import RepositoryMining
from transformers import RobertaTokenizer
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import re
import json
import random
import jsonlines
from functools import partial

# Constants
repositories="repositories/python-50.txt"
repos_dir="data/raw/python"
output_dir="data/parsed/python"
output_file = os.path.join(output_dir, 'dataset.jsonl')


def jobs(repo):
    """
    Tokenizes the commit messages and the diff code for each commit in a repository. 
    Determines whether diff code has been added or deleted.
    """
    repo_path = os.path.join(repos_dir, repo)
    if os.path.exists(repo_path):
        # Loops over all commits in the repository (Python code only)
        for commit in RepositoryMining(repo_path, only_modifications_with_file_types="py").traverse_commits():
            # Remove unecessary characters
            cleaned_msg = clean_msg(commit.msg)
            # Tokenize the commit message with a CodeBERT tokenizer
            tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
            msg_tokens = tokenizer.tokenize(cleaned_msg)

            # Only keep messages that are over a certain length, so that the model has something to train on
            if len(msg_tokens) > 128:
                continue
            
            # Loop over the modifications in the commit
            for mod in commit.modifications:
                # Determine whether the code has been added, or deleted.
                if not (mod.old_path and mod.new_path):
                    continue
                if os.path.splitext(mod.new_path)[1] not in "py":
                    continue
                if not mod.diff_parsed["added"]:
                    continue
                if not mod.diff_parsed["deleted"]:
                    continue

                added, deleted = [], []

                for _, code in mod.diff_parsed["added"]:
                    # Tokenize the diff code with CodeBERT tokenizer
                    added.extend(tokenizer.tokenize(code))

                for _, code in mod.diff_parsed["deleted"]:
                    # Tokenize the diff code with CodeBERT tokenizer
                    deleted.extend(tokenizer.tokenize(code))
                
                # Check to see if the source length is smaller than 256 chars. 
                # Messages that are too long will be difficult to train on.
                if added and deleted and len(added) + len(deleted) <= 253:
                    with jsonlines.open(output_file, mode="a") as writer:
                        writer.write({
                                "msg": msg_tokens,
                                "added": added,
                                "deleted": deleted
                                })

# Saves parsed (tokenized) code to json lines (.jsonl)
def save_as_jsonl(lines, path, mode):
    saved_path = os.path.join(path, mode)
    for line in lines:
        with jsonlines.open(f"{saved_path}.jsonl", mode="a") as writer:
            writer.write(line)

def clean_msg(message):
    """
    Performs a regex on the commit message to remove unnecessary characters
    """
    msg = message.split("\n")[0]
    msg = re.sub(r"(\(|)#([0-9])+(\)|)", "", msg)
    return msg

def main():
    repos = set()
    with open(repositories, encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            repos.add(line.replace('https://github.com/', ''))

    # Ensure that the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    func = partial(jobs)
    # Multiprocessing with 4 threads
    with Pool(processes=4) as pool:
        # Using tqdm to create a neat progress bar while processing data
        with tqdm(total=len(repos)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(func, repos))):
                pbar.update()

    data = []
    with open(output_file, encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            data.append(json.loads(line))

    # Creating train, validation and test datasets
    random.shuffle(data)
    n_data = len(data)
    save_as_jsonl(
        data[:int(n_data * 0.9)],
        path=output_dir, mode='train'
    )
    save_as_jsonl(
        data[int(n_data * 0.9):int(n_data * 0.95)],
        path=output_dir, mode='valid'
    )
    save_as_jsonl(
        data[int(n_data * 0.95):],
        path=output_dir, mode='test'
    )
