from queue import Queue
import git
from time import sleep
import os
from threading import Thread

class RepoScraper():
    """
    "Scrapes" Git Repositories, i.e. downloads by git cloning them. 
    """
    def __init__(self, repositories, download_path):
        """
        :param repositories: The total number of Github repositories to be downloaded
        :param download_path: The path to the directory for downloading all of the scraped repositories
        """
        self.queue = Queue()  # data structure for scraping repos
        self.count = 0  # keeps track of progress
        self.total_n_repos = len(repositories)
        self.download_path = download_path

        # Multithreading with 16 threads.
        # NB! More threads => higher memory usage
        for _ in range(16):
            _thread = Thread(target=self.cloner)
            _thread.daemon = True
            _thread.start()

    def clone_repos(self, repo_gh_file_path):
        """
        :param repo_gh_file_path: The file path to the Github repository, e.g: SebastianRokholt/CodeBERT-CommitMessage-Generator
        """
        try:
            # Clone repo to download dir
            git.Repo.clone_from(
                f'https://:@github.com/{repo_gh_file_path}.git',
                f'{self.download_path}/{repo_gh_file_path}'
            )

            sleep(0.2)  # Trying to avoid blacklisting due to high page request frequency

            self.count += 1 
            print(f"Progress: {self.count} out of {self.total_n_repos} downloaded. ")
        
        # Very basic error handling
        except git.exc.GitError as e:
            print(f"The following error occurred while attempting to clone repository from https://:@github.com/{repo_gh_file_path}.git: ")
            print(e)

    def cloner(self):
        """
        Runs the crawling method (clone_repos) on the repositories according to the crawl queue. 
        Called by the multithreading daemon. 
        Terminates when the crawl queue is empty. 
        """
        while True:
            repo_file_path = self.queue.get()  # Retrieves the next repo from the queue
            self.clone_repos(repo_file_path)  # Runs the crawling method on the repo
            self.queue.task_done()  # Registers the task (crawling the repo) as completed

    def join_queue(self):
        """
        Blocks until all repos in the crawl queue have been gotten and processed.
        Unblocks when count of unfinished tasks drops to zero.
        """
        self.queue.join()

    def put_queue(self, repo_file_path):
        """
        Puts a repo into the crawl queue. 
        """
        self.queue.put(repo_file_path)
        


def main():
    download_path="data/raw/python"
    os.makedirs(download_path, exist_ok=True)

    # Read the repositories to crawl from file
    repositories = []
    with open("repositories/python-50.txt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            repositories.append(line.replace('https://github.com/', ''))
    print("Repositories to crawl:", repositories)

    # Instantiate the repo scraper
    scraper = RepoScraper(repositories=set(repositories), download_path=download_path)

    # Start scraping
    for repo in repositories:
        # Put each repo in the crawl queue
        scraper.put_queue(repo)
    scraper.join_queue()


if __name__ == "__main__":
    main()