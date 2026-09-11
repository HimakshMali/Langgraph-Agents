"""
=============================================================================
GitHub API Integration Module
=============================================================================
This module handles direct HTTP communication with the GitHub REST API.
It extracts issues from target repositories, handles API pagination,
filters out pull requests, and packages the data into structured dictionaries.
=============================================================================
"""

import os
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

# =============================================================================
# 1. Environment Configuration & Secrets Management
# =============================================================================
# CONCEPT: Decoupling Configuration & Secrets
# Keeping credentials like API keys and tokens out of version control (Git)
# is a critical security practice. python-dotenv reads key-value pairs from
# a local .env file and sets them as environment variables.
load_dotenv()

OWNER: Optional[str] = os.getenv("OWNER")
REPO: Optional[str] = os.getenv("REPO")
GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")

# =============================================================================
# 2. HTTP Request Headers & API Specifications
# =============================================================================
# CONCEPT: HTTP Headers as Metadata
# Headers are key-value pairs sent alongside HTTP requests. They act as
# metadata describing the client, authentication credentials, and response formats:
#
# 1. "Accept": "application/vnd.github+json"
#    - GitHub uses custom vendor media types (vnd). This instructs GitHub's API
#      to return response data formatted according to their standard JSON schema.
#
# 2. "X-GitHub-Api-Version": "2022-11-28"
#    - Explicitly pinning the API version prevents breaking changes from affecting
#      our agent if GitHub updates default behavior in future API releases.
#
# 3. "Authorization": f"Bearer {GITHUB_TOKEN}"
#    - Authenticates the client. Authenticated requests benefit from 5,000
#      requests/hour rate limit (compared to only 60 requests/hour unauthenticated).
BASE_HEADERS: Dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}


# =============================================================================
# 3. GitHub Issue Fetching Functions
# =============================================================================

def fetch_all_issues(
    owner: Optional[str] = OWNER,
    repo: Optional[str] = REPO,
    state: str = "open"
) -> List[Dict[str, Any]]:
    """
    Fetches all issues from a GitHub repository across all pages, filtering out pull requests.

    -------------------------------------------------------------------------
    KEY CONCEPTS INVOLVED:
    -------------------------------------------------------------------------
    1. Pagination (per_page & page):
       - GitHub restricts responses to a maximum of 100 items per page.
       - We loop with `page += 1` until the API returns an empty list ([]),
         indicating we have retrieved all available records.

    2. Issues vs. Pull Requests in GitHub's REST API:
       - GitHub's backend models Pull Requests as issues with extra metadata.
       - As a result, the `/issues` endpoint returns BOTH issues and PRs!
       - Pull Requests contain a `"pull_request"` key in their JSON payload.
         Checking `"pull_request" not in item` ensures we return pure issues.

    3. HTTP Status Codes & Error Handling:
       - 200: OK / Success.
       - 401: Unauthorized (check if your GITHUB_TOKEN is valid).
       - 403: Forbidden (rate limit exceeded or insufficient permissions).
       - 404: Not Found (repository does not exist or is private).
    -------------------------------------------------------------------------
    """
    if not owner or not repo:
        raise ValueError("GitHub OWNER and REPO must be set in your .env or passed as arguments.")

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    page = 1
    all_issues: List[Dict[str, Any]] = []

    while True:
        params = {
            "state": state,      # 'open', 'closed', or 'all'
            "per_page": 100,     # Maximum allowed by GitHub REST API per page
            "page": page
        }
        response = requests.get(url, headers=BASE_HEADERS, params=params)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch issues from {owner}/{repo}: "
                f"HTTP {response.status_code} - {response.text}"
            )

        batch = response.json()
        if not batch:
            # Reached the last page (no more issues)
            break

        for item in batch:
            # GitHub treats PRs as issues; filter them out to keep only real issues
            if "pull_request" not in item:
                all_issues.append({
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "author": item.get("user", {}).get("login"),
                    "assignees": [a.get("login") for a in item.get("assignees", [])],
                    "comments": item.get("comments"),
                    "url": item.get("html_url"),
                    "state": item.get("state"),
                })

        page += 1

    return all_issues


# Backwards compatibility alias for the original spelling in existing code
fetch_all_isues = fetch_all_issues


def fetch_issues_assigned_to_user(
    username: str,
    owner: Optional[str] = OWNER,
    repo: Optional[str] = REPO,
    state: str = "all"
) -> List[Dict[str, Any]]:
    """
    Fetches issues assigned to a specific GitHub username in the repository.

    -------------------------------------------------------------------------
    KEY CONCEPTS INVOLVED:
    -------------------------------------------------------------------------
    - Query Filtering:
      The GitHub API accepts an `assignee` query parameter to filter issues
      directly on GitHub's servers, minimizing network payload size.
    - Default State:
      Defaults to state="all" so users can query both active and resolved
      issues assigned to a developer.
    -------------------------------------------------------------------------
    """
    if not owner or not repo:
        raise ValueError("GitHub OWNER and REPO must be set in your .env or passed as arguments.")

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    page = 1
    user_issues: List[Dict[str, Any]] = []

    while True:
        params = {
            "assignee": username,
            "state": state,
            "per_page": 100,
            "page": page
        }
        response = requests.get(url, headers=BASE_HEADERS, params=params)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch assigned issues for user '{username}': "
                f"HTTP {response.status_code} - {response.text}"
            )

        batch = response.json()
        if not batch:
            break

        for item in batch:
            if "pull_request" not in item:
                user_issues.append({
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "author": item.get("user", {}).get("login"),
                    "assignees": [a.get("login") for a in item.get("assignees", [])],
                    "url": item.get("html_url"),
                    "state": item.get("state"),
                })

        page += 1

    return user_issues
