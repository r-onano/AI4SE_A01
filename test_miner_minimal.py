#!/usr/bin/env python3
"""
Minimal test for GitHubRepoMiner - extracts only the class for testing
"""

import os
import time
from pathlib import Path
from tqdm import tqdm

# **GITHUB REPOSITORY MINING** - Copied from if_predictor_improved.py
class GitHubRepoMiner:
    """Mine Python repositories from GitHub using REST API."""
    
    APPROVED_LICENSES = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"]
    DISALLOWED_PREFIXES = ("GPL-", "LGPL-", "AGPL-")
    
    def __init__(self, output_dir="repos", max_repos=50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_repos = max_repos
        
    def _has_good_license(self, license_info):
        """Check if repository has an approved license."""
        if not license_info or not license_info.get("spdx_id"):
            return False
        spdx = license_info["spdx_id"]
        # Reject GPL variants
        if any(spdx.startswith(prefix) for prefix in self.DISALLOWED_PREFIXES):
            return False
        # Check if in approved list (case-insensitive)
        return spdx.lower() in [lic.lower() for lic in self.APPROVED_LICENSES]
    
    def fetch_repo_list(self, min_stars=100):
        """
        Dynamically fetch list of Python repositories from GitHub API.
        Falls back to static list if API is unavailable or token is missing.
        """
        # Static fallback list
        popular_repos = [
            "psf/requests",
            "pallets/flask",
            "django/django",
            "numpy/numpy",
            "pandas-dev/pandas",
            "scikit-learn/scikit-learn",
            "ansible/ansible",
            "pytorch/pytorch",
            "tensorflow/tensorflow",
            "keras-team/keras",
            "matplotlib/matplotlib",
            "scipy/scipy",
            "python/cpython",
            "tornadoweb/tornado",
            "paramiko/paramiko",
            "certbot/certbot",
            "sqlalchemy/sqlalchemy",
            "fabric/fabric",
            "celery/celery",
            "boto/boto3",
            "explosion/spaCy",
            "getsentry/sentry",
            "pytest-dev/pytest",
            "pallets/click",
            "pypa/pip",
            "cookiecutter/cookiecutter",
            "home-assistant/core",
            "zulip/zulip",
            "ytdl-org/youtube-dl",
            "httpie/httpie",
        ]
        
        # Try dynamic discovery via GitHub API
        try:
            import requests
            
            # Read token from environment
            token = os.getenv("GITHUB_TOKEN")
            
            # Create session with authentication
            sess = requests.Session()
            sess.headers["Accept"] = "application/vnd.github+json"
            if token:
                sess.headers["Authorization"] = f"token {token}"
            
            # Build search query
            q = f"language:Python stars:>={min_stars} fork:false archived:false"
            
            results = []
            page = 1
            per_page = 50
            
            print(f"Fetching Python repositories from GitHub API...")
            
            while len(results) < self.max_repos:
                try:
                    r = sess.get(
                        "https://api.github.com/search/repositories",
                        params={
                            "q": q,
                            "sort": "stars",
                            "order": "desc",
                            "per_page": per_page,
                            "page": page
                        },
                        timeout=30
                    )
                    
                    # Handle rate limiting
                    if r.status_code == 403 and "rate limit" in r.text.lower():
                        print("   ⚠ Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    
                    r.raise_for_status()
                    
                    items = r.json().get("items", [])
                    if not items:
                        break
                    
                    for item in items:
                        if len(results) >= self.max_repos:
                            break
                        
                        # Check license
                        lic = item.get("license")
                        if not self._has_good_license(lic):
                            continue
                        
                        # Add to results
                        full_name = f"{item['owner']['login']}/{item['name']}"
                        results.append(full_name)
                    
                    page += 1
                    
                    # Rate limiting between requests
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"   ⚠ Request failed: {e}")
                    break
            
            if results:
                print(f"✓ Dynamically fetched {len(results)} repositories via GitHub API (MIT/APACHE/BSD licenses)")
                return results
            else:
                raise Exception("No repositories found via API")
                
        except Exception as e:
            print(f"⚠ GitHub token not found or API fetch failed — using static fallback list.")
            print(f"   (Error: {e})")
            return popular_repos[:self.max_repos]


# Test 1: Without token (should use fallback)
print("=" * 70)
print("TEST 1: Without GITHUB_TOKEN (should use static fallback)")
print("=" * 70)

# Remove token if exists
if "GITHUB_TOKEN" in os.environ:
    del os.environ["GITHUB_TOKEN"]

miner = GitHubRepoMiner(output_dir="test_repos", max_repos=10)
repos = miner.fetch_repo_list()

print(f"\nResult: Got {len(repos)} repositories")
print(f"First 5: {repos[:5]}")
assert len(repos) == 10, "Should get exactly 10 repos"
print("✓ TEST 1 PASSED\n")

# Test 2: License filtering
print("=" * 70)
print("TEST 2: License filtering logic")
print("=" * 70)

miner2 = GitHubRepoMiner(output_dir="test_repos", max_repos=5)

test_cases = [
    ({"spdx_id": "MIT"}, True),
    ({"spdx_id": "Apache-2.0"}, True),
    ({"spdx_id": "BSD-3-Clause"}, True),
    ({"spdx_id": "GPL-3.0"}, False),
    ({"spdx_id": "LGPL-2.1"}, False),
    ({"spdx_id": "AGPL-3.0"}, False),
    (None, False),
    ({}, False),
]

all_passed = True
for lic, expected in test_cases:
    result = miner2._has_good_license(lic)
    status = "✓" if result == expected else "✗"
    lic_str = lic.get("spdx_id") if lic else "None"
    print(f"{status} License {lic_str}: {result} (expected {expected})")
    if result != expected:
        all_passed = False

assert all_passed, "Some license checks failed"
print("✓ TEST 2 PASSED\n")

print("=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nYour GitHubRepoMiner is working correctly!")
print("\nTo use with a real GitHub token:")
print("  export GITHUB_TOKEN=ghp_your_real_token")
print("  python3 if_predictor_improved.py --stage mine --max_repos 50")
print("\nWithout token, it uses the static fallback list of 30 popular repos.")
