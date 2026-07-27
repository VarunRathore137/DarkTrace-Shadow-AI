# Dark Trace AI - GitHub Actions CI/CD Deployment Guide

This guide provides a comprehensive, step-by-step breakdown of how to set up, understand, and execute Continuous Integration and Continuous Deployment (CI/CD) for **Dark Trace AI** using **GitHub Actions**.

---

## 📌 Architecture Overview

Dark Trace AI consists of two core containerized microservices:
1. **FastAPI Backend (`api/main.py`)**: Handles ML inference, risk scoring, slang/emoji detection, and vector similarity search.
2. **Streamlit Frontend (`streamlit_app.py`)**: Interactive web dashboard for intelligence visualization.

Our GitHub Actions workflow automates three main stages:
```
┌───────────────────────────┐     ┌──────────────────────────────┐     ┌──────────────────────────┐
│   1. Test & Quality       │ ──> │  2. Build & Publish Docker   │ ──> │   3. Deploy Microservices│
│   (Pytest & Dependencies) │     │  (FastAPI & Streamlit Images)│     │   (Cloud / GHCR Hosting) │
└───────────────────────────┘     └──────────────────────────────┘     └──────────────────────────┘
```

---

## 📁 File Location & Configuration

Place the workflow file in your repository under:
`darktrace-ai-/.github/workflows/deploy.yml`

---

## 📜 Full `.github/workflows/deploy.yml` Configuration

```yaml
name: Dark Trace AI - CI/CD Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:

jobs:
  test-and-lint:
    name: Run Tests & Quality Checks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository code
        uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Pytest test suite
        run: |
          pytest tests/ --doctest-modules --junitxml=junit/test-results.xml

  build-and-publish:
    name: Build & Publish Docker Containers
    needs: test-and-lint
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'

    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository code
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry (GHCR)
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata for API Docker image
        id: meta-api
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}/darktrace-api
          tags: |
            type=raw,value=latest
            type=sha,format=short

      - name: Build and push API Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta-api.outputs.tags }}
          labels: ${{ steps.meta-api.outputs.labels }}

      - name: Extract metadata for Frontend Docker image
        id: meta-frontend
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}/darktrace-frontend
          tags: |
            type=raw,value=latest
            type=sha,format=short

      - name: Build and push Frontend Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile.frontend
          push: true
          tags: ${{ steps.meta-frontend.outputs.tags }}
          labels: ${{ steps.meta-frontend.outputs.labels }}

  deploy:
    name: Trigger Cloud Deployment
    needs: build-and-publish
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'

    steps:
      - name: Deployment Notification / Webhook Trigger
        run: |
          echo "Deployment successful! Docker containers published to GHCR:"
          echo " - ghcr.io/${{ github.repository }}/darktrace-api:latest"
          echo " - ghcr.io/${{ github.repository }}/darktrace-frontend:latest"
```

---

## 🔍 Detailed Step-by-Step Breakdown & Relevance

### 1. Workflow Triggers (`on:`)

```yaml
on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:
```
* **What it does**: Defines what events trigger the pipeline.
* **Relevance to Dark Trace AI**:
  * `push`: Automatically validates and deploys new commits pushed directly to `main`.
  * `pull_request`: Runs test suites on incoming code changes before merging, preventing broken code or failing tests from entering production.
  * `workflow_dispatch`: Enables a manual **"Run workflow"** button inside GitHub Actions tab for manual deployment testing.

---

### 2. Job 1: `test-and-lint` (Continuous Integration)

```yaml
  test-and-lint:
    name: Run Tests & Quality Checks
    runs-on: ubuntu-latest
```
* **What it does**: Spawns an isolated Ubuntu Linux virtual machine runner (`ubuntu-latest`) to execute test suites.
* **Relevance**: Guarantees that code runs in a clean, reproducible environment identical to cloud servers.

#### Step 2.1: Checkout Code
```yaml
      - name: Checkout repository code
        uses: actions/checkout@v4
```
* **What it does**: Clones the GitHub codebase onto the runner environment.
* **Relevance**: Gives subsequent steps access to `requirements.txt`, `api/`, `streamlit_app.py`, and `tests/`.

#### Step 2.2: Setup Python & Caching
```yaml
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'
```
* **What it does**: Installs Python 3.10 and enables automatic caching for `pip` packages.
* **Relevance**: Dark Trace AI uses heavy dependencies (like `scikit-learn`, `xgboost`, `chromadb`, `sentence-transformers`). Caching `pip` packages reduces workflow execution time from minutes to seconds on repeat runs.

#### Step 2.3: Install Dependencies
```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
```
* **What it does**: Installs all required runtime and test packages listed in `requirements.txt`.
* **Relevance**: Ensures all ML libraries, FastAPI drivers, and test frameworks are ready for testing.

#### Step 2.4: Execute Pytest Suite
```yaml
      - name: Run Pytest test suite
        run: |
          pytest tests/ --doctest-modules --junitxml=junit/test-results.xml
```
* **What it does**: Runs `test_api.py` and `test_latency.py`.
* **Relevance**: 
  * Verifies backend API endpoints (`/predict`, `/health`) respond correctly.
  * Validates model inference latency constraints (`test_latency.py`).
  * **Gatekeeper**: If any test fails, GitHub Actions halts the pipeline immediately and cancels container building.

---

### 3. Job 2: `build-and-publish` (Docker Container Registry)

```yaml
  build-and-publish:
    name: Build & Publish Docker Containers
    needs: test-and-lint
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
```
* **What it does**: Containerizes both microservices into Docker images and publishes them to **GitHub Container Registry (GHCR)**.
* **Relevance**: 
  * `needs: test-and-lint`: Ensures Docker containers are built **ONLY** when all unit and performance tests pass.
  * `if: ...`: Prevents publishing images during pull-requests; only publishes on main branch merges.

#### Step 3.1: Log in to GHCR
```yaml
      - name: Log in to GitHub Container Registry (GHCR)
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
```
* **What it does**: Authenticates against GitHub's free image registry `ghcr.io` using the auto-generated `${{ secrets.GITHUB_TOKEN }}`.
* **Relevance**: Eliminates the need to configure third-party Docker Hub credentials or external API keys.

#### Step 3.2: Extract Metadata & Build API Container
```yaml
      - name: Extract metadata for API Docker image
        id: meta-api
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}/darktrace-api
          tags: |
            type=raw,value=latest
            type=sha,format=short

      - name: Build and push API Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta-api.outputs.tags }}
          labels: ${{ steps.meta-api.outputs.labels }}
```
* **What it does**: Builds the FastAPI backend image via `Dockerfile` and tags it with both `:latest` and Git commit SHA (e.g. `:a1b2c3d`), then pushes to GHCR.
* **Relevance**: Produces a standardized, production-ready container image for the core AI backend (`api/main.py`).

#### Step 3.3: Extract Metadata & Build Frontend Container
```yaml
      - name: Extract metadata for Frontend Docker image
        id: meta-frontend
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}/darktrace-frontend
          tags: |
            type=raw,value=latest
            type=sha,format=short

      - name: Build and push Frontend Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile.frontend
          push: true
          tags: ${{ steps.meta-frontend.outputs.tags }}
          labels: ${{ steps.meta-frontend.outputs.labels }}
```
* **What it does**: Builds the Streamlit dashboard image via `Dockerfile.frontend` and pushes to GHCR.
* **Relevance**: Produces a lightweight container image for the user interface dashboard (`streamlit_app.py`).

---

### 4. Job 3: `deploy` (Cloud Hosting Integration)

```yaml
  deploy:
    name: Trigger Cloud Deployment
    needs: build-and-publish
    runs-on: ubuntu-latest
```
* **What it does**: Final deployment trigger stage after Docker images are safely pushed.
* **Deployment Options**:
  * **Option A (Render / Railway / Cloud Run)**: Add a webhook curl command to trigger automatic container pulling.
  * **Option B (VPS via SSH)**: Use `appleboy/ssh-action` to run `docker compose pull && docker compose up -d` on your remote server.

---

## 🚀 How to Enable and Test

1. **Commit and Push**:
   ```bash
   git add .github/workflows/deploy.yml GITHUB_ACTIONS_GUIDE.md
   git commit -m "feat: add GitHub Actions CI/CD deployment workflow and guide"
   git push origin main
   ```
2. **Monitor Workflow**:
   * Navigate to your GitHub repository -> **Actions** tab.
   * Click on **"Dark Trace AI - CI/CD Pipeline"**.
   * Observe live logs for testing, Docker build, and container publishing.
3. **Verify Published Packages**:
   * Navigate to your GitHub Profile -> **Packages** tab to view your published `darktrace-api` and `darktrace-frontend` Docker images.
