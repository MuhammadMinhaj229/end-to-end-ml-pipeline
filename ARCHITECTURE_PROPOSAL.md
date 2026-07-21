# End-to-End MLOps Pipeline: Architecture & Strategic Plan Refinement

This document outlines the collaborative, industry-grade architecture and strategic execution plan for transitioning the machine learning project from an offline, localized state into a resilient, production-ready solution.

## 1. Project Objective Restated
The overarching goal of this project is to build and deploy a production-grade, end-to-end Machine Learning Operations (MLOps) system for predicting customer churn. We aim to bridge the "Data Science to Production" gap by migrating models from static Jupyter Notebooks to a highly available, robustly monitored, and accessible web ecosystem. This pipeline must provide immediate business value by enabling non-technical stakeholders to interface with predictions via a frontend dashboard, whilst serving real-time requests securely via an automated REST API.

## 2. Key Components of the Plan

### Deliverables
*   **Infrastructure-as-Code & Orchestration:** `docker-compose.yml` and `Dockerfile` ensuring parity across environments.
*   **Model Registry & Tracking:** MLflow configurations for experiment reproducibility.
*   **Backend API Services:** High-performance FastAPI endpoints for real-time and batch predictions.
*   **Business Dashboard:** Streamlit frontend allowing stakeholders to interact with customer data, batch process predictions, and interpret ML decisions via SHAP.
*   **CI/CD Automation Pipeline:** GitHub Actions workflow managing tests, builds, and automated deployments to Render.
*   **Monitoring & Drift Reports:** Scheduled EvidentlyAI jobs generating HTML reports on data and model degradation.

### Timelines (Proposed Agile Iterations)
*   **Sprint 1 (Weeks 1-2) - "The Foundation":** Standardize MLflow tracking and build out FastAPI endpoints. *Dependency: Finalized model baseline.*
*   **Sprint 2 (Weeks 3-4) - "The Interface & Orchestration":** Finalize Streamlit Dashboard and Dockerize both frontend and backend for local staging.
*   **Sprint 3 (Weeks 5-6) - "The Pipeline":** Implement GitHub Actions for CI/CD, finalize Pytest suite, and launch the v1 deployment on Render.
*   **Sprint 4 (Weeks 7-8) - "The Monitor":** Integrate EvidentlyAI for automated data and target drift reporting.

### Dependencies
*   Consistent and clean historical training data (the reference dataset).
*   Active Render cloud account tied to the source repository.
*   Access to external email/notification services (e.g., SendGrid, Slack API) for EvidentlyAI alerts.

## 3. High-Level Architectural Approach

### Major Modules & Interactions
The architecture utilizes a decoupled microservice-like approach:

1.  **The API Module (FastAPI):** Acts as the central nervous system. It consumes the latest registered model artifact (`churn_model.pkl`) and exposes endpoints (`/predict`, `/batch`). It utilizes Pydantic to strictly validate JSON inputs to protect the inference engine from malformed data.
2.  **The UI Module (Streamlit):** The client-facing dashboard. It has no direct access to the model; instead, it communicates strictly over HTTP with the API Module.
3.  **The Orchestration Layer (Docker):** Binds the API and UI in isolated containers inside a unified virtual network via `docker-compose`.
4.  **The Monitoring Module (EvidentlyAI):** Runs asynchronously. It queries recent prediction logs (live input data vs. baseline training data) to calculate statistical drift.

### Technology Considerations
*   **FastAPI vs. Flask/Django:** Chosen for native asynchronous I/O and Pydantic validation, offering <50ms inference latency which is essential for real-time fintech/e-commerce use cases.
*   **Render:** Selected over AWS/GCP for its developer-friendly "Platform-as-a-Service" capability, reducing DevOps overhead while offering solid automated deployment hooks.

### Scalability & Security Implications
*   **Scalability:** Because the API is completely stateless, the `api` container can be horizontally scaled infinitely behind a load balancer as traffic increases.
*   **Security:**
    *   Currently, the architecture relies on Pydantic to prevent injection attacks or bad payloads.
    *   *Improvement Required:* The endpoints are open. We must implement API Key validation (e.g., via HTTP Bearer headers) to restrict unauthorized access to the `/predict` endpoints.
    *   Environment variables (secrets) will be managed strictly within the Render dashboard and GitHub Actions Secrets, never hardcoded.

## 4. Critical Risks & Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Model Decay (Concept Drift)** | High (Financial loss from bad predictions) | Implement weekly automated EvidentlyAI reporting. Set strict drift thresholds that trigger automatic notifications to Data Science teams. |
| **Downtime during Deployments** | Medium (Interrupted service) | Utilize Render's Zero-Downtime deploy feature. Ensure the GitHub Actions pipeline runs full integration tests *before* triggering the Render webhook. |
| **Schema Breaking Changes** | High (UI and API disconnect) | Maintain strict versioning on the API endpoints (e.g., `/api/v1/predict`). Any breaking change requires a new `v2` endpoint, leaving `v1` intact until deprecation. |

## 5. Clarifying Questions for Alignment

To ensure we are fully aligned before executing code, please advise on the following:
1.  **Authentication:** Should the API utilize basic API Key authentication, OAuth2, or rely on internal network routing if used purely by internal services?
2.  **Data Storage for Monitoring:** Where should we log the incoming live inference requests to feed into EvidentlyAI? (e.g., PostgreSQL database, flat CSV in a mounted volume, or an external blob store like AWS S3?)
3.  **Alerting Mechanism:** How does the business prefer to receive model drift alerts? (e.g., Email, Slack channel, Jira ticket?)

## 6. Next Concrete Steps

1.  **Review and Approve:** Stakeholders review this `ARCHITECTURE_PROPOSAL.md` and provide answers to the clarifying questions.
2.  **Initialization:** The engineering team generates the boilerplate for the FastAPI backend (`src/api.py`, `src/schemas.py`) and standardizes the MLflow tracking scripts.
3.  **Initial Integration Test:** We will deploy a "Hello World" API endpoint through the CI/CD pipeline to Render to validate our deployment mechanism before committing complex machine learning logic.
