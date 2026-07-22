# End-to-End MLOps System: Implementation & Execution Plan

This document synthesizes the perspectives of Business Analysis, Data Analysis, and Software Engineering to outline the full development and deployment lifecycle for the production-grade Machine Learning pipeline.

## 1. System Architecture & Technology Stack

The solution implements a decoupled 6-layer architecture designed for scalability, maintainability, and reproducibility.

**Technology Stack:**
* **Modeling & Analytics:** XGBoost, Scikit-learn, Pandas, Numpy
* **Experiment Tracking:** MLflow
* **API / Backend Serving:** FastAPI, Uvicorn, Pydantic
* **Frontend / Dashboarding:** Streamlit
* **Containerization:** Docker, Docker Compose
* **CI/CD Pipeline:** GitHub Actions
* **Data/Model Monitoring:** EvidentlyAI
* **Testing:** Pytest, HTTPX
* **Deployment:** Render (Cloud Hosting)

**Component Integration:**
1. **Model Registry:** MLflow tracks training metrics and saves the best model artifact to `models/churn_model.pkl`.
2. **Backend API:** FastAPI (`src/api.py`) loads the model and exposes `POST /predict` and `POST /predict/batch`. Pydantic (`src/schemas.py`) validates incoming JSON payloads.
3. **Frontend UI:** Streamlit (`app/streamlit_app.py`) sends HTTP requests to the FastAPI backend and displays interactive dashboards for business users.
4. **Monitoring:** EvidentlyAI (`src/monitor.py`) analyzes incoming feature requests against training data to detect drift, triggering alerts or reports.
5. **Orchestration:** `docker-compose.yml` links the `api` and `ui` containers, ensuring networking and environment parity between local development and production.

## 2. Business Analysis Plan

**Stakeholder Needs & Problem Statement:**
Models built only on local environments create no business value and are prone to irreproducibility. Without API integrations, frontend access, or drift monitoring, business operations are disconnected from ML insights, leading to degraded decision-making and revenue loss.

**Success Metrics (KPIs):**
* **Deployment Speed:** Time from code push to production deploy (Target: < 5 minutes).
* **API Reliability:** Service uptime (Target: > 99.5%) and response time (Target: < 50ms for single predictions).
* **Model Trust:** Visibility into data/target drift and performance degradation.
* **Adoption:** Usage rate of the Streamlit dashboard by non-technical stakeholders.

**Risk Mitigation:**
* *Risk:* Model decay over time. *Mitigation:* Automated weekly EvidentlyAI drift reports.
* *Risk:* Downtime during updates. *Mitigation:* GitHub Actions CI/CD with test validation prior to automated deployment on Render.
* *Risk:* Schema changes breaking integration. *Mitigation:* Strict validation using Pydantic.

## 3. Data Analysis Plan

**Data Sources & Transformation:**
* **Inputs:** Customer demographic, account, and behavioral data (e.g., tenure, monthly charges).
* **ETL Pipeline:** Raw data is preprocessed inside the modeling pipeline (imputations, encodings, scaling).
* **Analytics Methods:**
  * **Predictive:** XGBoost classifier for predicting customer churn probability.
  * **Explainability:** SHAP values integrated into the Streamlit dashboard to explain individual customer predictions.

**Monitoring & Insights (EvidentlyAI):**
* **Data Drift:** Tracks shifts in input feature distributions (e.g., KS test for numerical features).
* **Target Drift:** Monitors shifts in the predicted churn rate.
* **Business Tie-Back:** Drift alerts signal the Data Science team to retrain the model, ensuring the business is acting on accurate predictions.

## 4. Development Roadmap & Milestones

**Phase 1: Foundation & Data Science (Weeks 1-2)**
* *Roles:* Data Scientist, Data Engineer
* Explore data, train baseline models (XGBoost/Scikit-learn).
* Implement MLflow tracking in `src/train.py` to log metrics and save the best model to `models/churn_model.pkl`.

**Phase 2: API Development & Containerization (Weeks 2-3)**
* *Roles:* Backend Engineer, MLOps Engineer
* Define API schemas using Pydantic in `src/schemas.py`.
* Build FastAPI endpoints in `src/api.py`.
* Implement logic in `src/predict.py`.
* Write `Dockerfile` and `docker-compose.yml` for local multi-service orchestration.

**Phase 3: Frontend & Business Integration (Weeks 3-4)**
* *Roles:* Frontend Engineer, Business Analyst
* Develop Streamlit UI in `app/streamlit_app.py`.
* Integrate UI with FastAPI endpoints.
* Implement SHAP visualizations for model explainability.

**Phase 4: Testing & Monitoring (Week 4)**
* *Roles:* QA Engineer, Data Scientist
* Write unit/integration tests in `tests/test_api.py` and `tests/test_predict.py` using Pytest.
* Set up EvidentlyAI monitoring scripts in `src/monitor.py` for automated reporting.

**Phase 5: CI/CD & Deployment (Week 5)**
* *Roles:* DevOps Engineer
* Configure `.github/workflows/ci_cd.yml` for automated testing and Docker builds.
* Setup Render platform for auto-deployment hooked to the main branch.

## 5. Code Organization Guidelines

* **Separation of Concerns:** Keep API routing (`api.py`), business logic (`predict.py`), and data structures (`schemas.py`) separate.
* **Dependency Management:** Maintain a locked `requirements.txt`.
* **Testing:** All new features must include corresponding Pytest coverage.
* **Version Control:** Use feature branches and Pull Requests. PRs must pass all GitHub Action checks before merging.

## 6. Testing Strategy

* **Unit Testing:** Validate individual functions in `predict.py` (e.g., data preprocessing logic) using mocked data.
* **Integration Testing:** Test FastAPI endpoints (`test_api.py`) with HTTPX `AsyncClient` to verify request/response lifecycles and Pydantic validation.
* **Deployment Validation:** Health check endpoints (`GET /health`) run in production to verify container health before traffic is routed.

## 7. Deployment Plan

* **Environment Setup:** Render platform configured to pull the latest Docker image or build from the repository. Environment variables stored securely in Render.
* **CI/CD Pipeline:**
  1. Triggered on push to `main`.
  2. Runs `pytest tests/`.
  3. Builds Docker image.
  4. Pushes to container registry (or builds directly on Render).
  5. Auto-deploys if tests pass.
* **Monitoring:** Uptime monitored via Render dashboard. Model drift monitored via Scheduled jobs running `src/monitor.py` (EvidentlyAI).
* **Rollback Strategy:** If a deployment fails or metrics degrade, GitHub Actions can be triggered to redeploy the previous successful commit tag.

## 8. Assumptions & Blockers

* **Assumptions:**
  * Render account is fully provisioned and connected to the GitHub repository.
  * Base training dataset is available and clean for the initial model build.
* **Blockers/Clarifications:**
  * Define the email/notification system for EvidentlyAI drift alerts (e.g., SendGrid, Slack webhook).
  * Determine database connection specifics if feature logging is required for long-term monitoring.
