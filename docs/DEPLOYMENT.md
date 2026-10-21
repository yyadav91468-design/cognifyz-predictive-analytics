
# Deployment Guide

## Current Status

The API and dashboard have been run locally during development. This document prepares the project for deployment; it does not claim that a public cloud deployment has been completed.

## Local Prerequisites

- Python 3.12
- Project dependencies from `requirements.txt`
- UCI Online Retail Excel dataset at `data/raw/Online Retail.xlsx`
- Trained model artifact at `models/best_sales_model.joblib`

## Build the Model

```powershell
python -m src.preprocess_data
python -m src.train_model
```

Confirm the model artifact and reports exist before starting the services.

## Run the API

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `/health`, `/model-info`, and `/docs` on the local API.

For a cloud service, use the hosting provider's required host and port configuration. Do not expose development services publicly without reviewing security, logging, resource usage, and model-file availability.

## Run the Dashboard

In a separate terminal:

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard loads the model and processed reports from the local project directories. Ensure those files are included in the deployment environment or generated during the build process.

## Dependencies

Install project dependencies using:

```powershell
python -m pip install -r requirements.txt
```

## Deployment Considerations

1. Keep the Python version compatible with the pinned or tested dependencies.
2. Ensure the trained model artifact is available at the path expected by the application.
3. If the API and dashboard are deployed separately, configure the dashboard to call the deployed API instead of assuming both services share a filesystem.
4. Configure health checks and capture startup errors in logs.
5. Avoid committing credentials, tokens, or private configuration.
6. Test all endpoints and dashboard pages after deployment.
7. Use HTTPS and suitable access controls for any public service.
8. Rebuild the model artifact when the preprocessing or feature schema changes.

## Verification Checklist

- [ ] Dependencies install successfully.
- [ ] Preprocessing completes successfully.
- [ ] Model training completes successfully.
- [ ] API health endpoint reports a loaded model.
- [ ] API prediction returns a numeric estimate.
- [ ] Dashboard opens and its pages render.
- [ ] Automated tests pass.
- [ ] Hosting configuration and public URLs are verified, if deployed.
