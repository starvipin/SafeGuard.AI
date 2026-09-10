# GitHub Actions: tests and website deployment

GitHub discovers workflow YAML files under `.github/workflows/`, so the workflows remain in this directory.

| File | Trigger | Action |
| --- | --- | --- |
| `ci.yml` | Push to `main`/`develop`, PRs targeting those branches, or manual run | Install locked dependencies → check syntax → run pytest |
| `deploy.yml` | Push to `main` or manual run | Sync website code and Dockerfile to the `sainivipin/SafeGuard-Live` HF Space |

```text
Push to GitHub main
  ├── ci.yml → tests
  └── deploy.yml → push code to HF Space → Docker build → app.py → website
```

CI and deployment are independent workflows. Deployment does not wait for the CI result. The existing deployment triggers and push behavior are preserved.

`deploy.yml` reads the GitHub secret `HF_TOKEN`. The root `Dockerfile` runs `app.py` on port `5000`; `docker-compose.yml` supports local container execution. These files remain in the root to preserve deployment entrypoints.

- `src/model_training/step_04_upload_to_hf.py` uploads trained model files to an HF **model repository**.
- `deploy.yml` sends website code to an HF **Space**.

Organizing local files does not deploy them. A push to `main` or a manual deployment workflow run syncs code to the live Space.
