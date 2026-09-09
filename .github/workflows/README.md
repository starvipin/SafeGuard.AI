# GitHub Actions: tests aur website deployment

GitHub workflow YAML files `.github/workflows/` mein discover karta hai, isliye Actions yahin rakhe hain.

| File | Kab chalti hai? | Kya karti hai? |
| --- | --- | --- |
| `ci.yml` | `main`/`develop` push, un branches ke PRs, manual run | Locked dependencies → syntax check → pytest |
| `deploy.yml` | `main` push ya manual run | Website code aur Dockerfile ko `sainivipin/SafeGuard-Live` HF Space par sync |

```text
GitHub main par push
  ├── ci.yml → tests
  └── deploy.yml → HF Space ko code push → Docker build → app.py → website
```

Existing CI aur deploy independent workflows hain. Deploy CI result ka wait nahi karta. Deployment workflow ke triggers aur push behavior restructuring mein change nahi kiye gaye.

`deploy.yml` GitHub secret `HF_TOKEN` use karta hai. `Dockerfile` root `app.py` chalata hai, port `5000` par. `docker-compose.yml` local container ke liye hai. Existing deployment entrypoints preserve karne ke liye dono root mein hain.

- `model_training/step_04_upload_to_hf.py` → trained model HF **model repo** mein.
- `deploy.yml` → website code HF **Space** mein.

Local folders organize karne se deployment nahi hota. `main` par push ya manual deploy workflow run live site update karta hai.
