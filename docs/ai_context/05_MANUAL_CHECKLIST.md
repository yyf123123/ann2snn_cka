# Manual Checklist

## Already Confirmed by Author

- [x] Do not upload the paper PDF to GitHub.
- [x] Use MIT License.
- [x] README should include paper-reported results only; do not claim full reproduction.
- [x] Keep VGG-16 and ImageNet-related code or descriptions; do not delete them.
- [x] The main execution entry is `resnet18_cifar10.py`.
- [x] Do not publish checkpoints or model weight files (`.pth`, `.pt`, `.ckpt`).
- [x] Preserve existing absolute server paths for now (e.g., `/home/lbz/git-hub/...`).
- [x] Let the AI agent verify runnable commands before claiming they work.
- [x] Do not delete existing Python files.
- [x] Do not rewrite algorithms.
- [x] Do not move entry scripts.
- [x] First round: documentation, rules, README, LICENSE, .gitignore, and audit only.
- [x] No git commit unless explicitly requested.

## Still Need Manual Review

- [ ] Confirm whether README should be English-only or bilingual (Chinese + English).
- [ ] Confirm final BibTeX entry after paper metadata is finalized (title, authors, venue, year).
- [ ] Confirm whether to provide a minimal demo checkpoint externally in the future.
- [ ] Confirm whether to add `argparse` for command-line configuration later.
- [ ] Confirm whether to split flat files into `src/` package structure later.
- [ ] Confirm whether to keep generated CKA `.npy` files out of git (currently in `.gitignore`).
- [ ] Confirm whether to add GitHub Actions (CI) later.
- [ ] Confirm whether to provide a lightweight `requirements-minimal.txt`.
- [ ] Confirm whether to add `scripts/run_resnet18_cifar10.sh` later.
- [ ] Confirm whether model checkpoint paths should be configurable via environment variables.
- [ ] Confirm whether VGG-16 `models.py` rebuild code is complete (currently may need author review).

## Audit Findings

| Item | Status |
|------|--------|
| Sensitive data (tokens, passwords) | None found |
| Checkpoint files in repo | None found |
| Generated files in repo | None found |
| Paper PDF in repo | None found |
| Hard-coded local paths | Found (recorded in audit, intentionally preserved) |
| `.claude/` directory | Present (local only, in `.gitignore`) |
