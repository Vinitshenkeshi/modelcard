# modelcard

Turn a trained scikit-learn estimator into a deterministic Markdown model card - the kind of documentation MLOps and model-governance reviews ask for, generated straight from the artifact instead of hand-written.

## Problem

Model cards (intended use, training data, metrics, limitations, feature importances) are widely recommended but rarely kept up to date, because writing them by hand is tedious and drifts from the actual model. modelcard derives the card from the estimator and an eval-metrics dict, so the documentation is regenerated in one command and is byte-identical given identical inputs - safe to commit and diff in CI.

## What it does

- Reads the estimator's class name and, for tree/linear models, its feature importances (feature_importances_ or mean-absolute coef_).
- Renders a metrics table (sorted by name, floats to 4 dp for stable diffs).
- Emits sections for intended use, out-of-scope uses, training data, limitations, and ethical considerations from typed metadata objects.
- Ships a library and a CLI (modelcard demo, modelcard render).
- Fully offline: the demo uses scikit-learn's bundled Iris dataset - no network, no downloads, no API keys.

## Quickstart

From a fresh clone (requires uv):

```bash
uv sync
uv run modelcard demo --out card.md   # writes a model card for a demo Iris model
uv run pytest -q                       # 28 tests
```

### Library usage

```python
from sklearn.ensemble import RandomForestClassifier
from modelcard import DatasetInfo, ModelInfo, render_model_card

est = RandomForestClassifier().fit(X_train, y_train)

card = render_model_card(
    est,
    metrics={"accuracy": 0.97, "f1_macro": 0.96},
    dataset=DatasetInfo(
        name="My dataset", description="...",
        n_samples=1000, n_features=4,
        feature_names=["a", "b", "c", "d"],
        target_names=["neg", "pos"],
    ),
    model=ModelInfo(
        name="My Classifier", version="1.0",
        intended_use="...",
        limitations=["Small training set."],
    ),
)
print(card)  # Markdown string, ends with a newline
```

## Real output

Running uv run modelcard demo trains a 50-tree RandomForest on Iris (75/25 split, random_state=42) and produces the card below. The demo scores accuracy 0.9211 on 38 held-out test samples - these numbers are asserted in tests/test_demo.py, so the README and the code cannot drift. The full file lives at examples/iris_model_card.md; an excerpt:

```markdown
# Model Card: Iris Species Classifier

**Version:** 0.1.0
**Estimator:** `RandomForestClassifier`
**Maintainer:** Vinit Shenkeshi
**License:** MIT

## Evaluation Metrics

| Metric | Value |
| --- | --- |
| accuracy | 0.9211 |
| f1_macro | 0.9230 |
| n_test_samples | 38 |
| precision_macro | 0.9246 |
| recall_macro | 0.9231 |

## Top Feature Importances (up to 10)

| Feature | Importance |
| --- | --- |
| petal length (cm) | 0.4394 |
| petal width (cm) | 0.4253 |
| sepal length (cm) | 0.1091 |
| sepal width (cm) | 0.0263 |
```

### Rendering your own model

Pickle any fitted estimator, describe it in a JSON spec, and render:

```bash
uv run modelcard render --estimator model.pkl --spec spec.json --out card.md
```

where spec.json has metrics, dataset, and model objects (matching the DatasetInfo / ModelInfo fields).

## Development

```bash
uv run ruff check .      # lint (E,F,I,UP,B,SIM,RUF, line-length 100)
uv run mypy src tests    # strict type-checking
uv run pytest -q         # tests
```

CI runs all three on every push and PR (see .github/workflows/ci.yml).

## What I'd build next

- HTML / PDF renderers alongside Markdown (the render layer is already isolated).
- Permutation importance for models exposing neither feature_importances_ nor coef_.
- Metric-threshold gates: fail CI if a card's metrics regress below a committed baseline.
- Data-drift / bias sub-cards computed from a supplied evaluation split.

## Maintainer

Vinit Shenkeshi is an AI/ML Engineer with 4+ years of experience building end-to-end machine learning solutions. He specializes in developing scalable, production-ready ML systems and maintaining robust documentation standards for model governance.

- Email: vinitshenkeshi1@gmail.com
- LinkedIn: https://www.linkedin.com/in/vinitshenkeshi/