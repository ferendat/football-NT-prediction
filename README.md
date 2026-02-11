# International Football Match Predictor

Predicts international football match outcomes using a TensorFlow neural network trained on historical match data from 1872 to present. Outputs win/draw/loss probabilities for any two national teams.

## Features
- Team embedding layers to capture latent team strength representations
- Rolling 5-match win/draw/loss rate features per team
- Home advantage encoding with neutral venue support
- One-hot encoded tournament, city, and country context
- Automatic team name standardization via `former_names.csv`
- Softmax output for win/draw/loss probability distribution

## Dataset
[International Football Results 1872–2017 – Kaggle (martj42)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Two files required:
- `results.csv` — match results with scores, teams, tournament, and venue
- `former_names.csv` — maps historical team names to current names

## Requirements
```
tensorflow
pandas
numpy
scikit-learn
```

## Usage
```
python football_predictor.py
```
Update the file paths at the top of the script to match your local directory, then call `predict_match()` at the bottom:

```python
predict_match("Germany", "France", 2025)
```

## Technical Details
- **Architecture:** Dual embedding inputs (home/away team) + numeric feature branch, concatenated into 3 dense layers (128 → 64 → 32 units)
- **Activation:** Swish with L2 regularization (0.0001) and 20% dropout
- **Embedding size:** `sqrt(num_teams)` per team
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical crossentropy
- **Training:** 44 epochs, batch size 32, 80/20 train-val split
- **Output classes:** Home Win (0), Draw (1), Away Win (2)

## Performance
Trained on ~48,000 matches across all international competitions.

| Metric | Value |
|---|---|
| Validation Accuracy | ~77.9% |
| Validation Loss | ~0.475 |

## Output
```
Home Win Probability: 58.27%
Draw Probability:     22.41%
Away Win Probability: 19.32%
```

## Limitations
- Rolling stats are computed from the full dataset, not split by train/test — minor data leakage possible
- One-hot encoding of city/country produces very high-dimensional sparse features
- No recency weighting — a match from 1920 carries equal base weight to one from 2024
- `former_names.csv` must be kept up to date for newly renamed nations

## License
MIT
