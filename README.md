# International Football Match Predictor

Predicts the outcome of international football matches (home win / draw / away win) with a small TensorFlow neural network trained on historical results going back to 1872. Give it two national teams and it prints a probability for each outcome.

The script also trains a few simple baselines on the same data, so you can see whether the neural network is actually worth it.

## Features

- Team embeddings (one shared layer for home and away) to learn a latent strength for each team
- Rolling form over each team's last 5 matches (win/draw/loss rates, goals for and against), counting home and away games together
- Elo rating going into every match, updated after the result
- Neutral venue flag
- Team names standardized through `former_names.csv`, only inside the dates a name was actually used
- Softmax output with a probability for each of the three outcomes
- Comparison against three baselines: class frequencies, an Elo-only model, and logistic regression

## Dataset

[International football results from 1872](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) on Kaggle (martj42).

You need two files:

- `results.csv`: match results with scores, teams, tournament and venue
- `former_names.csv`: maps old team names to current ones

## Requirements

- tensorflow
- pandas
- numpy
- scikit-learn

## Usage

```
python football_model.py
```

Change the file paths at the top of the script to wherever your CSVs are. At the bottom you can call `predict_match()` for any pair of teams:

```python
predict_match("Germany", "France", neutral=False)
```

Set `neutral=True` for a match on neutral ground, like a tournament game.

## How it works

- **Split:** matches are sorted by date and split in time, not randomly. Train is everything before 2016, validation is 2016-2019, and test is 2020 onwards. The dates are easy to change at the top of the script.
- **No leakage:** form features only use matches played *before* the one being predicted, and Elo is recorded before it gets updated with the result. The scaler and the team vocabulary are fit on the training period only. Teams the model hasn't seen get an "unknown" embedding.
- **Architecture:** home and away team embeddings plus the numeric features, concatenated and passed through three dense layers (128 → 64 → 32).
- **Regularization:** swish activations, L2 (0.0001) and 30% dropout.
- **Embedding size:** `sqrt(num_teams)`, kept between 4 and 16.
- **Training:** Adam (lr=0.001), categorical crossentropy, batch size 64. It runs up to 100 epochs and stops early when validation loss stops improving.
- **Output classes:** Home Win (0), Draw (1), Away Win (2)

## Performance

About 48,000 matches in total: 39,409 for training, 3,919 for validation and 4,652 for testing.

| Metric | Value |
|---|---|
| Test accuracy | ~60.6% |
| Test log-loss | ~0.871 |
| Always predicting a home win | ~47.5% |

An earlier version of this project reported ~78% accuracy. That number was inflated: the rolling features included the match being predicted, and the train/validation split was random, so the model was effectively seeing the future. After fixing both, accuracy dropped to the numbers above, which are much more realistic.

The script prints a full comparison (accuracy, log-loss, Brier score, draw behaviour) and a bootstrap check of whether the network really beats the Elo and logistic regression baselines. In my runs it only just beats a plain Elo model, so most of the predictive power comes from team strength.

## Example output

```
Germany win: 37.93%
Draw:           30.21%
France win: 31.86%
```

## Limitations

- The neural network is barely better than a simple Elo-based model, so don't expect a big edge from the extra complexity.
- Draws are the hardest outcome to predict. The probabilities look sensible, but a draw is rarely the single most likely result.
- Elo uses the same K factor for every match, so a friendly counts as much as a World Cup game.
- Form is a plain 5-match window, and the rest of the history only reaches the model through Elo.
- The train/validation/test dates are fixed, so results will shift a bit if you change them.
- `former_names.csv` needs to be kept up to date when countries are renamed.

## License
MIT
