# TAMU Datathon 2023: Patient Survival Prediction (1st Place)

**1st place, healthcare (TD Hospital) challenge, TAMU Datathon 2023 (October 2023).** Team 45: Arul Dhar, Taylor Smith and Bao Nguyen.

During the event we built a model that predicts whether a hospital patient survives from their medical record, scored on a hidden test set.

## Result

| Version | Hidden test-set accuracy |
| --- | --- |
| First submission (aggressive outlier removal) | 28% |
| Final submission (keep outliers, median imputation) | **64%** |

## What we learned: over-cleaning caused overfitting

The dataset had obviously bad values, such as ages between 150 and 200. Our first instinct was to drop every outlier row. That produced 97-100% accuracy on our own training data but only **28%** on the hidden test set: we had trained on a narrow slice of patients that didn't resemble the test set, and the test data itself could not be filtered the same way.

The fix was to keep the outliers and repair the data instead:

- missing values filled with each column's median
- duplicate rows removed
- features standardized with `StandardScaler`

That raised test accuracy to **64%**. The lesson carried into later work: cleaning choices are modelling choices, and only held-out data tells you whether they helped.

## Approach

1. **Exploration** (`code/dataextract.py`): histograms and scatter plots per feature, and median values for survivors vs. non-survivors, to spot outliers and useful signals.
2. **Feature selection:** 16 numeric features kept from that analysis (e.g. age, blood chemistry, glucose, temperature, reflex, cost). Categorical fields such as race, disability and DNR status were left out for lack of time to encode them properly.
3. **Model** (`code/neural_network.py`): a Keras sequential network (128 -> 64 -> 1, ReLU then sigmoid), binary cross-entropy, Adam, 20 epochs, with train/validation/test splits.
4. **Submission** (`code/TDHospitalSampleSubmission.py`): a small Flask service that loads the trained model and returns predictions in the format the organizers' scorer expected.

## Repository structure

| Path | Contents |
| --- | --- |
| `code/dataextract.py` | Exploratory analysis and outlier checks |
| `code/neural_network.py` | Preprocessing, training and evaluation |
| `code/TDHospitalSampleSubmission.py` | Flask prediction endpoint for submission |
| `docs/Datathon Presentation 2023.pdf` | Final presentation |
| `docs/original-plan.pdf` | Initial plan |
| `resources/data-science.md` | Reference links we used |

The competition dataset (`TD_HOSPITAL_TRAIN.csv`) belongs to the organizers and is not included.

## With more time

- Encode the categorical fields (race, disability, primary diagnosis, DNR) instead of dropping them
- Compare against simpler baselines such as logistic regression and tree ensembles
- Use cross-validation rather than a single split to choose features

## License

MIT. Copyright (c) 2023 Taylor Smith, Arul Dhar and Bao Nguyen.
