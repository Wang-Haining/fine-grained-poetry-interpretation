## Dataset statistics

### Split sizes by source

| split      | source               |   rows |
|:-----------|:---------------------|-------:|
| train      | public_domain_poetry |  33820 |
| train      | poetry_foundation    |  12400 |
| validation | public_domain_poetry |   1875 |
| validation | poetry_foundation    |    693 |
| test       | public_domain_poetry |   1859 |
| test       | poetry_foundation    |    709 |


### Masking rates (Poetry Foundation)

| split      | source               |   poem_null_rate |   interpretation_null_rate |   rows |
|:-----------|:---------------------|-----------------:|---------------------------:|-------:|
| train      | public_domain_poetry |      5.91366e-05 |                          0 |  33820 |
| train      | poetry_foundation    |      1           |                          0 |  12400 |
| validation | public_domain_poetry |      0           |                          0 |   1875 |
| validation | poetry_foundation    |      1           |                          0 |    693 |
| test       | public_domain_poetry |      0           |                          0 |   1859 |
| test       | poetry_foundation    |      1           |                          0 |    709 |


### Text length (public-domain only; word counts)

| split      | field                |    mean |   median |   p05 |     p95 |   max |
|:-----------|:---------------------|--------:|---------:|------:|--------:|------:|
| train      | poem_words           | 420.71  |      164 |  32   | 1307.05 | 49750 |
| train      | interpretation_words | 667.2   |      639 | 484   |  929.05 |  3100 |
| validation | poem_words           | 438.113 |      172 |  30   | 1425.9  | 17084 |
| validation | interpretation_words | 672.097 |      645 | 477.7 |  943.3  |  2419 |
| test       | poem_words           | 410.409 |      169 |  31   | 1192.5  | 25257 |
| test       | interpretation_words | 664.167 |      643 | 482.9 |  905    |  2458 |


### Label coverage

| split      |   emotions_empty_rate |   themes_empty_rate |   themes_50_empty_rate |   themes_50_len_median |   themes_50_len_p95 |
|:-----------|----------------------:|--------------------:|-----------------------:|-----------------------:|--------------------:|
| train      |                     0 |         0.000194721 |             0.00227174 |                      5 |                   5 |
| validation |                     0 |         0.000778816 |             0.00350467 |                      5 |                   5 |
| test       |                     0 |         0.00116822  |             0.00389408 |                      4 |                   5 |


### Sentiment distribution

| split      | sentiment   |   count |       pct |
|:-----------|:------------|--------:|----------:|
| test       | negative    |    1406 | 0.547508  |
| test       | neutral     |     166 | 0.0646417 |
| test       | positive    |     996 | 0.38785   |
| train      | negative    |   26616 | 0.575855  |
| train      | neutral     |    2971 | 0.0642795 |
| train      | positive    |   16633 | 0.359866  |
| validation | negative    |    1490 | 0.580218  |
| validation | neutral     |     175 | 0.0681464 |
| validation | positive    |     903 | 0.351636  |


### Top themes_50 (overall)

| theme_50     |   count |
|:-------------|--------:|
| nature       |   19762 |
| death        |   19472 |
| love         |   15378 |
| loss         |   11800 |
| time         |    9949 |
| hope         |    9635 |
| grief        |    9047 |
| history      |    8891 |
| identity     |    7371 |
| religion     |    6642 |
| war          |    6209 |
| spirituality |    5996 |
| beauty       |    5920 |
| loneliness   |    5811 |
| existential  |    5674 |
| family       |    5150 |
| self         |    5005 |
| violence     |    3907 |
| politics     |    3659 |
| home         |    3644 |
