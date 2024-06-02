## Processed data structure

All processed data should be in the form of a `csv` file with the following columns:

- `index`: index corresponding to the original ordering of the dataset. This index enables merging with original data set (with additonal attributes/columns).
- `text_a`: first text, can be any string.
- `text_b`: second text, can be any string.
- `preferred_text`: which text is preferred. Can be one of
    - `text_a`: text a is preferred.
    - `text_b`: text b is preferred.
    - `tie`: both texts are equally good. Note: not all data sets contain ties.
    - `tie (bothbad)`: both texts are too bad to judge. Note: not all data sets contain ties.