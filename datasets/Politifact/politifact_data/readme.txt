Data File format:
-tab separated
-Truth-Ratings are listed as integers from 0-5 (0 is True, 5 is Pants-on-Fire False)

-Row in full_politifact and direct_quotes_only: 
Speaker    Truth-Rating[0-5]    Statement    PolitiFact Explanation    URL

-Row in train, dev, test files: 
Speaker    Statement     Truth-Rating[0-5]


List of files:
-full_politifact_data.csv: the complete set of statements crawled
-direct_quotes_only.csv: filters out all of the statements that are paraphrased from speakers
-train/dev/test: our split of the data with each speaker's quotes all in the same partition