# aircraft-trajectory-prediction
Repository for the DEAL trajectory prediction study

# How to run

To run the different parts of the evaluation, follow the next steps:
* Run store-flights.R to sample the flights stored in the RData file and store them as R tensors.
* Run create-numpy-files.R to transform the former output into .npy files.
* Run find-turns.py to store a list of the flights with more accumulated directional changes.
* Run the .py files with the name of a model to store the results obtained with such model (e.g. "ground-truth.py", "lstm-diff-phase.py", etc.).
* Run metrics.py to compute the metrics obtained by each technique for each trajectory input size.
* Run join-summary.py to create a final csv file containing the concatenation of all metric files.
* Run file plots.R to generate plots.
