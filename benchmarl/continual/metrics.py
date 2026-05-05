import numpy as np


def compute_average_success(R):
    final_row = list(R.values())[-1]
    return np.mean(list(final_row.values()))


def compute_forgetting(R):

    tasks = list(list(R.values())[0].keys())

    forgetting = []

    for task in tasks:

        past_scores = [row.get(task) for row in R.values() if task in row]

        max_past = max(past_scores[:-1])
        final = past_scores[-1]

        forgetting.append(max_past - final)

    return np.mean(forgetting)