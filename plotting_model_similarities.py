from betterplots import pyplot as plt

import betterplots.betterstyle as bs

bs.set_style("betterstyle")

# load the results json file
import json

results_2b_path = "results_2b.json"
results_27b_path = "results_27b.json"
with open(results_2b_path, "r") as f:
    results_2b = json.load(f)

with open(results_27b_path, "r") as f:
    results_27b = json.load(f)
# plot histogram of scores
scores_2b = [res["score"] for res in results_2b]
scores_27b = [res["score"] for res in results_27b]
# for each cmd category, plot histogram of scores
categories_2b = list(set(res["cmd"] for res in results_2b))
categories_27b = list(set(res["cmd"] for res in results_27b))
num_categories_2b = len(categories_2b)
num_categories_27b = len(categories_27b)

fig, axs = plt.subplots(
    figsize=(12, 9),
    nrows=4,
    ncols=3,
    constrained_layout=True,
    sharex=True,
    # sharey=True,
)
for i, category in enumerate(sorted(categories_2b)):
    cat_scores_2b = [res["score"] for res in results_2b if res["cmd"] == category]
    cat_scores_27b = [res["score"] for res in results_27b if res["cmd"] == category]
    if num_categories_2b > 1:
        ax = axs[i // 3, i % 3]
    else:
        ax = axs
    ax.hist(
        cat_scores_2b, bins=10, range=(0, 1), edgecolor="black", alpha=0.6, label="2B"
    )
    ax.hist(
        cat_scores_27b,
        bins=10,
        range=(0, 1),
        fill=False,
        edgecolor="black",
        # alpha=0.5,
        linestyle="dashed",
        label="27B",
    )
    # add text box with legend inside the plot
    ax.legend(loc="upper right", bbox_to_anchor=(0.3, 0.90), fontsize=10, frameon=False)
    ax.text(
        0.05,
        0.95,
        f"Task: {category} (n={len(cat_scores_2b)})",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    # ax.(f"Task: {category} (n={len(cat_scores_2b)})")
    # ax.set_xlabel("Cosine Similarity Score (%)")
    # ax.set_ylabel("Frequency")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    # ax.set_yticks([])

# axs[0, 0].set_xticks([0, 20, 40, 60, 80, 100])
axs[3, 0].set_xticks([0.10, 0.30, 0.50, 0.70, 0.90])
axs[3, 0].set_xlabel("Cosine Similarity", fontsize=13, weight="bold")
axs[3, 1].set_xticks([0.10, 0.30, 0.50, 0.70, 0.90])
axs[3, 1].set_xlabel("Cosine Similarity", fontsize=13, weight="bold")
axs[3, 2].set_xticks([0.10, 0.30, 0.50, 0.70, 0.90])
axs[3, 2].set_xlabel("Cosine Similarity", fontsize=13, weight="bold")

# axs[0, 0].set_yticks([10, 30, 50])
axs[0, 0].set_ylabel("N", fontsize=13, weight="bold")
# axs[1, 0].set_yticks([10, 30, 50, 70, 90])
axs[1, 0].set_ylabel("N", fontsize=13, weight="bold")
# axs[2, 0].set_yticks([10, 30, 50])
axs[2, 0].set_ylabel("N", fontsize=13, weight="bold")
# axs[3, 0].set_yticks([10, 30, 50, 70])
axs[3, 0].set_ylabel("N", fontsize=13, weight="bold")

# ax.grid(ais="y")
# plt.suptitle("Histogram of Command Cosine Similarity Scores by Category", fontsize=16)
plt.savefig("similarity_histograms_by_task.pdf")
plt.show()

# plot an overall histogram of scores
plt.figure(figsize=(6, 4))
plt.hist(scores_2b, bins=20, range=(0, 1), edgecolor="black", alpha=0.6)
plt.hist(
    scores_27b,
    bins=20,
    range=(0, 1),
    fill=False,
    edgecolor="black",
    linestyle="dashed",
)
plt.legend(["2B", "27B"])
# plt.title("Overall Command Cosine Similarity Scores")
plt.xlabel("Cosine Similarity", fontsize=13, weight="bold")
plt.ylabel("N", fontsize=13, weight="bold")
plt.xlim(0, 1)
plt.savefig("similarity_histogram_overall.pdf")
plt.show()

# print mean and stddev of scores
import numpy as np

mean_2b = np.mean(scores_2b)
std_2b = np.std(scores_2b)
mean_27b = np.mean(scores_27b)
std_27b = np.std(scores_27b)
print(
    f"2B: Mean={mean_2b:.4f}, StdDev={
std_2b:.4f}"
)
print(f"27B: Mean={mean_27b:.4f}, StdDev={std_27b:.4f}")
