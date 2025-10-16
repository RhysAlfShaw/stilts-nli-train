import pandas
import matplotlib.pylab as plt

# link to stylefile

style_file_location = "/home/rhys/work/mplrc_sotiria.mplstyle"
# set style
plt.style.use(style_file_location)
# open data
df_train_27b = pandas.read_csv("train_loss_27b.csv")
df_eval_27b = pandas.read_csv("eval_loss_27b.csv")
df_train_2b = pandas.read_csv("train_loss.csv")
df_eval_2b = pandas.read_csv("eval_loss.csv")

min_loss_27b = df_eval_27b["loss"].min()
min_loss_2b = df_eval_2b["loss"].min()
step_min_loss_2b = df_eval_2b.loc[df_eval_2b["loss"].idxmin(), "step"]
step_min_loss_27b = df_eval_27b.loc[df_eval_27b["loss"].idxmin(), "step"]
fig, ax = plt.subplots(2, 1, figsize=(6, 8))

ax[1].plot(
    df_train_27b["step"],
    df_train_27b["loss"],
    label="Training Loss",
    color="grey",
    zorder=1,
)
ax[1].plot(
    df_eval_27b["step"],
    df_eval_27b["loss"],
    label="Evaluation Loss",
    color="black",
    # marker="o",
    linestyle="--",
    markersize=5,
    zorder=2,
)
ax[1].scatter(
    x=step_min_loss_27b,
    y=min_loss_27b,
    marker="*",
    s=100,
    color="black",
    label="Best Model",
    zorder=3,
)
ax[1].set_xlabel("Training Steps")
ax[1].set_ylabel("Cross Entropy Loss")
ax[1].set_yscale("log")
ax[1].set_title("27B Model")
ax[1].legend()
ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)
ax[0].plot(
    df_train_2b["step"],
    df_train_2b["loss"],
    label="Training Loss",
    color="grey",
    zorder=1,
)
ax[0].plot(
    df_eval_2b["step"],
    df_eval_2b["loss"],
    label="Evaluation Loss",
    color="black",
    # marker="o",
    linestyle="--",
    markersize=5,
    zorder=2,
)
ax[0].scatter(
    x=step_min_loss_2b,
    y=min_loss_2b,
    marker="*",
    s=100,
    color="black",
    label="Best Model",
    zorder=3,
)
ax[0].set_xlabel("Training Steps")
ax[0].set_ylabel("Cross Entropy Loss")
ax[0].set_yscale("log")
ax[0].set_title("2B Model")
ax[0].legend()
ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)

plt.savefig("loss_eval_plot.pdf")
