import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Accuracy results from model evaluation
model_names = ["Text - XGBoost", "Text - Random Forest", "Image - XGBoost", "Image - Random Forest", "Merged Model"]
accuracies = [0.74, 0.72, 0.91, 0.91, 0.62]  # Replace these values with actual accuracies

# ✅ Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

# ✅ Create bar plot
ax = sns.barplot(x=model_names, y=accuracies, palette="viridis")

# ✅ Annotate bars with accuracy values
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")

# ✅ Labels & title
plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy Comparison of Different Models", fontsize=14)

# ✅ Rotate x-axis labels for readability
plt.xticks(rotation=15)

# ✅ Save and show the plot
plt.savefig("datasets/accuracy_comparison_chart.png")  # Saves the figure
plt.show()
