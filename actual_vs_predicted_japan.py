import matplotlib.pyplot as plt

# Known / reference magnitude (example: strong Japan earthquake)
actual_magnitude = 6.3   # known large earthquake
predicted_magnitude = 4.56

labels = ["Actual Magnitude", "Predicted Magnitude"]
values = [actual_magnitude, predicted_magnitude]

plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.ylabel("Magnitude")
plt.title("Actual vs Predicted Earthquake Magnitude")

for i, v in enumerate(values):
    plt.text(i, v + 0.05, f"{v}", ha='center')

plt.tight_layout()
plt.show()
