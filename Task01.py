import pandas as pd

data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"],
]

df = pd.DataFrame(data, columns=["Outlook", "Temperature", "Humidity", "Wind", "Play"])

for feature in ["Outlook", "Temperature", "Humidity", "Wind"]:
    print(f"\n--- Частотна таблиця для {feature} ---")
    print(pd.crosstab(df[feature], df["Play"]))

print("\n\n=== Ймовірності P(значення | клас) ===")

for feature in ["Outlook", "Temperature", "Humidity", "Wind"]:
    freq = pd.crosstab(df[feature], df["Play"])
    p_yes = (freq["Yes"] / freq["Yes"].sum()).round(2)
    p_no = (freq["No"] / freq["No"].sum()).round(2)
    prob_table = pd.DataFrame({
        "P(значення | Yes)": p_yes,
        "P(значення | No)": p_no
    })
    print(f"\n{feature}:")
    print(prob_table)

p_yes = (df["Play"] == "Yes").mean()
p_no = 1 - p_yes
print(f"\nАпріорні ймовірності: P(Yes) = {p_yes:.2f}, P(No) = {p_no:.2f}")
