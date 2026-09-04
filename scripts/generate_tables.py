"""Generate all result tables as CSV."""
import csv, os; os.makedirs("results/tables", exist_ok=True)

def table_1():
    rows = [
        {"Framework":"DeepSense","Year":2017,"Energy_mJ":146,"F1_%":91.5},
        {"Framework":"EdgeML","Year":2020,"Energy_mJ":128.2,"F1_%":75.3},
        {"Framework":"Federated-Sense","Year":2021,"Energy_mJ":138.7,"F1_%":84.5},
        {"Framework":"EfficientSensing","Year":2021,"Energy_mJ":111.0,"F1_%":89.7},
        {"Framework":"DL-EgSense","Year":2025,"Energy_mJ":51.0,"F1_%":92.7},
    ]
    with open("results/tables/table_1_comparison.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("Table 1 saved.")

if __name__ == "__main__":
    table_1()
    print("Tables saved to results/tables/")
