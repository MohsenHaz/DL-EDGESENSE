"""Transcribe the tables of a manuscript .docx into paper_artifacts/tables_as_reported/*.csv (provenance record).
These CSVs are the numbers *as reported in the manuscript*. They are what the figure scripts use until measured
tables exist (paper_artifacts/tables_generated/), and what scripts/verify_numbers.py compares against."""
import csv, sys, os, docx

MAP = {2: "table03", 3: "table04", 4: "table05", 5: "table06", 6: "table07", 7: "table08",
       9: "table10", 10: "table11", 11: "table12", 16: "table17", 18: "table19"}

def main(path, out="paper_artifacts/tables_as_reported"):
    d = docx.Document(path); os.makedirs(out, exist_ok=True)
    for i, name in MAP.items():
        with open(f"{out}/{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            for r in d.tables[i].rows:
                w.writerow([c.text.strip() for c in r.cells])

if __name__ == "__main__":
    main(sys.argv[1])
