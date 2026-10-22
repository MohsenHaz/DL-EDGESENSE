"""Compare numbers in the manuscript tables (paper_artifacts/tables_as_reported) with (a) internal arithmetic and
(b) values regenerated from measurements (paper_artifacts/tables_generated), using claims.yaml.
Exit code 1 on any failure. Items that need measured data are reported as PENDING, never silently passed."""
import csv, os, re, sys, yaml

def num(s): return float(re.search(r"-?\d+(?:\.\d+)?", s.replace("–", "-").replace("−", "-")).group())
def rd(p): return list(csv.reader(open(p, newline="")))
R = "paper_artifacts/tables_as_reported"; G = "paper_artifacts/tables_generated"
fails, pend = [], []

def chk(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond: fails.append(name)

# --- arithmetic checks on the reported tables
t3 = {re.sub(r"\s*\[\d+\]", "", r[0]): r for r in rd(f"{R}/table03.csv")[1:]}
ref = num(t3["DeepSense"][1])
for n, r in t3.items():
    chk(f"Table 3 savings vs DeepSense: {n}", abs((ref - num(r[1])) / ref * 100 - num(r[3])) < 0.06)
t19 = rd(f"{R}/table19.csv")[1:]; full = num(t19[0][2])
for r in t19[1:]:
    e = num(r[2]); chg = num(r[5].replace("%", ""))
    chk(f"Table 19 energy change: {r[0]}", abs((full - e) / full * 100 - chg) < 0.06, f"table {chg}, formula {(full-e)/full*100:.1f}")
    if r[7] not in ("—", ""):
        chk(f"Table 19 energy saved: {r[0]}", abs((e - full) / e * 100 - num(r[7])) < 0.06, f"table {r[7]}, formula {(e-full)/e*100:.1f}")
t8 = rd(f"{R}/table08.csv"); hdr = t8[0]; row = {r[0]: r for r in t8[1:]}
e = row["Energy per inference [mJ]"]; j = row["Energy per inference [J]"]
for i in range(1, len(e)):
    chk(f"Table 8 mJ/J consistent col {hdr[i]}", abs(num(e[i]) / 1000 - num(j[i])) < 1e-4)
lat = row["Inference Latency [ms]"]; ip = row["Mean In-Inference Power, E/latency [mW]†"]
for i in range(1, len(e)):
    chk(f"Table 8 in-inference power = E/latency col {hdr[i]}", abs(num(e[i]) / num(lat[i]) * 1000 - num(ip[i])) < 1.0)
# same operating point must have the same latency / energy in every table
t4 = {r[0]: r for r in rd(f"{R}/table04.csv")[1:]}; t12 = rd(f"{R}/table12.csv")[1:]
chk("Table 4 DL-EgSense energy == Table 12 High-Energy energy", num(t4["DL-EgSense"][1]) == num(t12[0][4]))
chk("Table 4 DL-EgSense FLOPs == Table 12 High-Energy FLOPs", num(t4["DL-EgSense"][5]) == num(t12[0][2]),
    f"Table 4 {t4['DL-EgSense'][5]} M vs Table 12 {t12[0][2]} M")
# Table 4/5 report the end-to-end pipeline (incl. runtime I/O buffers); Tables 8/12 report the model-only values of the mode
chk("Table 4 end-to-end latency >= Table 8 High-Energy model-only latency", num(t4["DL-EgSense"][3]) >= num(lat[3]))
chk("Table 4 end-to-end memory >= Table 12 High-Energy model memory", num(t4["DL-EgSense"][4]) >= num(t12[0][3]))
t17 = rd(f"{R}/table17.csv")[1:]
jet = [r for r in t17 if r[0].lower().startswith("jetson")][0]
chk("Table 8 Jetson energy == Table 17 Jetson measured energy", num(e[2]) == num(jet[2]), f"{e[2]} vs {jet[2]}")
for r in t17:
    chk(f"Table 17 error recomputed: {r[0]}", abs(abs(num(r[1]) - num(r[2])) / num(r[2]) * 100 - num(r[3])) < 0.06,
        f"table {r[3]}, recomputed {abs(num(r[1])-num(r[2]))/num(r[2])*100:.1f}")
# --- claims.yaml vs measured
claims = yaml.safe_load(open("claims.yaml"))["claims"]
if not os.path.exists(f"{G}/table03.csv"):
    pend.append("all measured-vs-reported checks (no measured tables: add traces, run `make aggregate tables`)")
else:
    g3 = {r[0]: r for r in rd(f"{G}/table03.csv")[1:]}
    for c in claims:
        if c.get("measured_key") and c["measured_key"] in g3:
            v = num(g3[c["measured_key"]][1]); chk(f"claim {c['id']}: {c['text']}", abs(v - c["value"]) <= c.get("tol", 0.5), f"measured {v}")
for p in pend: print("PENDING", p)
print(f"\n{len(fails)} failed, {len(pend)} pending"); sys.exit(1 if fails else 0)
