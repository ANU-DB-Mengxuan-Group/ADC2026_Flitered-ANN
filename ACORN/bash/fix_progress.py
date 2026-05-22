import csv
import json

dataset = "ytb_video"
base = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_{dataset}"
summary_file = f"{base}/results/{dataset}/summary.csv"
progress_file = f"{base}/progress.json"

progress = {}

with open(summary_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        M = int(float(row['M']))
        M_beta = int(float(row['M_beta']))
        gamma = int(float(row['gamma']))
        scenario = row['scenario']
        build_key = json.dumps([M, M_beta, gamma, None])
        if build_key not in progress:
            progress[build_key] = {'build_done': True}
        search_key = json.dumps([M, M_beta, gamma, scenario])
        progress[search_key] = {'search_done': True}

with open(progress_file, 'w') as f:
    json.dump({'progress': progress}, f, indent=2)

print("Done")