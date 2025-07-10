import os
import glob
import matplotlib.pyplot as plt

def plot_clusters(cluster_dir="../clusters", pattern="cluster*.txt", out_file="../output/clusters_plot.png"):
    """
    Reads all files matching `pattern` in `cluster_dir` and plots
    their (x,y) points on a single scatter plot.
    """
    # find all cluster files
    file_paths = sorted(glob.glob(os.path.join(cluster_dir, pattern)))
    if not file_paths:
        raise FileNotFoundError(f"No files matching {pattern} in {cluster_dir}")

    plt.figure(figsize=(8, 6))

    # a short list of markers and colors to cycle through
    markers = ['o', 's', 'D', '^']
    colors  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for idx, file_path in enumerate(file_paths):
        # load data skipping header
        data = []
        with open(file_path, 'r') as f:
            header = next(f)  # skip header
            for line in f:
                parts = line.strip().split(',')
                # [vic_id, x, y, ...]
                if len(parts) < 3:
                    continue
                x = float(parts[1])
                y = float(parts[2])
                data.append((x, y))

        xs, ys = zip(*data)
        label = os.path.splitext(os.path.basename(file_path))[0]
        plt.scatter(xs, ys,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    label=label,
                    alpha=0.7,
                    edgecolor='k')

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Victim Clusters")
    plt.legend(title="Cluster files")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=150)
    print(f"Saved cluster plot to {out_file}")

if __name__ == "__main__":
    plot_clusters()
