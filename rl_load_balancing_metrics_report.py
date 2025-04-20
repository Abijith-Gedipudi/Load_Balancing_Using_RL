for method in methods:
    print(f"{method}:\n"
          f"  Avg CPU Utilization: {np.mean(metrics[method]['cpu_util']):.2f}%\n"
          f"  Avg Response Time: {np.mean(metrics[method]['response_time']):.2f} ms\n"
          f"  Avg Load Balance (Std Dev): {np.mean(metrics[method]['load_balance']):.2f}\n")
