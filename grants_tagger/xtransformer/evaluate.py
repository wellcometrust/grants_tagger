import json

from sklearn.metrics import precision_recall_fscore_support
import scipy.sparse as sp
import typer


def evaluate(y_test_path, y_pred_path, results_path, pr_curve_path):
    Y_test = sp.load_npz(y_test_path)
    Y_pred_proba = sp.load_npz(y_pred_path)

    pr_curve = []
    for th in [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        Y_pred = Y_pred_proba > th
        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        pr_curve.append({"precision": p, "recall": r, "f1": f1, "threshold": th})
        print(f"Th: {th} P: {p} R: {r} f1: {f1}")

    with open(pr_curve_path, "w") as f:
        f.write(json.dumps({"pr_curve": pr_curve}))

    best_metrics = max(pr_curve, key=lambda x: x["f1"])
    with open(results_path, "w") as f:
        f.write(json.dumps(best_metrics))


if __name__ == "__main__":
    typer.run(evaluate)
