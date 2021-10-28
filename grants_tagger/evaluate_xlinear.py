from sklearn.metrics import precision_recall_fscore_support
import scipy.sparse as sp
import typer


def evaluate_xlinear(y_test_path, y_pred_path):
    Y_test = sp.load_npz(y_test_path)
    Y_pred_proba = sp.load_npz(y_pred_path)
    
    for th in [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        Y_pred = Y_pred_proba > th
        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        print(f"Th: {th} P: {p} R: {r} f1: {f1}")

if __name__ == "__main__":
    typer.run(evaluate_xlinear)
