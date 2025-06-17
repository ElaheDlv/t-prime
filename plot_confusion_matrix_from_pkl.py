#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot a confusion matrix from a pickle file, showing percentages per true class"
    )
    parser.add_argument(
        '--pkl', required=True,
        help="Path to the confusion matrix pickle"
    )
    parser.add_argument(
        '--labels', nargs='+', required=False,
        help="List of class labels in order"
    )
    parser.add_argument(
        '--cmap', default='Blues',
        help="Matplotlib colormap for the matrix"
    )
    args = parser.parse_args()

    # Load the confusion matrix
    cm = pickle.load(open(args.pkl, 'rb'))
    cm = np.array(cm, dtype=float)

    # Compute percentages
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, row_sums, where=row_sums!=0) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pct, interpolation='nearest', cmap=args.cmap)
    plt.title('Confusion Matrix (%)')
    plt.colorbar(im, ax=ax, format='%.1f')

    # Tick labels
    n = cm.shape[0]
    if args.labels and len(args.labels) == n:
        labels = args.labels
    else:
        labels = [str(i) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Annotate percentages
    fmt = lambda v: f"{v:.1f}%"
    for i in range(n):
        for j in range(n):
            val = pct[i, j]
            ax.text(j, i, fmt(val), ha='center', va='center', fontsize=8,
                    color='white' if val > pct.max()/2 else 'black')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
