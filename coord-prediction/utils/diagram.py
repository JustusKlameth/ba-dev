import numpy as np
import matplotlib.pyplot as plt


def plot_lorenz(ax, df, name, s=3):
    """
    Plottet die Lorenz-Kurve der gegebenen Daten in dem gegebenen Subplot.

    Args:
        ax (matplotlib.axes): Der Subplot in dem die Daten geplottet werden sollen.
        df (Pandas Dataframe): Die Daten.
    """
    # Daten filtern und sortieren
    errors = df["Error"].values
    errors_without_nan = errors[~np.isnan(errors)]
    errors_without_nan_sorted = np.sort(errors_without_nan)

    # Lorenz-Fehler berechnen
    y_lorenz = errors_without_nan_sorted.cumsum() / errors_without_nan_sorted.sum()
    y_lorenz = np.insert(y_lorenz, 0, 0)

    xs = np.arange(y_lorenz.size) / (y_lorenz.size - 1)

    # Lorenz-Kurve
    ax.scatter(xs, y_lorenz, s=s, label='Lorenz-Kurve')

    # Gleichverteilung
    ax.plot([0, 1], [0, 1], color='k', label='Gleichverteilung')

    # Label
    ax.set_xlabel("Anteil der Antworten\n(Prozent)")
    ax.set_ylabel("Anteil des Fehlers\n(Prozent)")
    ax.set_title(name)
    ax.label_outer()

    # 10% Fehler Markierung
    for (x, y) in zip(xs, y_lorenz):

        if (y >= 0.1):
            ax.axvline(x=x, color='r', label='Anteil der Fragen, die 10 % des Fehlers verursachen')
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks([x])
            break