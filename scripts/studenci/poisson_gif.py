import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.special import factorial


def poisson_pmf(k_values, lambda_rate):
    """Zwraca wartości PMF rozkładu Poissona dla wektora k oraz parametru lambda."""
    k_values = np.asarray(k_values)
    return np.exp(-lambda_rate) * np.power(lambda_rate, k_values) / factorial(k_values)


def generate_poisson_er_gif(lambda_rate=3, steps_per_bar=10, outfile="poisson_izba_przyjec.gif"):
    """Generuje GIF prezentujący budowanie rozkładu Poissona dla kontekstu SOR.

    - lambda_rate: średnia liczba pacjentów na godzinę (λ)
    - steps_per_bar: liczba klatek na jeden słupek (dla efektu wzrostu)
    - outfile: nazwa zapisywanego GIF-a
    """

    # Zakres k dobrany tak, aby objąć masę prawdopodobieństwa (~ do λ + 4√λ)
    k_max = int(np.ceil(lambda_rate + 4 * np.sqrt(lambda_rate)))
    k_values = np.arange(0, k_max + 1)
    pmf_values = poisson_pmf(k_values, lambda_rate)

    # Przygotowanie wykresu
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18, top=0.88)

    # Słupki w jednym kolorze
    bar_color = "#4C78A8"
    bars = ax.bar(k_values, np.zeros_like(k_values, dtype=float), color=bar_color, width=0.8, edgecolor="#333333")

    ax.set_xlim(-0.5, k_max + 0.5)
    ax.set_ylim(0, max(pmf_values) * 1.25)
    ax.set_xlabel("Liczba pacjentów w godzinę (k)")
    ax.set_ylabel("Prawdopodobieństwo")

    title_text = ax.set_title("Izba przyjęć: rozkład Poissona", fontsize=14, fontweight="bold")

    # Wzórś
    formula_text = ax.text(
        0.02,
        0.95,
        r"$P(K=k)=\dfrac{e^{-\lambda}\lambda^{k}}{k!}$",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        color="#222222",
    )

    subtitle = ax.text(
        0.02,
        0.85,
        "Średnio przyjmuje się λ = 3 pacjentów/godz. (noc na SOR)",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color="#444444",
    )

    # Linia/krzywa nad słupkami (drugi etap)
    line_handle, = ax.plot([], [], color="#E45756", lw=2, marker="o", ms=4, zorder=3)

    bars_frames = (k_max + 1) * steps_per_bar
    curve_frames = (k_max + 1)
    total_frames = bars_frames + curve_frames

    def init():
        for bar in bars:
            bar.set_height(0.0)
        return tuple(bars)

    def animate(frame_idx):
        if frame_idx < bars_frames:
            current_bar = frame_idx // steps_per_bar
            within_bar_step = frame_idx % steps_per_bar
            growth = (within_bar_step + 1) / steps_per_bar

            # Ustaw wysokości: pełna wysokość dla poprzednich, rosnąca dla bieżącego, 0 dla kolejnych
            for i, bar in enumerate(bars):
                if i < current_bar:
                    bar.set_height(pmf_values[i])
                elif i == current_bar:
                    bar.set_height(pmf_values[i] * growth)
                else:
                    bar.set_height(0.0)

            # Linia ukryta w pierwszym etapie
            line_handle.set_data([], [])
        else:
            # Etap 2: dorysowanie krzywej na słupkach
            for i, bar in enumerate(bars):
                bar.set_height(pmf_values[i])
            curve_count = frame_idx - bars_frames + 1
            curve_count = int(np.clip(curve_count, 0, len(k_values)))
            line_handle.set_data(k_values[:curve_count], pmf_values[:curve_count])

        return tuple(bars) + (title_text, formula_text, subtitle, line_handle)

    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=60, blit=True)

    # Zapis GIF
    writer = PillowWriter(fps=max(8, int(1000 / 60)))
    anim.save(outfile, writer=writer)

    return outfile


if __name__ == "__main__":
    path = generate_poisson_er_gif(lambda_rate=3, steps_per_bar=10, outfile="poisson_izba_przyjec.gif")
    print(f"Zapisano GIF: {path}")


