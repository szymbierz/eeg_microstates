import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

def visualize_uniform_distribution():
    # --- 1. Parametry Rozkładu Jednostajnego ---
    # a: Początek przedziału (min. błąd) = -5 minut
    # b: Koniec przedziału (max. błąd) = +5 minut
    
    a = -5
    b = 5
    loc = a         # Llokalizacja (początek przedziału)
    scale = b - a   # Skala (długość przedziału) = 10 minut
    
    x = np.linspace(a - 1, b + 1, 500) # Szerszy zakres dla osi X
    
    # --- 2. Wykres 1: Funkcja Gęstości Prawdopodobieństwa (PDF) ---
    pdf = uniform.pdf(x, loc, scale)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)
    
    ax1 = axes[0]
    ax1.plot(x, pdf, 'b-', lw=3)
    
    # Podkreślenie prostokątnego kształtu
    ax1.fill_between([a, b], [1/scale, 1/scale], color='skyblue', alpha=0.5) 
    
    ax1.set_title('Wykres 1: Funkcja Gęstości $f(x)$ (FGP)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Błąd pomiaru czasu [minuty]')
    ax1.set_ylabel('Gęstość Prawdopodobieństwa $f(x)$')
    ax1.set_ylim(0, 1.2 * (1/scale))
    ax1.set_xlim(a - 1, b + 1)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Etykieta wysokości
    ax1.text(0, 1/scale + 0.01, r'$\frac{1}{b-a} = \frac{1}{10} = 0.1$', 
             ha='center', color='darkblue', fontsize=12)
    
    # Wizualizacja prawdopodobieństwa w przedziale (np. P(-2 < X < 3))
    x_prob = np.linspace(-2, 3, 100)
    pdf_prob = uniform.pdf(x_prob, loc, scale)
    ax1.fill_between(x_prob, pdf_prob, color='red', alpha=0.3, 
                     label='Prawdopodobieństwo: P(-2 < X < 3)')
    ax1.legend(loc='upper left')

    # --- 3. Wykres 2: Dystrybuanta (CDF) ---
    cdf = uniform.cdf(x, loc, scale)
    
    ax2 = axes[1]
    ax2.plot(x, cdf, 'g-', lw=3)
    
    ax2.set_title('Wykres 2: Dystrybuanta $F(x)$ (CDF)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Błąd pomiaru czasu [minuty]')
    ax2.set_ylabel('Prawdopodobieństwo Skumulowane $F(x)$')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(a - 1, b + 1)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # Dodanie punktów charakterystycznych
    ax2.axvline(a, color='gray', linestyle='--', lw=1)
    ax2.axvline(b, color='gray', linestyle='--', lw=1)
    ax2.text(a, -0.05, f'a={a}', ha='center', color='gray')
    ax2.text(b, -0.05, f'b={b}', ha='center', color='gray')
    ax2.text(0, 0.5, 'Mediana (0)', ha='center', color='red', fontsize=12)

    plt.show()

if __name__ == '__main__':
    visualize_uniform_distribution()