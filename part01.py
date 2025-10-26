#!/usr/bin/env python3
"""
IZV cast1 projektu 2025
Autor: Rastislav Uhliar (xuhliar00)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene
na prednasce
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def wave_inference_bad(
    x: NDArray[any], y: NDArray[any], sources: NDArray[any], wavelength: float
) -> NDArray[any]:
    """
    Referencni implementace, ktera je pomala a nevyuziva numpy efektivne;
    nezasahujte do ni!
    """
    k = 2 * np.pi / wavelength
    Z = np.zeros(x.shape + y.shape)
    for sx, sy in sources:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                R = np.sqrt((x[i] - sx) ** 2 + (y[j] - sy) ** 2)
                Z[j, i] += np.cos(k * R) / (1 + R)
    return Z


def wave_inference(
    x: NDArray[any], y: NDArray[any], sources: NDArray[any], wavelength: float
) -> NDArray[any]:
    """
    Calculate the wave interference from multiple sources

    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        sources: 2D Array of source positions
        wavelength: Wavelength of the wave

    Returns:
        2D array of wave amplitudes
    """
    k = 2 * np.pi / wavelength

    # 2D array created using meshgrid
    X, Y = np.meshgrid(x, y)
    # New dimension added to change the shape for broadcasting
    X = X[:, :, np.newaxis]
    Y = Y[:, :, np.newaxis]

    # Distance calculation with broadcasting
    dx = X - sources[:, 0]
    dy = Y - sources[:, 1]

    # Distance of point from the source
    d = np.sqrt(dx**2 + dy**2)

    # Wave calculateion
    wave = np.cos(k * d) /  (1 + d)

    # Sum of all sources for each point
    Z = np.sum(wave, axis=2)
    return Z

def plot_wave(
    Z: NDArray[any],
    x: NDArray[any],
    y: NDArray[any],
    show_figure: bool = False,
    save_path: str | None = None,
):
    """
    Plots the wave interference as a 2D map 

    Args:
        Z: 2D wave of amplitudes
        x: X Coordinates for the Z columns
        y: Y Coordinates for the Z rows
        show_figure: Displays the plot
        save_path: Saves file to the provided path 
    """
    fig, ax = plt.subplots()
    im = ax.imshow(
        Z,
        extent=(x.min(), x.max(), y.min(), y.max()),
        # Changed where the graph starts withouth this it would be upside down
        origin="lower",
        cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="Amplituda vlny")

    ax.set_xlabel("X pozice")
    ax.set_ylabel("Y pozice")
    ax.set_title("Vlnové pole")

    if save_path:
        plt.savefig(save_path)

    if show_figure:
        plt.show()
    
    plt.close(fig)


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generates and plots visualization of sine and cosine functions in two graphs. First one colors the region between the curves and the second one shows the maximum values with colors and minimum values with grey dashed line

    Args:
        show_figure: Displays the plot
        save_path: Saves file to the provided path 
    """ 
    # show_figure = True

    x = np.linspace(0, 4 * np.pi, 1000)

    sin_x = np.sin(x)
    cos_x = np.cos(x)

    min_vals = np.minimum(sin_x, cos_x)
    max_vals = np.maximum(sin_x, cos_x)

    cos_is_max = cos_x >= sin_x

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x, sin_x, color="grey")
    ax1.plot(x, cos_x, color="grey")

    ax1.fill_between(x, min_vals, max_vals, where=cos_is_max, color="green", alpha=0.3) # alpha for the color opacity
    ax1.fill_between(x, min_vals, max_vals, where=~cos_is_max, color="green", alpha=0.3)

    # Makes sure the 0 is on the left corner
    ax1.margins(x=0)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_ylabel("f(x)")

    # Removes the edges
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)


    # Arrays of maximum values
    max_cos = np.where(cos_is_max, max_vals, np.nan)
    max_sin = np.where(~cos_is_max, max_vals, np.nan)
    
    ax2.plot(x, max_cos, color='orange')
    ax2.plot(x, max_sin, color='blue')
    ax2.plot(x, min_vals, color='gray', linestyle='--')

    ax2.margins(x=0)
    ax2.set_ylabel("f(x)")
    ax2.set_xlabel(r'$x$')
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel("")

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


    # Ticks locations and labes for x axis
    ticks = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]) * np.pi
    labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', 
                 r'$2\pi$', r'$\frac{5\pi}{2}$', r'$3\pi$', r'$\frac{7\pi}{2}$', 
                 r'$4\pi$']
    
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)

    if save_path:
        plt.savefig(save_path)
    
    if show_figure:
        plt.show()



def download_data() -> Dict[str, List[Any]]:
    """
    Scrapes weather station website and extracts station names, latitudes, longtitudes and heights.
    
    Returns:
        Dictionary with four keys for each scraped variable.
    """

    # url = "https://ehw.fit.vutbr.cz/izv/stanice.html"
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"

    response = requests.get(url)
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table')
    table = tables[1]
    # print(table)
    rows = table.find_all('tr')
    # print(rows)

    positions = []
    lats = []
    longs = []
    heights = []

    # The first row contains table column names therefore i am skiping it
    for row in rows[1:]:
        cells = row.find_all('td')
        # print(cells)
        positions.append(cells[0].text.strip())
        # print(positions)
        lats.append(float(cells[2].text.strip().replace(',', '.').replace('°', '')))
        # print(lats)
        longs.append(float(cells[4].text.strip().replace(',', '.').replace('°', '')))
        # print(longs)
        heights.append(float(cells[6].text.strip().replace(',', '.').replace('°', '')))
        # print(heights)
    
            
    return {
        'positions': positions,
        'lats': lats,
        'longs': longs,
        'heights': heights
    }


if __name__ == "__main__": # changed main to __main__ so it runs
    X = np.linspace(-10, 10, 200)
    Y = np.linspace(-10, 10, 200)
    A = wave_inference_bad(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]), 2)
    plot_wave(A, X, Y, show_figure=True)
    generate_sinus()
