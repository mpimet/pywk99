import numpy as np

def plot_moisture_mode_number(ax, mode_num = 1, **kwargs):
    w_min, w_max = ax.get_ylim() 
    k_min, k_max = ax.get_xlim()  
    EARTH_RADIUS =  6371000.0 # m
    SECONDS_PER_DAY = 86400
    m_const = 0.01 # From Adames, JAS, 2019
    w_cpd = np.linspace(w_min, w_max, 500)
    k_num = np.linspace(k_min, k_max, 500)
    w_mode_cpd = SECONDS_PER_DAY * np.sqrt(mode_num / m_const * (k_num / (2 * np.pi * EARTH_RADIUS))**2)
    ax.plot(k_num, w_mode_cpd, **kwargs)