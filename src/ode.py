import numpy as np

def edo_duffing_soroll(t, z, random, epsilon, funcio_soroll):
    """Paràmetres:
        t: temps
        z: posició al pla R^2
        random: nombre real aleatori entre 0 i 2
        epsilon: paràmetre entre 0 i 1 que regula el soroll
        funcio_soroll: pertorbació que depèn del temps
    Retorna el camp vectorial del sistema d'EDOs x'=y+epsilon*random*f(t); y'=x-x^3.
    """
    x, y = z
    soroll = epsilon *  random * funcio_soroll(t)
    return [y + soroll, x - x**3]


def edo_duffing_no_autonom(t, z):
    """Paràmetres:
        t: temps
        z: posició al pla R^2
    Retorna el camp vectorial del sistema d'EDOs 
    x'=-y; y'=-f(t)(x-ax^3);
    on f(t) = 1 - epsilon * cos(omega * t + phi)
    """
    x, y = z
    a = 0.5
    
    def epsilon(x):
        return 0.1 * (x + 4.5) / 2
    
    def f(t, x):
        omega = 3 * np.pi / 2
        phi = np.pi / 4
        return 1 - epsilon(x) * np.cos(omega * t + phi)
    
    return [-y, - f(t, x) * (x - a * x**3)]


def edo_duffing(t, z):
    """Paràmetres:
        t: temps
        z: posició al pla R^2
    Retorna el camp vectorial del sistema d'EDOs x'=y; y'=x-x^3.
    """
    x, y = z
    return [y, x - x**3]


def edo_bickley_jet(t, z):
    """Paràmetres:
        t: temps
        z: posició al pla R^2
    Retorna el camp vectorial del sistema d'EDOs de Bickley jet.
    """
    x, y = z
    
    U0 = 62.66 # m/s velocitat del jet
    L = 1770 # m amplada del jet
    c1 = 0.205*U0
    c2 = 0.461*U0
    radi_Terra = 6371000 # m
    k1 = 2/radi_Terra # nombre d'ona
    k2 = 2*k1
    A1 = 0.15 # amplitud d'ona
    A2 = 0.30

    def stream0(y):
        return - U0*L*np.tanh(y/L)
    
    def stream1(x, y, t):
        sech = 1/np.cosh(y/L)
        suma = A1*np.cos(k1*x - c1*k1*t) + A2*np.cos(k2*x - c2*k2*t)
        return U0 * L * sech * suma
    
    def stream_function(x, y, t):
        return stream0(y) + stream1(x, y, t)
    
    camp_vectorial = c2*y + stream_function(x, y, t)
    return camp_vectorial