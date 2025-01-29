import math

def exp_scheduler(epoch, epochs_max, coef_min=1e-6, coef_max=1, k=-2):
    t = epoch/(epochs_max)
    return (math.exp(-k*t) - math.exp(-k))/(1 - math.exp(-k))

def cosine_scheduler(epoch, epochs_max, coef_min=1e-6, coef_max=1):
    return coef_min + (0.5)*(coef_max-coef_min)*(1 + math.cos(epoch*math.pi/epochs_max))