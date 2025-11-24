import time
from ga_runner_multi import run_ga_multi

params = dict(
    pop_size=200,
    ngen=200,
    cxpb=0.4,
    mutpb=0.7,
    seed=0,
    show_plots=True,
    show_anim=True,
    debug_interval=50
)

t0 = time.time()
run_ga_multi(**params)
t1 = time.time()

print("Tiempo por run:", t1 - t0, "segundos")
