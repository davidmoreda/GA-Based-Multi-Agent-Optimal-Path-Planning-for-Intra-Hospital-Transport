import time
import random
import math
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter

from ga_core import (
    prepare_environment,
    create_graph,
    build_route,
    mutate_macro_detour,
    mutate_segment,
    mutate_long_wait,
    mutate_wait,
    mutate_shift_start,
    mutate_conflict,
    detect_conflicts,
    evaluate,
    evaluate_clean_distance
)


# ============================================================
# MUTACIONES NUEVAS (usan tus operadores, no cambian penalizaciones)
# ============================================================

def massive_time_shift(route, max_shift=150):
    """Gran desincronización temporal: clave para eliminar penal_temporal."""
    k = random.randint(40, max_shift)
    return [route[0]] * k + route


def rebuild_route_smart(G, start, pick, drop, avoid_nodes=None):
    """
    Reconstruye ruta evitando nodos problemáticos (conflictivos).
    Usa tus propios shortest_path, pero con exclusión dinámica.
    """
    def clean_path(path):
        return path[:-1]

    try:
        if avoid_nodes:
            G2 = G.copy()
            G2.remove_nodes_from(avoid_nodes)
        else:
            G2 = G

        r1 = nx.shortest_path(G2, start, pick, weight="weight")
        r2 = nx.shortest_path(G2, pick, drop, weight="weight")
        r3 = nx.shortest_path(G2, drop, start, weight="weight")

        return clean_path(r1) + clean_path(r2) + r3
    except:
        return None


# ============================================================
# VECINDARIO PRO: multi-vecino + LNS + resolución de conflictos
# ============================================================

def sa_neighbor_pro(ind, env, G, starts, picks, drops):
    """
    Vecindario híbrido:
    - aplica varias mutaciones en paralelo
    - reconstrucciones inteligentes
    - desincronización temporal fuerte
    - usa tus mutadores locales + estratégicos
    """

    new_ind = [r[:] for r in ind]
    conflicts, bad_nodes = detect_conflicts(new_ind)

    for i in range(len(new_ind)):
        r = new_ind[i]

        # 1) MUTACIÓN GLOBAL SI HAY MUCHOS CONFLICTOS
        if conflicts and random.random() < 0.40:
            # conflicts es una lista de nodos conflictivos, no tripletas
            rebuilt = rebuild_route_smart(
                G,
                starts[i], picks[i], drops[i],
                avoid_nodes=conflicts
            )
            if rebuilt:
                new_ind[i] = rebuilt
                continue


        # 2) DESINCRONIZACIÓN INTENSA
        if random.random() < 0.20:
            r = massive_time_shift(r)

        # 3) MUTACIONES GRANDES (usas tus mutadores)
        if random.random() < 0.40:
            r = mutate_macro_detour(r, G, env)

        if random.random() < 0.35:
            r = mutate_segment(r, G)

        if random.random() < 0.40:
            r = mutate_long_wait(r, G)

        # 4) RESOLVER CONFLICTOS EN LOCAL
        if conflicts and random.random() < 0.50:
            r = mutate_conflict(r, G, conflicts)

        new_ind[i] = r

    return new_ind


# ============================================================
# SA PRO (Simulated Annealing con Restart Adaptativo)
# ============================================================

def run_sa(
    n_iter=15000,
    start_temp=300.0,
    end_temp=0.01,
    seed=42,
    show_plots=True,
    debug_interval=500
):
    random.seed(seed)
    np.random.seed(seed)

    # 1) ENTORNO BASE
    env, starts, picks, drops = prepare_environment()
    G = create_graph(env)

    # 2) RUTAS BASE
    base = []
    for k in range(len(starts)):
        r = build_route(G, starts[k], picks[k], drops[k])
        if r is None:
            raise RuntimeError(f"No hay ruta base para el agente {k}")
        base.append(r)

    def penalized_cost(ind):
        return evaluate(ind, picks, drops, base)[0]

    # 3) ESTADO INICIAL
    current = [r[:] for r in base]
    curr_cost = penalized_cost(current)

    best = [r[:] for r in current]
    best_cost = curr_cost
    best_clean = evaluate_clean_distance(best)

    clean_best_hist = []
    clean_curr_hist = []

    # 4) SCHEDULE
    alpha = (end_temp / start_temp) ** (1 / max(1, n_iter))

    start_t = time.time()
    restart_counter = 0
    last_improvement = 0

    # ============================================================
    # BUCLE PRINCIPAL SA PRO
    # ============================================================
    for it in range(1, n_iter + 1):
        T = start_temp * (alpha ** it)

        # Generar multi-vecinos y elegir el mejor candidato de ellos
        candidates = []
        for _ in range(4):  # 4 vecinos simultáneos
            c = sa_neighbor_pro(current, env, G, starts, picks, drops)
            candidates.append(c)

        costs = [penalized_cost(c) for c in candidates]
        idx = int(np.argmin(costs))
        candidate = candidates[idx]
        cand_cost = costs[idx]

        # ACCEPT
        if cand_cost < curr_cost:
            accept = True
        else:
            delta = cand_cost - curr_cost
            accept = random.random() < math.exp(-delta / T)

        if accept:
            current = candidate
            curr_cost = cand_cost

        # UPDATE BEST
        if curr_cost < best_cost:
            best = [r[:] for r in current]
            best_cost = curr_cost
            best_clean = evaluate_clean_distance(best)
            last_improvement = it

        clean_best_hist.append(best_clean)
        clean_curr_hist.append(evaluate_clean_distance(current))

        # ====================================================
        # RESTART ADAPTATIVO SI NO MEJORA EN X ITERACIONES
        # ====================================================
        if it - last_improvement > 1200 and T < start_temp * 0.5:
            restart_counter += 1
            current = [r[:] for r in base]  # vuelve a base
            curr_cost = penalized_cost(current)
            last_improvement = it
            T = start_temp  # reset temp

        # OUTPUT
        if debug_interval and (it % debug_interval == 0 or it == 1):
            conflicts, mind = detect_conflicts(best)
            print(
                f"[IT {it:5d}]  T={T:.4f}  BestPen={best_cost:.2f}  "
                f"CleanBest={best_clean:.2f}  CurrClean={clean_curr_hist[-1]:.2f}  "
                f"Conflicts={len(conflicts)}  MinDist={mind:.2f}  Restarts={restart_counter}"
            )

    end_t = time.time()

    print("\n===== SA PRO COMPLETADO =====")
    print(f"Best penalized = {best_cost:.2f}")
    print(f"Best clean dist = {best_clean:.2f}")
    print(f"Tiempo total = {end_t - start_t:.2f}s")
    print(f"Reinicios usados = {restart_counter}")

    return {
        "best_penalized": best_cost,
        "best_distance": best_clean,
        "routes": best,
        "clean_best": clean_best_hist,
        "clean_avg": clean_curr_hist,
        "time_sec": end_t - start_t,
        "restarts": restart_counter
    }


if __name__ == "__main__":
    run_sa(
        n_iter=15000,
        start_temp=300.0,
        end_temp=0.01,
        seed=42,
        show_plots=False,
        debug_interval=400
    )
