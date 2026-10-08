import os
import atexit
import numpy as np
from dataclasses import dataclass
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from multiprocessing.shared_memory import SharedMemory


@dataclass(frozen=True)
class PlotConfig:  ## Essentially a vessel for data.
    """
       The arg "frozen=True" tells @dataclass to generate __setattr__ and __delattr__ methods that
        immediately raise a FrozenInstanceError if anything tries to write to the instance after construction.
        Each worker gets its own copy of the config via pickling.
    """
    resolution:      int
    frames_per_year: int  ## 1200, usually: at 60 fps, this means 1 Jy passes in 20 seconds of "real time".
    ax_lim:          float  ## Axis-limit: 1.275*max_dist_from_Sun_to_Neptune.
    planet_names:    tuple[str, ...]
    planet_colors:   tuple[str, ...]
    scaling_labels:  dict[str, str]   ## choice of {"linear": "AU", "sqrt": "√AU", "log": "log₁₀(1+AU)"}
    scaling_options: frozenset[str]   ## choice of {"linear", "sqrt", "log"}


## These must match the module-level functions in your main module.
def _linear(r): return r
def _sqrt(r):   return np.sqrt(r)
def _log(r):    return np.log10(1.0 + r)

_SCALE_FNS = {"linear": _linear, "sqrt": _sqrt, "log": _log}
ASTRO_UNIT = 1.496e11  ## units: [m]

def _scale_xy(positions: np.ndarray, scale_type: str):
    """
       Scales positions into astronomical units (avg. orb. dist. b/n Sun & Earth, defined above as 1.496e+11 m).

       ARGUMENTS:
        positions : (..., 2) array in metres.

       RETURNS : (..., 2) array in scaled AU — same shape as input.
    """
    xy_au  = positions / ASTRO_UNIT                   ## metres → AU
    r      = np.hypot(xy_au[..., 0], xy_au[..., 1])   ## (...,)
    r_safe = np.where(r > 0, r, 1.0)                  ## avoid /0 at origin
    scale  = _SCALE_FNS[scale_type](r_safe) / r_safe  ## (...,)

    return xy_au * scale[..., np.newaxis]             ## (..., 2)


_g_cfg   = None
_g_shape = None
_g_arr_x = None
_g_shm_x = None

def _worker_init(cfg: PlotConfig, shm_x_name, arr_shape, scale_type: str):
    """
       Initializes a "_render_frame_worker(...)" in a computationally-efficient manner.
        Declares the necessary global variables, and then loads them in every time a new "_render_frame_worker(...)" is spawned.
        Helper method for "_render_frame_worker(...)", below.
    """
    global _g_cfg, _g_shape, _g_arr_x, _g_shm_x
    
    _g_cfg = cfg
    _g_shape = arr_shape
    
    _g_shm_x = SharedMemory(name=shm_x_name, create=False)
    _g_arr_x = np.ndarray(arr_shape, dtype=np.float64, buffer=_g_shm_x.buf)
    
    atexit.register(lambda: _g_shm_x.close())


def _render_frame_worker(task: tuple) -> None:
    """
       Renders a single frame from the simulation to a .jpeg file on the local drive.
        This is a "task function" to be performed by a P-core (one individual so-called "CPU" on the SoC)
            during a simulation. Multiple (12) P-cores will be doing likewise at the same time, thus our
            operation is PARALLELIZED!

       ARGUMENTS:
        task (frame_idx (int), out_path (str)), where:
            frame_idx : row-index into the shared-memory position array;
            out_path  : path to destination-file for the .jpeg output.

       RETURNS: VOID
    """
    frame_idx, scale_type, out_dir, elapsed_jy = task  ## unpacking tuple from argument.

    cfg = _g_cfg
    arr = _g_arr_x

    ##### APPROPRIATELY SCALE THE POSITIONS #####
    curr_m  = arr[frame_idx]     ## (n_planets, 2)
    trail_m = arr[:frame_idx+1]  ## (history, n_planets, 2)

    ## The function _scale_xy(...) handles any shape like (..., 2), so we can vectorize over the planets.
    curr_s = _scale_xy(curr_m, scale_type)  ## (Np,2)
    trail_s = _scale_xy(trail_m, scale_type)  ## (history,Np,2)

    ##### BUILD THE OUTPUT FIGURE #####
    fig = Figure(figsize=(8, 8), dpi=cfg.resolution)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ## White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    ## Fixed frame — mirrors the 1.275 * r_max logic from plot()
    ax.set_xlim(-cfg.ax_lim, cfg.ax_lim)
    ax.set_ylim(-cfg.ax_lim, cfg.ax_lim)

    ## Black box spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1)

    ## Tick styling
    ax.tick_params(axis="both", colors="black", labelsize=11, direction="out")

    ## Grid
    ax.grid(True, color="lightgray", linewidth=0.5, linestyle="-", zorder=0)

    ## Axis labels + title
    scale_label = cfg.scaling_labels[scale_type]   # e.g. "log_{10}(1 + AU)"
    ax.set_xlabel(f"{scale_label}", color="black", fontsize=13)
    ax.set_ylabel(f"{scale_label}", color="black", fontsize=13)
    ax.set_title("The Solar System", color="black", fontsize=18)

    ## Timestamp box (lower-left), matching the image
    ax.text(
        0.02, 0.02, f"∆t = {elapsed_jy:.3f} Jy",
        transform=ax.transAxes,
        fontsize=11, color="black",
        verticalalignment="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.3"),
    )

    ## Sun at the origin (heliocentric reference-frame):
    ##  Param "zorder" determines what gets plotted on top of what.
    ax.scatter(0.0, 0.0, marker="*", s=400, c="goldenrod", edgecolor="k", zorder=5)

    ## Dashed-line trails and current-dot for position of the planets:
    for p, (name, color) in enumerate(zip(cfg.planet_names, cfg.planet_colors)):
        ax.plot(
            trail_s[:, p, 0], trail_s[:, p, 1],
            color=color, linestyle="--", linewidth=2, alpha=1.0
        )
        ax.scatter(
            curr_s[p, 0], curr_s[p, 1],
            color=color, s=60, edgecolor="k", zorder=4
        )
        ax.annotate(
            name,
            xy=(curr_s[p, 0], curr_s[p, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            color="black",
        )

    ##### SAVE AND RELEASE THE FIGURE FROM MEMORY #####
    canvas.draw()

    # Construct the full file path
    filename = f"frame_{frame_idx:06d}.jpg"
    out_path = os.path.join(out_dir, filename)

    fig.savefig(
        out_path,
        format="jpeg",
        dpi=cfg.resolution,
        pil_kwargs={"quality": 85}
    )

    fig.clf()
    del fig
