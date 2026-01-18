"""
FEA Thermal Visualization using PyVista ImageData (optimized for regular grids)
Note: DO not auto remove TODOs

Assumptions:
- Input voxels are REGULAR rectangular boxes with uniform spacing
- Grid example : 88×88×8 cells (89×89×9 nodes)
- X/Y spacing: 0.000398m, Z spacing: 0.000200m

"""

import csv
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import centralized rendering configuration and project path resolver
from environment import get_project_path

WINDOW_SIZE = (800, 600)
KELVIN_TO_CELSIUS = 273.15  # Temperature conversion constant

# Import PyVista after environment is configured by environment module
import pyvista as pv

pv.OFF_SCREEN = True


# ---------------- DATA LOADING ----------------


def load_all_temperatures(file):
    """Load sparse temperature data from node_temps.csv.

    IMPORTANT BUG DOCUMENTATION - DATA MISMATCH ISSUE:
    ==================================================
    According to the DES specification: "Each row has the temperature history of the node
    with node id corresponding to the row number." This means row N contains data for node ID N.

    However, due to a bug in the DES Rust simulation code (DES_thermal_simulation/src/main.rs:216-218),
    the pre-allocation of datastorernode is commented out:

        let mut datastorernode = Vec::with_capacity(nodeveclen);
        // for i in 0..nodeveclen{
        //   datastorernode.push(Vec::with_capacity(1000));
        // }

    This causes node_temps.csv to only contain rows for nodes that were activated during the
    thermal simulation. Nodes that were never heated (e.g., nodes from unprocessed elements or
    nodes that remained at ambient temperature) don't get rows written to the file.

    Example data mismatch:
    - nodefile.csv may have 3,382,607 nodes (all nodes from the mesh topology)
    - node_temps.csv may only have 2,043,088 rows (nodes that were heated during simulation)
    - Missing ~1.3M nodes remained at ambient temperature throughout

    Impact on visualization:
    - elementfile.csv references node IDs up to 3,382,606 (0-indexed)
    - node_temps.csv only has temperature data for node IDs 0 to 2,043,087
    - Accessing temps_matrix[node_id] with node_id >= 2,043,088 causes IndexError

    This function correctly handles the mismatch by:
    1. Using enumerate() so row index = node ID (per DES spec)
    2. Returning only the nodes that have data
    3. Letting the caller handle missing nodes by defaulting to ambient temperature

    Performance note: Using dict lookup is O(1) and doesn't impact performance.

    Args:
        file: Path to node_temps.csv

    Returns:
        dict[node_id] -> (times, temps): Temperature history for nodes with data (float32 arrays)
    """
    temp_data = {}

    with open(file) as f:
        for node_id, line in enumerate(f):
            try:
                # Use float32 for 50% memory reduction (sufficient precision for temps)
                values = np.fromstring(
                    line.strip().rstrip(","), sep=",", dtype=np.float32
                )
                if len(values) % 2 == 0 and len(values) > 0:
                    temp_data[node_id] = (values[::2], values[1::2])  # times, temps
            except ValueError as e:
                print(f"⚠️ Skipping node {node_id}: parse error - {e}")

    n_rows = max(temp_data.keys()) + 1 if temp_data else 0
    print(
        f"📊 Loaded temperature data: {len(temp_data)} active nodes from {n_rows} rows"
    )
    return temp_data


def analyze_grid_structure(nodes):
    """
    Analyze regular grid structure from node coordinates and create PyVista grid.

    Args:
        nodes: Node coordinate array (n_nodes, 3)

    Returns:
        tuple: (grid, dimensions_cells, coord_to_idx) where:
            - grid: PyVista ImageData object (reusable)
            - dimensions_cells: Tuple of cell dimensions for internal use
            - coord_to_idx: Dict mapping coordinates to indices
    """
    # Find unique coordinates in each dimension
    unique_x = np.sort(np.unique(nodes[:, 0]))
    unique_y = np.sort(np.unique(nodes[:, 1]))
    unique_z = np.sort(np.unique(nodes[:, 2]))

    # Grid dimensions - PyVista ImageData expects NUMBER OF POINTS, not cells
    # For n points, you get (n-1) cells
    dims_points = (len(unique_x), len(unique_y), len(unique_z))
    dims_cells = (len(unique_x) - 1, len(unique_y) - 1, len(unique_z) - 1)

    # Spacing (assuming uniform)
    spacing = (
        unique_x[1] - unique_x[0] if len(unique_x) > 1 else 1.0,
        unique_y[1] - unique_y[0] if len(unique_y) > 1 else 1.0,
        unique_z[1] - unique_z[0] if len(unique_z) > 1 else 1.0,
    )

    # Origin (minimum corner)
    origin = (unique_x[0], unique_y[0], unique_z[0])

    # Create lookup tables for coordinate to index mapping
    coord_to_idx = {
        "x": {coord: i for i, coord in enumerate(unique_x)},
        "y": {coord: i for i, coord in enumerate(unique_y)},
        "z": {coord: i for i, coord in enumerate(unique_z)},
    }

    print(
        f"📐 Grid structure: {dims_cells[0]}x{dims_cells[1]}x{dims_cells[2]} cells ({dims_points[0]}x{dims_points[1]}x{dims_points[2]} points)"
    )
    print(f"📏 Spacing: ({spacing[0]:.6f}, {spacing[1]:.6f}, {spacing[2]:.6f})")
    print(f"📍 Origin: ({origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f})")

    # Create PyVista grid once (will be reused for all frames)
    grid = pv.ImageData(dimensions=dims_points, spacing=spacing, origin=origin)

    return grid, dims_cells, coord_to_idx


def build_element_to_grid_mapping(nodes, elements_0idx, coord_to_idx):
    """
    Map each element to its (i,j,k) position in the regular grid.

    SIMPLIFIED: Vectorized coordinate-to-index mapping using np.searchsorted.

    Args:
        nodes: Node coordinates (n_nodes, 3)
        elements_0idx: Element connectivity (n_elements, 8), 0-indexed
        coord_to_idx: Dict mapping coordinates to indices

    Returns:
        element_grid_indices: (n_elements, 3) array of (i,j,k) indices
    """
    # Get first node of each element (corner node) - vectorized
    first_nodes = elements_0idx[:, 0]
    first_node_coords = nodes[first_nodes]  # (n_elements, 3)

    # Extract sorted unique coordinates
    unique_x = np.array(sorted(coord_to_idx["x"].keys()))
    unique_y = np.array(sorted(coord_to_idx["y"].keys()))
    unique_z = np.array(sorted(coord_to_idx["z"].keys()))

    # Vectorized searchsorted for all dimensions at once
    i_indices = np.searchsorted(unique_x, first_node_coords[:, 0])
    j_indices = np.searchsorted(unique_y, first_node_coords[:, 1])
    k_indices = np.searchsorted(unique_z, first_node_coords[:, 2])

    # Stack into (n_elements, 3) array
    element_grid_indices = np.column_stack([i_indices, j_indices, k_indices]).astype(
        np.int32
    )

    return element_grid_indices


# ---------------- IMAGEDATA FRAME PRECOMPUTATION ----------------


def precompute_all_frames_optimized(
    dimensions_cells,
    element_grid_indices,
    elements_0idx,
    temp_data_sparse,
    n_nodes,
    time_steps,
):
    """
    Preprocessing from raw data to per-frame grid.

    Args:
        dimensions_cells: Tuple of cell dimensions (x, y, z)
        element_grid_indices: (n_elements, 3) mapping of elements to (i,j,k)
        elements_0idx: (n_elements, 8) element connectivity (0-indexed)
        temp_data_sparse: dict[node_id] -> (times, temps)
        n_nodes: Total number of nodes
        time_steps: Array of timesteps

    Returns:
        per_frame_data: 4D array (n_steps, x, y, z) of temperatures
    """
    n_steps = len(time_steps)
    n_elements, nodes_per_elem = elements_0idx.shape
    time_steps_array = np.array(time_steps)

    # STEP 1: Extract node activation times (scalar per node, not array)
    node_activation_time = np.full(n_nodes, np.inf, dtype=np.float32)
    for node_id, (times, _) in temp_data_sparse.items():
        node_activation_time[node_id] = times[0]

    # STEP 2: Compute element activation times
    # Element activates when ALL nodes active = max of node activation times
    element_activation_times = np.max(
        node_activation_time[elements_0idx], axis=1
    )  # (n_elements,)

    # STEP 3: Identify elements ever active (activate before simulation ends)
    elements_ever_active = element_activation_times <= time_steps_array[-1]

    # STEP 4: Get unique used nodes (from ever-active elements) - boolean indexing
    used_node_ids = np.unique(elements_0idx[elements_ever_active])
    n_used = len(used_node_ids)
    print(
        f"📊 Sparse optimization: {n_used}/{n_nodes} nodes ({100 * n_used / n_nodes:.1f}%)"
    )

    # STEP 5: Sparse interpolation (only used nodes)
    print(f"⏱️  Sparse interpolation ({n_used} nodes) + grid population...")
    temps_matrix = np.full((n_nodes, n_steps), KELVIN_TO_CELSIUS, dtype=np.float32)
    for node_id in used_node_ids:
        # Handle nodes that may not have temperature data (due to Rust simulation bug)
        # Missing nodes default to ambient temperature (already initialized in temps_matrix)
        if node_id in temp_data_sparse:
            times, node_temps = temp_data_sparse[node_id]
            temps_matrix[node_id, :] = np.interp(time_steps_array, times, node_temps)
        # else: keep default ambient temperature KELVIN_TO_CELSIUS

    # STEP 6: Compute element active mask for each timestep
    # Element active at timestep t if t >= element_activation_time
    element_active_mask = (
        time_steps_array[None, :] >= element_activation_times[:, None]
    )  # (n_elements, n_steps) broadcasting

    # STEP 7: Populate grid (compute element temps + assign to grid)
    per_frame_data = np.full((n_steps,) + dimensions_cells, np.nan, dtype=np.float32)

    elem_indices, time_indices = np.where(element_active_mask)

    # Compute temps for active (element, timestep) pairs (handles empty arrays gracefully)
    n_active = len(elem_indices)
    active_node_indices = elements_0idx[elem_indices].flatten()
    time_indices_expanded = np.repeat(time_indices, nodes_per_elem)
    temps_active = temps_matrix[active_node_indices, time_indices_expanded]
    temps_reshaped = temps_active.reshape(n_active, nodes_per_elem)
    element_temps = np.mean(temps_reshaped, axis=1) - KELVIN_TO_CELSIUS

    # Extract grid coordinates and assign (empty arrays do nothing, no check needed)
    grid_coords = element_grid_indices[elem_indices]  # (n_active, 3)
    per_frame_data[
        time_indices, grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
    ] = element_temps

    return per_frame_data


def add_calibrated_floor(plotter, grid):
    """
    Adds a calibrated grid floor and scale text to the plotter.

    Args:
        plotter: PyVista Plotter object
        grid: PyVista ImageData object (source of bounds/spacing)
    """
    bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_min = bounds[4]
    z_height = bounds[5] - bounds[4]

    # Create floor plane at z_min with 20% padding
    floor_padding = 0.2
    floor_x_size = x_range * (1 + floor_padding)
    floor_y_size = y_range * (1 + floor_padding)
    floor_center_x = (bounds[0] + bounds[1]) / 2
    floor_center_y = (bounds[2] + bounds[3]) / 2

    # Offset floor slightly below the mesh to avoid z-fighting
    # Use a small fraction of the Z height or a fixed small value
    z_offset = max(z_height * 0.01, 1e-6)
    z_floor = z_min - z_offset

    # Use 10mm major grid spacing for visual reference
    major_interval = 0.010  # 10mm in meters

    # Ensure valid resolution (avoid 0)
    i_res = max(1, int(floor_x_size / major_interval))
    j_res = max(1, int(floor_y_size / major_interval))

    # Create floor plane
    floor = pv.Plane(
        center=(floor_center_x, floor_center_y, z_floor),
        direction=(0, 0, 1),
        i_size=floor_x_size,
        j_size=floor_y_size,
        i_resolution=i_res,
        j_resolution=j_res,
    )

    plotter.add_mesh(
        floor,
        color="lightgray",
        opacity=0.3,
        show_edges=True,
        edge_color="gray",
        line_width=1,
    )

    # Add calibration text
    grid_spacing_mm = grid.spacing[0] * 1000  # Convert to mm
    plotter.add_text(
        f"Calibrated grid floor (10mm spacing)\nGrid resolution: {grid_spacing_mm:.2f}mm",
        position=(0.02, 0.02),
        viewport=True,
        font_size=8,
        color="black",
    )


def plot_imagedata_fea(
    grid,
    temps_grid,
    time_step,
    save_path,
    global_clim,
):
    """
    Render a single frame using ImageData (optimized for regular grids).

    Args:
        grid: Pre-created PyVista ImageData grid (reused across frames)
        temps_grid: 3D numpy array (x, y, z) of temperatures (NaN for inactive)
        time_step: Current time value for display
        save_path: Where to save the rendered frame
        global_clim: Tuple of (min, max) for consistent color scale across all frames
    """
    # Flatten temperature data in Fortran order (PyVista expects X-fastest)
    temps_flat = temps_grid.flatten(order="F")

    # Early exit if no valid data
    if not np.any(~np.isnan(temps_flat)):
        print(f"⚠️ Skipped {time_step}s — All cells masked.")
        return 0

    # Update cached grid with new temperature data
    grid.cell_data["Temperature (°C)"] = temps_flat

    # Threshold to keep only non-NaN cells (removes inactive cells completely)
    grid_filtered = grid.threshold(value=(-np.inf, np.inf), scalars="Temperature (°C)")

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.background_color = "white"

    plotter.add_mesh(
        grid_filtered,
        scalars="Temperature (°C)",
        cmap="jet",
        scalar_bar_args={
            "title": "Temperature (°C)",
            "vertical": True,
            "title_font_size": 14,
            "label_font_size": 12,
            "color": "black",
            "fmt": "%.1f",
            "position_x": 0.88,
            "position_y": 0.15,
            "width": 0.08,
            "height": 0.7,
        },
        clim=global_clim,
    )

    # Add calibrated grid floor
    add_calibrated_floor(plotter, grid)

    # Add title
    plotter.add_text(
        f"t = {time_step:.1f}s", position="upper_edge", font_size=12, color="black"
    )

    # Add axes
    plotter.show_axes()

    # Set to isometric view
    plotter.view_isometric()

    # Save screenshot
    plotter.screenshot(save_path)
    plotter.close()


# ---------------- MAIN EXECUTION ----------------


def _worker_init():
    """
    Initialize worker process for parallel frame generation.
    Ensures environment configuration is applied in each worker.
    """
    # Re-import environment config to ensure worker setup matches main process
    # The module's configure_rendering() runs at import time

    # Ensure PyVista is configured for offscreen rendering
    import pyvista as pv

    pv.OFF_SCREEN = True


def _process_single_frame(args):
    """
    Worker function to process a single frame in parallel.
    Must be at module level for multiprocessing pickling.
    Environment is configured via _worker_init.

    Receives grid and 3D temperature array for a single frame.
    """
    (
        i,
        t,
        grid,
        temps_grid,
        frames_output_dir,
        global_clim,
    ) = args

    plot_imagedata_fea(
        grid,
        temps_grid,
        t,
        frames_output_dir / f"frame_{i:05d}.png",
        global_clim,
    )


def _resolve_project_dir(project_num: str, projects_root=None) -> Path:
    """Return the absolute project directory (respects legacy layouts automatically)."""
    if projects_root is None:
        return get_project_path(project_num)
    return Path(projects_root) / project_num


def generate_frames(project_num, projects_root=None, timestep=1.0, max_workers=None):
    """
    Generate thermal analysis frames for a given project using parallel processing.

    Args:
        project_num (str): Project number (e.g., "001")
        projects_root (Path | str | None): Base directory containing project folders.
            Defaults to the canonical path from environment.get_project_path().
        glass_temperature (float): Glass transition temperature in °C
        timestep (float): Time resolution in seconds between frames (default: 1.0)
        max_workers (int): Maximum number of parallel workers (default: CPU count - 1)

    Returns:
        str: Path to the generated frames directory
    """

    full_start = time.perf_counter()

    project_dir = _resolve_project_dir(project_num, projects_root)
    input_dir = project_dir / "frame_generator_input"  # FEA data from Rust code
    frames_output_dir = project_dir / "frames_filtered_active_only"  # Generated frames
    progress_file = project_dir / "progress.json"  # Progress tracking

    t0_load = time.perf_counter()

    def update_progress(current, total, message=""):
        progress_data = {
            "current": current,
            "total": total,
            "percentage": int((current / total) * 100) if total > 0 else 0,
            "message": message,
        }
        with open(progress_file, "w") as f:
            json.dump(progress_data, f)

    node_coords = (
        pd.read_csv(input_dir / "nodefile.csv", header=None).iloc[:, 1:].to_numpy()
    )
    elements = (
        pd.read_csv(input_dir / "elementfile.csv", header=None, dtype=int)
        .iloc[:, 1:]
        .to_numpy()
    )
    # node_temps is in a diagonal time temp format:
    # 0.0011390252100840215 , 472.91476202218854 , 0.002278050420168043 , 470.88385648350817 , 0.003417075630252064
    # 0.002278050420168043  , 471.10764071210446 , 0.003417075630252064 , 469.16008370672324 , 0.004556100840336165
    # 0.003417075630252064  , 471.0804633444287  , 0.004556100840336165 , 469.10829846981693 , 0.005695126050420186   | notice .003 x3 , 0.002 x2, 0.001 x1 diagonally
    # it is also the largest file used here

    temp_data_sparse = load_all_temperatures(input_dir / "node_temps.csv")
    n_nodes = len(
        node_coords
    )  # Get actual node count from nodefile, not from temp data
    t1_load = time.perf_counter()
    print(f"[Timing] Data loading: {t1_load - t0_load:.2f}s")
    print(
        f"📊 Total nodes from nodefile: {n_nodes}, nodes with temp data: {len(temp_data_sparse)}"
    )
    if len(temp_data_sparse) < n_nodes:
        print(
            f"⚠️  Missing temperature data for {n_nodes - len(temp_data_sparse)} nodes "
            f"(will default to ambient {KELVIN_TO_CELSIUS:.2f}K)"
        )

    # Create output directory for frames (clear old frames first)
    if frames_output_dir.exists():
        print(f"🗑️  Clearing old frames from {frames_output_dir}")
        shutil.rmtree(frames_output_dir)
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Does moving the element filtering to before interpolation save processing?
    # TODO: Switch from nodes to elements earlier?

    # --- TIME STEP CONTROL ---
    dt = timestep  # seconds per frame (configurable)
    t_start = 0.0

    # Auto-calculate t_end from last line of activation_times.csv
    t0_time = time.perf_counter()
    activation_file = input_dir / "activation_times.csv"
    with open(activation_file, "r") as f:
        # Read all lines and parse the last one with CSV
        lines = f.readlines()
        if not lines:
            raise ValueError(f"{activation_file} is empty or missing data")

        last_row = list(csv.reader([lines[-1]]))[0]
        last_time = float(last_row[0])
        t_end = np.ceil(last_time)  # Round up to nearest integer
        print(f"📊 Last activation time: {last_time:.2f}s → t_end set to {t_end:.0f}s")
    t1_time = time.perf_counter()
    print(f"[Timing] Time step calculation: {t1_time - t0_time:.2f}s")

    n_steps = int(round((t_end - t_start) / dt)) + 1
    time_steps = [t_start + i * dt for i in range(n_steps)]

    # Determine number of workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    print(f"🚀 Using {max_workers} parallel workers for frame generation")

    update_progress(0, n_steps, "Starting parallel frame generation...")

    t0_prep = time.perf_counter()

    # Geometric preprocessing (grid structure analysis + grid creation)
    grid, dimensions_cells, coord_to_idx = analyze_grid_structure(node_coords)
    elements_0idx = elements - 1
    element_grid_indices = build_element_to_grid_mapping(
        node_coords, elements_0idx, coord_to_idx
    )

    # Thermal preprocessing (all-in-one: activation, interpolation, grid population)
    per_frame_data = precompute_all_frames_optimized(
        dimensions_cells,
        element_grid_indices,
        elements_0idx,
        temp_data_sparse,
        n_nodes,
        time_steps,
    )

    t_prep_end = time.perf_counter()
    print(f"[Timing] Total precomputation: {t_prep_end - t0_prep:.2f}s")

    # Calculate global temperature range from raw data (vectorized, no frame generation)
    all_temps = np.concatenate([temps for _, temps in temp_data_sparse.values()])
    global_clim = (
        float(all_temps.min() - KELVIN_TO_CELSIUS),
        float(all_temps.max() - KELVIN_TO_CELSIUS),
    )
    print(
        f"📊 Global temperature range: {global_clim[0]:.1f}°C to {global_clim[1]:.1f}°C"
    )

    # Prepare arguments for parallel workers (grid + per-frame data)
    frame_args = []
    for i, t in enumerate(time_steps):
        frame_args.append(
            (
                i,
                t,
                grid,  # Shared grid object (same for all frames)
                per_frame_data[i],  # 3D temperature array for this frame
                frames_output_dir,
                global_clim,
            )
        )

    # Process frames in parallel using spawn context for clean worker processes
    t0_proc = time.perf_counter()
    completed_frames = 0
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=max_workers, initializer=_worker_init) as pool:
        for _ in pool.imap_unordered(_process_single_frame, frame_args):
            completed_frames += 1
            update_progress(
                completed_frames,
                n_steps,
                f"Generated {completed_frames}/{n_steps} frames",
            )

    t1_proc = time.perf_counter()
    print(f"[Timing] Frame generation (multiprocessing): {t1_proc - t0_proc:.2f}s")

    update_progress(n_steps, n_steps, "All frames generated successfully!")

    print(f"[Timing] Full process: {time.perf_counter() - full_start:.2f}s")
    return frames_output_dir


def generate_mask_frame(
    project_num, projects_root=None, tg=105.0, dHigh=15.0, dLow=45.0, time_s=-1.0
):
    """
    Generate a single hot/cold mask visualization frame.

    Args:
        project_num (str): Project number (e.g., "001")
        projects_root (Path | str | None): Base directory for project folders
        tg (float): Glass transition temperature in Celsius (default: 105°C)
        dHigh (float): Temperature above Tg for hot mask (default: 15°C)
        dLow (float): Temperature below Tg for cold mask (default: 45°C)
        time_s (float): Time point to visualize (-1 = last time, default: -1)

    Returns:
        str: Path to the generated mask image
    """
    print(
        f"🎭 Generating hot/cold mask frame (Tg={tg}°C, ΔHigh={dHigh}°C, ΔLow={dLow}°C)"
    )

    project_dir = _resolve_project_dir(project_num, projects_root)
    input_dir = project_dir / "frame_generator_input"
    output_dir = project_dir  # Output masks directly to project root

    # Load data
    node_coords = np.atleast_2d(
        np.loadtxt(
            input_dir / "nodefile.csv",
            delimiter=",",
            usecols=(1, 2, 3),
            dtype=np.float32,
        )
    )
    elements = np.atleast_2d(
        np.loadtxt(
            input_dir / "elementfile.csv",
            delimiter=",",
            usecols=tuple(range(1, 9)),
            dtype=np.int32,
        )
    )
    temp_data_sparse = load_all_temperatures(input_dir / "node_temps.csv")
    n_nodes = len(
        node_coords
    )  # Get actual node count from nodefile, not from temp data

    # Determine target time
    if time_s < 0:
        # Find maximum time from all node temperature data
        max_time = max(times[-1] for times, _ in temp_data_sparse.values())
        target_time = max_time
        print(f"📊 Using last time: {target_time:.2f}s")
    else:
        target_time = time_s
        print(f"📊 Using specified time: {target_time:.2f}s")

    # Compute element centroids
    elements_0idx = elements - 1

    # Interpolate node temperatures at target time
    node_temps_K = np.full(
        n_nodes, KELVIN_TO_CELSIUS, dtype=np.float32
    )  # Ambient temp (default for missing nodes)
    for node_id, (times, temps) in temp_data_sparse.items():
        # Only interpolate if node_id is within bounds (handles missing temp data gracefully)
        if node_id < n_nodes:
            node_temps_K[node_id] = np.interp(target_time, times, temps)

    # Compute element temperatures (average of 8 nodes, in Celsius)
    element_temps_C = (
        node_temps_K[elements_0idx].mean(axis=1) - KELVIN_TO_CELSIUS
    )  # (n_elements,)

    # Determine valid elements (those that have been activated by target_time)
    # Load activation times
    activation_times = np.loadtxt(
        input_dir / "activation_times.csv", delimiter=",", dtype=np.float32
    )
    element_activation_times = activation_times[:, 0]
    valid_mask = element_activation_times <= target_time

    # Apply hot/cold masks
    hot_mask = valid_mask & (element_temps_C > (tg + dHigh))
    cold_mask = valid_mask & (element_temps_C < (tg - dLow))
    base_mask = valid_mask & (~hot_mask) & (~cold_mask)

    print(
        f"🔴 Hot regions: {np.sum(hot_mask)} elements (>{tg + dHigh:.1f}°C, deformation risk)"
    )
    print(
        f"🔵 Cold regions: {np.sum(cold_mask)} elements (<{tg - dLow:.1f}°C, warping risk)"
    )
    print(f"⚪ Base regions: {np.sum(base_mask)} elements (normal range)")

    # Create PyVista visualization
    pv_grid, dims_cells, coord_to_idx = analyze_grid_structure(node_coords)
    element_grid_indices = build_element_to_grid_mapping(
        node_coords, elements_0idx, coord_to_idx
    )

    # Create mask color array (RGB values for each cell)
    n_cells = np.prod(dims_cells)
    mask_colors = np.full((n_cells, 3), np.nan, dtype=np.float32)

    # Flatten grid indices for assignment
    for elem_idx in range(len(elements_0idx)):
        i, j, k = element_grid_indices[elem_idx]
        flat_idx = np.ravel_multi_index((i, j, k), dims_cells, order="F")

        if hot_mask[elem_idx]:
            mask_colors[flat_idx] = [
                1.0,
                0.0,
                0.0,
            ]  # Red (PyVista rgb=True uses 0-1 range)
        elif cold_mask[elem_idx]:
            mask_colors[flat_idx] = [0.0, 0.0, 1.0]  # Blue
        elif base_mask[elem_idx]:
            mask_colors[flat_idx] = [0.784, 0.784, 0.784]  # Light grey (200/255)

    # Assign colors to grid
    pv_grid.cell_data["MaskColor_R"] = mask_colors[:, 0]
    pv_grid.cell_data["MaskColor_G"] = mask_colors[:, 1]
    pv_grid.cell_data["MaskColor_B"] = mask_colors[:, 2]

    # Threshold to remove NaN cells
    valid_cells_mask = ~np.isnan(mask_colors[:, 0])
    if np.sum(valid_cells_mask) == 0:
        print("⚠️ No valid cells to visualize")
        return None

    grid_filtered = pv_grid.threshold(value=(-np.inf, np.inf), scalars="MaskColor_R")

    # Render with PyVista
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.background_color = "white"

    # Combine RGB channels for visualization
    rgb_colors = np.column_stack(
        [
            grid_filtered.cell_data["MaskColor_R"],
            grid_filtered.cell_data["MaskColor_G"],
            grid_filtered.cell_data["MaskColor_B"],
        ]
    )

    plotter.add_mesh(
        grid_filtered,
        scalars=rgb_colors,
        rgb=True,
        show_scalar_bar=False,
    )

    # Add calibrated grid floor
    add_calibrated_floor(plotter, pv_grid)

    # Add legend
    plotter.add_text(
        f"Hot/Cold Mask Analysis (t = {target_time:.1f}s)\n"
        f"🔴 Hot (>{tg + dHigh:.1f}°C): {np.sum(hot_mask)} elements | "
        f"🔵 Cold (<{tg - dLow:.1f}°C): {np.sum(cold_mask)} elements | "
        f"⚪ Normal: {np.sum(base_mask)} elements",
        position=(0.02, 0.94),  # viewport coords keep text away from the top border
        viewport=True,
        font_size=10,
        color="black",
        name="mask_stats",
    )

    # Add axes
    plotter.show_axes()

    # Set to isometric view
    plotter.view_isometric()

    # Save screenshot
    output_path_png = output_dir / "hot_cold_mask.png"
    plotter.screenshot(str(output_path_png))
    plotter.close()

    print(f"✅ Mask frame PNG saved to: {output_path_png}")

    # Export 3MF file for 3D printing/viewing WITH COLORS
    output_path_3mf = output_dir / "hot_cold_mask.3mf"
    try:
        # PyVista ImageData doesn't support 3MF directly, so extract surface first
        surface = grid_filtered.extract_surface()

        # Convert to trimesh for 3MF export
        import trimesh

        # Get vertices and faces from PyVista PolyData
        vertices = surface.points
        # PyVista faces array: [n_points, id1, id2, ..., n_points, id1, id2, ...]
        # We need to extract faces properly, handling triangles and quads
        faces = surface.faces
        face_list = []
        i = 0
        while i < len(faces):
            n_points = faces[i]
            if n_points == 3:  # Triangle
                face_list.append(faces[i + 1 : i + 4])
            elif n_points == 4:  # Quad - split into 2 triangles
                # Triangle 1: points 0, 1, 2
                face_list.append(faces[i + 1 : i + 4])
                # Triangle 2: points 0, 2, 3
                face_list.append([faces[i + 1], faces[i + 3], faces[i + 4]])
            i += n_points + 1
        faces = np.array(face_list, dtype=np.int64)

        # Get face colors from PyVista surface (convert 0-1 range to 0-255 uint8)
        face_colors = None
        if "MaskColor_R" in surface.cell_data:
            face_colors = np.column_stack(
                [
                    surface.cell_data["MaskColor_R"] * 255,
                    surface.cell_data["MaskColor_G"] * 255,
                    surface.cell_data["MaskColor_B"] * 255,
                    np.full(
                        len(surface.cell_data["MaskColor_R"]), 255
                    ),  # Alpha channel
                ]
            ).astype(np.uint8)

        # Create trimesh object with colors
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)

        # Export to 3MF with colors
        mesh.export(str(output_path_3mf), file_type="3mf")
        print(f"✅ Mask 3MF file saved to: {output_path_3mf} (with colors)")
    except Exception as e:
        print(f"⚠️ Failed to export 3MF: {e}")

    return output_path_png


if __name__ == "__main__":
    # Original functionality when run as main script

    # Get project number from command line or use default
    project_num = input("Enter project number: ") or "001"

    t_main_start = time.perf_counter()
    generate_frames(project_num)
    t_main_end = time.perf_counter()
    print(f"[Timing] Main total run: {t_main_end - t_main_start:.2f}s")
