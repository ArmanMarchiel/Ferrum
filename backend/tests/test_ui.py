"""Headless browser tests for the viewer.

These exist because the two worst bugs in this project so far were only
visible in a browser: a synchronous pipeline that wedged the server so no
upload request ever arrived, and a camera that framed the world origin while
the model sat somewhere else entirely. Neither could be caught by testing the
Python alone.

Skipped automatically when playwright or its browser is unavailable.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent          # backend/
pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

# How far the drawn content may sit from the canvas centre, in pixels.
# Measured from the rendered image, not from projection maths: an earlier
# version of this test projected a single world point, reported 0 px, and
# passed while the grid was visibly stuck in a corner of the screen.
CENTRE_TOL = 60.0
SIDEBAR_PX = 330
# The viewport's own background, as set on the three.js scene. It was a dark
# theme once; the light rebuild left this constant behind, so every "is the
# scene centred" measurement was being taken against a colour that is nowhere
# on the page.
BG = (0xF4, 0xF5, 0xF7)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def origin(tmp_path_factory):
    """A live server, and the URL it is on. `/` is the filesystem home."""
    port = _free_port()
    # Point the project store at a throwaway directory: these tests save
    # projects, and they must not write into the developer's real ~/.cad2shell.
    env = dict(os.environ,
               CAD2SHELL_DATA=str(tmp_path_factory.mktemp("fsdata")))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT / "src", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.skip("server did not start")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def server(origin):
    """The CAD editor's URL.

    `/` is the filesystem now, so every test that drives the viewer wants
    `/editor`. Keeping that behind this fixture means the viewer tests say
    `page.goto(server)` exactly as they did when the editor was the root.
    """
    return origin + "/editor"


@pytest.fixture(scope="module")
def browser():
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch()
            except Exception as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            yield b
            b.close()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"playwright unavailable: {exc}")


@pytest.fixture(scope="module")
def part_stl(tmp_path_factory):
    import trimesh
    p = tmp_path_factory.mktemp("ui") / "part.stl"
    trimesh.creation.box(extents=[40, 30, 25]).export(p)
    return str(p)


def _offset(page, tmp_path=None):
    """Pixels between the centre of the DRAWN content and the canvas centre.

    Screenshots the page, hides the overlays first so only the 3-D scene is
    measured, and finds the bounding box of every pixel that is not the scene
    background. This is what the user actually sees.
    """
    import numpy as np
    from PIL import Image
    import io as _io

    page.evaluate(
        "()=>{for(const s of ['#hud','#legend','#status'])"
        "{const e=document.querySelector(s); if(e) e.style.display='none';}}"
    )
    page.wait_for_timeout(350)
    raw = page.screenshot()
    page.evaluate(
        "()=>{for(const s of ['#hud','#legend','#status'])"
        "{const e=document.querySelector(s); if(e) e.style.display='';}}"
    )

    im = np.array(Image.open(_io.BytesIO(raw)).convert("RGB")).astype(int)
    H, W, _ = im.shape
    scale = W / page.viewport_size["width"]
    # Crop to the canvas by its real position, so the header above it and the
    # sidebar beside it are not mistaken for drawn content.
    box = page.eval_on_selector(
        "canvas",
        "e => { const r = e.getBoundingClientRect();"
        "  return {top: r.top, left: r.left, w: r.width, h: r.height}; }")
    top = int(box["top"] * scale)
    left = int(box["left"] * scale)
    canvas = im[top:top + int(box["h"] * scale), left:left + int(box["w"] * scale)]
    ch, cw, _ = canvas.shape
    lit = np.abs(canvas - np.array(BG)).sum(axis=2) > 10
    ys, xs = np.nonzero(lit)
    if len(xs) == 0:
        pytest.fail("nothing was drawn in the 3-D view")
    dx = (xs.min() + xs.max()) / 2 - cw / 2
    dy = (ys.min() + ys.max()) / 2 - ch / 2
    clipped = bool(xs.min() <= 1 or xs.max() >= cw - 2 or
                   ys.min() <= 1 or ys.max() >= ch - 2)
    return abs(dx), abs(dy), clipped


def _wait_done(page, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = page.inner_text("#status")
        if s.startswith("Done"):
            return s
        if "rror" in s:
            pytest.fail(f"pipeline reported an error: {s}")
        page.wait_for_timeout(250)
    pytest.fail("mould did not finish in time")


def test_page_loads_without_console_errors(server, browser):
    errors = []
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server, wait_until="load")
    page.wait_for_timeout(2500)
    assert not errors, f"page errors: {errors}"
    assert page.query_selector("canvas") is not None
    page.close()


def test_demo_parts_are_not_in_the_ui(server, browser):
    page = browser.new_page()
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    assert page.query_selector("#demos") is None
    page.close()


@pytest.mark.parametrize("size", [(1280, 720), (1600, 900), (2560, 1400)])
def test_empty_scene_is_centred(server, browser, size):
    """The world origin must sit at the centre of the canvas before any upload.

    Regression: `fit()` needs a model to measure, so with no model the camera
    kept a hardcoded position aimed at an unset orbit target and the axes
    marker ended up in a corner of the viewport.
    """
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(2500)

    # Ask where the origin actually projects rather than inferring it from
    # pixels. The drawn ground plane is wider than it is deep and runs off the
    # near edge by design, so its lit-pixel bounding box is not centred even
    # when the camera is aimed dead on -- which is what this test is about.
    probe = page.evaluate("window.__probe()")
    dx = abs(probe["originPx"]["x"] - probe["centrePx"]["x"])
    dy = abs(probe["originPx"]["y"] - probe["centrePx"]["y"])
    assert dx <= CENTRE_TOL and dy <= CENTRE_TOL, \
        f"the world origin is off centre by ({dx:.0f},{dy:.0f})px"
    page.close()


def test_part_renders_before_the_mould_is_built(server, browser, part_stl):
    """Phase 1: the uploaded part must appear almost immediately.

    The mould takes seconds to minutes; the user should not watch an empty
    grid for that whole time.
    """
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)

    t0 = time.time()
    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    part_time = time.time() - t0
    assert part_time < 15.0, f"part took {part_time:.1f}s to render"

    # the mould is still building at this point
    status = page.inner_text("#status")
    assert "Building" in status or "Part loaded" in status or "Done" in status
    page.close()


def test_view_stays_centred_through_the_whole_workflow(server, browser, part_stl):
    """Centred when empty, after the part loads, and after the mould arrives.

    Regression: `fit()` centred the bounding *box* while the grid and axes
    stayed pinned to the world origin, so once the gating tree (which extends
    well above and below the casting) was added, the marker drifted off centre.
    """
    page = browser.new_page(viewport={"width": 1600, "height": 900})

    def check(label):
        # What this test is really about is the orbit target staying under the
        # content: the grid and axes are pinned to it, so if it drifts they
        # drift with it. Reading the projected origin says that directly,
        # where a lit-pixel bounding box also picks up the ground plane
        # running off the near edge, which it does at every stage by design.
        probe = page.evaluate("window.__probe()")
        dx = abs(probe["originPx"]["x"] - probe["centrePx"]["x"])
        dy = abs(probe["originPx"]["y"] - probe["centrePx"]["y"])
        assert dx <= CENTRE_TOL and dy <= CENTRE_TOL, \
            f"{label}: off centre by ({dx:.0f},{dy:.0f})px"

    page.goto(server, wait_until="load")
    page.wait_for_timeout(2500)
    check("empty scene")

    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    page.wait_for_timeout(600)
    check("after the part loaded")

    _wait_done(page)
    page.wait_for_timeout(800)
    check("after the mould was built")

    page.click("#c-shell")
    page.wait_for_timeout(300)
    page.click("#fit")
    page.wait_for_timeout(800)
    check("after toggle + fit")
    page.close()


def test_all_three_visuals_load_and_toggle(server, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    page.set_input_files("#file", part_stl)
    _wait_done(page)

    layers = page.evaluate("()=>window.__probe().layers")
    assert layers == {"part": True, "tree": True, "shell": True}, layers

    for key in ("part", "shell", "tree"):
        page.click(f"#c-{key}")
        page.wait_for_timeout(200)
        assert page.evaluate("()=>window.__probe().layers")[key] is False, f"{key} did not hide"
        page.click(f"#c-{key}")
        page.wait_for_timeout(200)
        assert page.evaluate("()=>window.__probe().layers")[key] is True, f"{key} did not show"
    page.close()


def test_results_table_is_populated(server, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    page.set_input_files("#file", part_stl)
    _wait_done(page)
    rows = page.eval_on_selector_all("#stats tr", "e=>e.length")
    assert rows >= 10, f"results table has only {rows} rows"
    text = page.inner_text("#stats")
    for label in ("shell wall", "watertight", "yield"):
        assert label in text, f"results table is missing {label!r}"
    page.close()


@pytest.mark.parametrize("dpr", [1, 2, 3])
def test_canvas_is_sized_in_css_pixels_at_every_dpr(server, browser, dpr):
    """The canvas must never be larger than its container.

    Regression, and the single hardest bug in this viewer: `setSize(w, h, false)`
    tells three.js not to write the canvas's CSS size. Paired with a pixel ratio
    of 2 on any Retina display, the canvas got a 2x backing buffer with nothing
    constraining its layout size, so it rendered at double the container and the
    scene spilled off the bottom-right -- indistinguishable from a camera that
    is not centred. It reproduced only at dpr >= 2, which is why every dpr=1
    measurement kept insisting the view was fine.
    """
    page = browser.new_page(viewport={"width": 1400, "height": 800},
                            device_scale_factor=dpr)
    page.goto(server, wait_until="load")
    page.wait_for_timeout(2500)

    m = page.evaluate("""()=>{
        const c = document.querySelector('canvas');
        const v = document.querySelector('#view');
        const r = c.getBoundingClientRect();
        return {cssW: r.width, cssH: r.height,
                viewW: v.clientWidth, viewH: v.clientHeight};
    }""")
    assert abs(m["cssW"] - m["viewW"]) <= 2, \
        f"canvas is {m['cssW']}px wide inside a {m['viewW']}px container (dpr={dpr})"
    assert abs(m["cssH"] - m["viewH"]) <= 2, \
        f"canvas is {m['cssH']}px tall inside a {m['viewH']}px container (dpr={dpr})"

    dx, dy, clipped = _offset(page)
    assert not clipped, f"content runs off the viewport at dpr={dpr}"
    assert dx <= CENTRE_TOL and dy <= CENTRE_TOL, \
        f"off centre by ({dx:.0f},{dy:.0f})px at dpr={dpr}"
    page.close()


def test_shells_appear_only_on_generate(server, browser, part_stl):
    """Dropping a file shows the part alone; the mould waits for the button.

    Shell generation takes seconds to minutes and its parameters are meant to
    be set first, so building automatically on drop would throw that work away.
    """
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)

    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    page.wait_for_timeout(1200)

    layers = page.evaluate("()=>window.__probe().layers")
    assert layers["part"] is True
    assert layers["shell"] is False, "the shell was built without pressing Generate"
    assert layers["fired"] is False, "the fired shell was built without pressing Generate"

    page.click("#go")
    _wait_done(page)
    page.wait_for_timeout(600)
    after = page.evaluate("()=>window.__probe().layers")
    assert after["shell"] is True and after["fired"] is True, \
        f"Generate did not produce both shells: {after}"
    page.close()


def test_zoom_buttons_change_the_camera_distance(server, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    page.wait_for_timeout(600)

    start = page.evaluate("()=>window.__probe().dist")
    page.click("#zoom-in")
    page.wait_for_timeout(300)
    closer = page.evaluate("()=>window.__probe().dist")
    assert closer < start, f"zoom in did not move closer ({start} -> {closer})"

    page.click("#zoom-out")
    page.click("#zoom-out")
    page.wait_for_timeout(300)
    further = page.evaluate("()=>window.__probe().dist")
    assert further > closer, f"zoom out did not move away ({closer} -> {further})"
    page.close()


@pytest.mark.parametrize("axis,index", [("x", 0), ("y", 1), ("z", 2)])
def test_rotate_buttons_turn_the_model(server, browser, part_stl, axis, index):
    """Each button turns the model 90 degrees about that world axis."""
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    page.wait_for_timeout(600)

    before = page.evaluate("()=>window.__probe().rotation")
    page.click(f'[data-rot="{axis}"]')
    page.wait_for_timeout(500)
    after = page.evaluate("()=>window.__probe().rotation")
    assert after != before, f"rotate {axis} did nothing"
    assert abs(after[index]) == 90, f"expected 90 degrees on {axis}, got {after}"

    page.click("#reset")
    page.wait_for_timeout(500)
    assert page.evaluate("()=>window.__probe().rotation") == [0, 0, 0], \
        "reset did not clear the rotation"
    page.close()


def test_new_upload_clears_the_previous_mould(server, browser, part_stl, tmp_path):
    """Uploading again must not leave the old shells on screen.

    Two bugs did this: `upload()` cleared part/shell/tree but not `fired`,
    which was added later; and re-picking the same file fires no change event,
    so the handler never ran at all and the previous mould simply stayed.
    """
    import trimesh
    other = tmp_path / "other.stl"
    trimesh.creation.box(extents=[30, 20, 15]).export(other)

    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)

    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    page.click("#go")
    _wait_done(page)
    page.wait_for_timeout(600)
    built = page.evaluate("()=>window.__probe().layers")
    assert built["shell"] and built["fired"], f"nothing was built: {built}"

    page.set_input_files("#file", str(other))
    page.wait_for_timeout(2500)
    after = page.evaluate("()=>window.__probe().layers")
    assert after["part"] is True
    assert after["shell"] is False, "the previous shell survived a new upload"
    assert after["fired"] is False, "the previous fired shell survived a new upload"
    page.close()


def test_layer_controls_live_only_in_the_legend(server, browser, part_stl):
    """One control per layer, in the legend. The sidebar copy was redundant.

    The sidebar listed the same four layers as the legend, with descriptions
    that restated the label ("Original CAD -- the part you uploaded").
    """
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)

    assert page.query_selector(".vis") is None, "the sidebar visuals section is back"
    boxes = page.eval_on_selector_all("#legend input[type=checkbox]", "e=>e.length")
    assert boxes == 4, f"expected 4 layer checkboxes in the legend, found {boxes}"
    # exactly one control per layer, nowhere else
    for key in ("part", "shell", "fired", "tree"):
        n = page.eval_on_selector_all(f"#c-{key}", "e=>e.length")
        assert n == 1, f"{key} has {n} controls"
        assert page.eval_on_selector(f"#c-{key}", "e=>!!e.closest('#legend')"), \
            f"{key}'s control is not in the legend"
    assert page.query_selector("#legend #opacity") is not None, \
        "the opacity slider did not move into the legend"
    page.close()


def test_every_layer_toggles_from_the_legend(server, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=30000)
    page.click("#go")
    _wait_done(page)
    page.wait_for_timeout(600)

    for key in ("part", "shell", "fired", "tree"):
        before = page.evaluate("()=>window.__probe().layers")[key]
        page.click(f"#c-{key}")
        page.wait_for_timeout(250)
        after = page.evaluate("()=>window.__probe().layers")[key]
        assert after != before, f"{key} did not toggle"
        page.click(f"#c-{key}")
        page.wait_for_timeout(250)
        assert page.evaluate("()=>window.__probe().layers")[key] == before, \
            f"{key} did not restore"
    page.close()


# -- multi-cavity preview --------------------------------------------------
#
# The cluster has to be visible as soon as the quantity is set, before any
# mould is built. Shipping this without a test was what let an earlier version
# render one part while the server was quite correctly building four -- the
# controls all responded, so nothing looked broken.

_PART_MESH_COUNT = """()=>{
    const g = window.__layers.part;
    if(!g) return 0;
    let n = 0; g.traverse(o=>{ if(o.isMesh) n++; });
    return n;
}"""

# Each instance's own world position, derived from the shared geometry's bbox
# centre pushed through that mesh's world matrix. THREE is module-scoped in the
# page, so the matrix is applied by hand rather than with a Vector3.
_PART_CENTRES = """()=>{
    const g = window.__layers.part; const out = [];
    g.traverse(o=>{ if(o.isMesh){
        o.updateMatrixWorld(true);
        const bb = o.geometry.boundingBox;
        const cx = (bb.min.x+bb.max.x)/2, cy = (bb.min.y+bb.max.y)/2;
        const m = o.matrixWorld.elements;
        out.push([m[0]*cx + m[4]*cy + m[12], m[1]*cx + m[5]*cy + m[13]]);
    }});
    return out.sort((a,b)=>(a[1]-b[1])||(a[0]-b[0]));
}"""


def _load_part(page, server, part_stl):
    page.goto(server, wait_until="load")
    page.wait_for_timeout(2000)
    page.set_input_files("#file", part_stl)
    page.wait_for_timeout(6000)


def _set_quantity(page, n, arrangement=None):
    page.fill("#qty", str(n))
    page.dispatch_event("#qty", "input")
    if arrangement:
        page.select_option("#layout", arrangement)
    page.wait_for_timeout(600)


@pytest.mark.parametrize("n", [1, 2, 4, 6, 9])
def test_quantity_shows_that_many_parts_before_generating(server, browser,
                                                          part_stl, n):
    """Setting the quantity lays the cluster out immediately.

    No Generate press anywhere in this test: the parts must appear as soon as
    the number changes, which is the whole point of putting the control in the
    part section rather than the process section.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    _load_part(page, server, part_stl)
    _set_quantity(page, n)
    assert page.evaluate(_PART_MESH_COUNT) == n
    page.close()


def test_quantity_control_lives_in_the_part_section(server, browser, part_stl):
    """Quantity describes what is being moulded, not how it is processed."""
    page = browser.new_page()
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    section = page.evaluate("""()=>{
        const q = document.querySelector('#qty');
        const sec = q.closest('.sec');
        return sec.querySelector('.hdr').textContent.trim();
    }""")
    assert "part" in section.lower()
    page.close()


def test_preview_positions_match_what_the_server_builds(server, browser, part_stl):
    """A preview that disagrees with the result would be worse than none.

    The placement maths is written twice -- once in layout.py for the build and
    once in the page for the preview -- so this pins the two together.
    """
    import numpy as np
    from cad2shell import Config, ingest, layout
    import trimesh

    page = browser.new_page()
    _load_part(page, server, part_stl)
    base = ingest.from_mesh(trimesh.load(part_stl, force="mesh"))

    for n in (2, 4, 6):
        _set_quantity(page, n)
        shown = np.array(page.evaluate(_PART_CENTRES))
        cfg = Config(quantity=n, shell_thickness=6.0, voxel_pitch=1.5)
        expected = np.array(sorted(
            ([i.centre[0], i.centre[1]] for i in layout.build(base, cfg).instances),
            key=lambda p: (p[1], p[0])))
        assert np.abs(shown - expected).max() < 1e-6, \
            f"preview and server disagree on the {n}-up layout"
    page.close()


def test_changing_quantity_clears_a_stale_mould(server, browser, part_stl):
    """A shell built for one part must not stay on screen next to four.

    Leaving it there reads as the new build having failed, and the shell shown
    is genuinely the wrong mould for what is on the plane.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    page.click("#go")
    page.wait_for_function(
        "()=>window.__layers.shell !== null", timeout=180_000)
    assert page.evaluate("()=>window.__layers.shell !== null")

    _set_quantity(page, 4)
    assert page.evaluate("()=>window.__layers.shell === null"), \
        "the single-part shell survived a change of quantity"
    assert page.evaluate(_PART_MESH_COUNT) == 4
    page.close()


# -- arranging parts by hand -----------------------------------------------

def test_clicking_a_part_selects_it(server, browser, part_stl):
    """Selection is the entry point for every other manipulation."""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    _load_part(page, server, part_stl)
    _set_quantity(page, 4)
    page.evaluate("()=>window.__select(2)")
    page.wait_for_timeout(400)
    assert page.evaluate("()=>window.__selected()") == 2
    assert page.is_visible("#selinfo")
    page.close()


def test_moving_a_part_updates_the_arrangement(server, browser, part_stl):
    """What the gizmo edits is the array that gets POSTed, not just the mesh.

    Reading transforms back off the meshes at submit time was the alternative,
    and it is how a preview and a build drift apart -- one array is the truth.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 2)
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:200, y:0, z:0, rot:90}])""")
    page.wait_for_timeout(500)
    pl = page.evaluate("()=>window.__placements()")
    assert len(pl) == 2
    assert pl[1]["x"] == 200 and pl[1]["rot"] == 90
    page.close()


def test_overlapping_parts_are_flagged_and_block_the_build(server, browser, part_stl):
    """Parts may be parked anywhere while arranging, but not built that way.

    Two castings closer than two shell walls have no ceramic between them --
    their shells merge and the parts come out joined. The viewer says so and
    disables Generate rather than letting a doomed build run.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 2)
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:3, y:0, z:0, rot:0}])""")
    page.wait_for_timeout(500)
    assert page.is_visible("#clash")
    assert page.is_disabled("#go")

    # move them apart again and the build unblocks
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:200, y:0, z:0, rot:0}])""")
    page.wait_for_timeout(500)
    assert not page.is_visible("#clash")
    assert not page.is_disabled("#go")
    page.close()


@pytest.mark.parametrize("per_tier,tiers", [(4, 1), (4, 2), (4, 3), (2, 2)])
def test_stacking_tiers_multiplies_the_parts(server, browser, part_stl,
                                             per_tier, tiers):
    """Quantity is parts PER TIER; stacking repeats the arranged tier upward.

    The count must not compound: an earlier version read the quantity field
    back as a total after tiering had already written it, so every edit squared
    the number of parts.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    page.fill("#qty", str(per_tier))
    page.dispatch_event("#qty", "input")
    page.fill("#tiers", str(tiers))
    page.dispatch_event("#tiers", "input")
    page.wait_for_timeout(700)

    assert page.evaluate(_PART_MESH_COUNT) == per_tier * tiers
    levels = page.evaluate(
        "()=>[...new Set(window.__placements().map(p=>Math.round(p.z)))]")
    assert len(levels) == tiers
    page.close()


def test_reset_layout_discards_a_hand_arrangement(server, browser, part_stl):
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 4)
    auto = page.evaluate("()=>window.__placements().map(p=>p.x)")

    page.evaluate("""()=>window.__setPlacements([
        {x:0,y:0,z:0,rot:0},{x:300,y:0,z:0,rot:45},
        {x:0,y:300,z:0,rot:90},{x:300,y:300,z:0,rot:135}])""")
    page.wait_for_timeout(400)
    assert page.evaluate("()=>window.__placements()[1].rot") == 45

    page.click("#resetlayout")
    page.wait_for_timeout(600)
    assert page.evaluate("()=>window.__placements().map(p=>p.x)") == auto
    page.close()


def test_a_hand_arrangement_is_what_gets_built(server, browser, part_stl):
    """The mould must wrap the parts where the user put them.

    This measures the shell that comes back, not the request that went out: a
    correct POST followed by a mould built from the nominal layout would look
    identical from the client side and be just as wrong.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 2)
    # push the two parts far apart; the shell has to span them
    page.evaluate("""()=>window.__setPlacements([
        {x:-90, y:0, z:0, rot:0}, {x:90, y:0, z:0, rot:0}])""")
    page.wait_for_timeout(500)
    assert not page.is_disabled("#go")

    page.click("#go")
    page.wait_for_function("()=>window.__layers.shell !== null", timeout=300_000)
    page.wait_for_timeout(1000)
    width = page.evaluate("""()=>{
        const m = window.__layers.shell;
        m.geometry.computeBoundingBox();
        const b = m.geometry.boundingBox;
        return b.max.x - b.min.x;
    }""")
    # 180 mm between centres plus the parts themselves and two shell walls
    assert width > 200, f"shell is only {width:.0f} mm wide; it did not span the arrangement"
    page.close()


# -- the gizmo sits on the part it manipulates -----------------------------

@pytest.fixture(scope="module")
def offcentre_stl(tmp_path_factory):
    """A part modelled well away from its own origin.

    Real CAD exports routinely arrive like this, and it is the case that
    exposes a gizmo placed at the raw placement offset: the handles end up
    adrift by exactly the part's own translation.
    """
    import trimesh
    m = trimesh.creation.box(extents=[80, 40, 30])
    m.apply_translation([120, 60, 15])
    p = tmp_path_factory.mktemp("offcentre") / "off.stl"
    m.export(p)
    return str(p)


@pytest.mark.parametrize("stl_fixture", ["part_stl", "offcentre_stl"])
def test_gizmo_sits_on_the_selected_part(server, browser, request, stl_fixture):
    """The handles must be on the part, not floating somewhere near it.

    Both readings are taken in world space. Comparing a root-local gizmo
    position against a world mesh centre reports a false failure of exactly the
    view offset, which is a trap worth naming: `fit()` translates the whole
    model group, so anything measured outside it is in a different frame.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    _load_part(page, server, request.getfixturevalue(stl_fixture))
    _set_quantity(page, 4)

    for i in range(4):
        page.evaluate(f"()=>window.__select({i})")
        page.wait_for_timeout(250)
        gizmo = page.evaluate("()=>window.__gizmoPos()")
        centre = page.evaluate(f"()=>window.__meshCentre({i})")
        worst = max(abs(a - b) for a, b in zip(gizmo, centre))
        assert worst < 0.5, (
            f"gizmo is {worst:.1f} mm from the centre of part {i}: "
            f"{gizmo} vs {centre}")
    page.close()


# -- the X/Y/Z buttons follow the selection --------------------------------

_ROTATIONS = "()=>window.__placements().map(p=>Math.round(((p.rot%360)+360)%360))"


def test_rotate_button_turns_only_the_selected_part(server, browser, part_stl):
    """With a part selected the buttons edit that part, not the whole scene.

    Turning the entire view when one casting is selected was the confusing
    behaviour: the click appeared to work, but every part moved together and
    the arrangement was unchanged.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 4)
    assert page.evaluate(_ROTATIONS) == [0, 0, 0, 0]

    page.evaluate("()=>window.__select(1)")
    page.wait_for_timeout(250)
    page.click("[data-rot='z']")
    page.wait_for_timeout(400)
    assert page.evaluate(_ROTATIONS) == [0, 90, 0, 0]

    page.click("[data-rot='z']")
    page.wait_for_timeout(400)
    assert page.evaluate(_ROTATIONS) == [0, 180, 0, 0]
    page.close()


def test_alt_rotate_turns_every_part(server, browser, part_stl):
    """The 'unless all selected' case: Alt applies the turn to the cluster."""
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 4)
    page.evaluate("()=>window.__select(0)")
    page.wait_for_timeout(250)

    page.keyboard.down("Alt")
    page.click("[data-rot='z']")
    page.keyboard.up("Alt")
    page.wait_for_timeout(500)
    assert page.evaluate(_ROTATIONS) == [90, 90, 90, 90]
    page.close()


def test_rotate_button_turns_the_view_when_nothing_is_selected(server, browser,
                                                               part_stl):
    """Deselected, the buttons keep their original job of turning the view.

    Inspecting a finished mould from another angle is still worth a button, and
    it must not quietly edit the arrangement to do it.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 4)
    before = page.evaluate(_ROTATIONS)

    page.keyboard.press("Escape")
    page.wait_for_timeout(250)
    assert page.evaluate("()=>window.__selected()") is None
    page.click("[data-rot='x']")
    page.wait_for_timeout(400)
    assert page.evaluate(_ROTATIONS) == before, \
        "rotating the view changed the arrangement"
    page.close()


# -- the clash check must not reject valid layouts -------------------------

@pytest.fixture(scope="module")
def oblong_stl(tmp_path_factory):
    """A long, thin part -- a plummer block's proportions.

    Aspect ratio is what exposes a sloppy clash test: a 165x83 part has a
    185 mm diagonal, so treating the diagonal as the footprint demands more
    than twice the clearance the short axis actually needs.
    """
    import trimesh
    p = tmp_path_factory.mktemp("oblong") / "oblong.stl"
    trimesh.creation.box(extents=[165, 83, 48]).export(p)
    return str(p)


@pytest.mark.parametrize("arrangement", ["grid", "radial"])
@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_the_automatic_layout_never_reports_a_clash(server, browser, oblong_stl,
                                                    arrangement, n):
    """The tool must not reject a layout it generated itself.

    The automatic layouts space instances at exactly the minimum clearance, so
    any false positive here means the check disagrees with the placement maths
    -- and the user is told to move parts that are already correctly spaced,
    with no way to satisfy it.

    Radial is the case that broke: every instance is rotated, and the old test
    substituted the part's diagonal for BOTH axes whenever either part was
    turned. A rotated rectangle's real footprint is
    (|w cos t| + |h sin t|) by (|w sin t| + |h cos t|) -- 83 mm wide at 90
    degrees, not 185.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    _load_part(page, server, oblong_stl)
    page.fill("#qty", str(n))
    page.dispatch_event("#qty", "input")
    page.select_option("#layout", arrangement)
    page.wait_for_timeout(700)

    assert not page.is_visible("#clash"), (
        f"the {arrangement} layout for {n} oblong parts was reported as "
        "clashing, but it is the spacing the layout itself chose")
    assert not page.is_disabled("#go")
    page.close()


def test_genuinely_overlapping_parts_are_still_caught(server, browser, oblong_stl):
    """Loosening the test must not blind it to a real overlap."""
    page = browser.new_page()
    _load_part(page, server, oblong_stl)
    _set_quantity(page, 2)
    # 20 mm apart on a 165 mm part: unambiguously interpenetrating
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:20, y:0, z:0, rot:0}])""")
    page.wait_for_timeout(500)
    assert page.is_visible("#clash")
    assert page.is_disabled("#go")
    page.close()


def test_a_rotated_part_packs_closer_on_its_short_axis(server, browser, oblong_stl):
    """Turning a part 90 degrees must let it sit closer across its new width.

    This is the whole point of rotating to save space: a 165x83 part turned on
    its side is 83 wide, so two of them need 83 + gap between centres, not
    165 + gap. A check that ignores the rotation cannot see the saving.
    """
    page = browser.new_page()
    _load_part(page, server, oblong_stl)
    _set_quantity(page, 2)
    # both turned 90 degrees, so each is 83 mm wide in X
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:90}, {x:110, y:0, z:0, rot:90}])""")
    page.wait_for_timeout(500)
    assert not page.is_visible("#clash"), \
        "two parts turned onto their short axis were reported as clashing"

    # unrotated at the same spacing they genuinely would overlap
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:110, y:0, z:0, rot:0}])""")
    page.wait_for_timeout(500)
    assert page.is_visible("#clash"), \
        "two unrotated 165 mm parts 110 mm apart must clash"
    page.close()


# -- the clash test measures metal, not boxes ------------------------------

@pytest.fixture(scope="module")
def hollow_stl(tmp_path_factory):
    """An oblong part with a bore -- a plummer block's shape and proportions.

    Two properties matter. It fills only part of its bounding box, so box
    separation and true separation are different numbers; and its vertices sit
    at corners, with none across the middle of its long faces, which is what
    defeats a vertex-sampling distance.
    """
    import trimesh
    block = trimesh.creation.box(extents=[165, 83, 48])
    bore = trimesh.creation.cylinder(radius=30, height=60)
    bore.apply_translation([30, 0, 24])
    p = tmp_path_factory.mktemp("hollow") / "hollow.stl"
    trimesh.boolean.difference([block, bore], engine="manifold").export(p)
    return str(p)


def test_closer_is_never_reported_as_clearer(server, browser, hollow_stl):
    """The distance principle: moving two parts together never un-flags them.

    A box test violates this constantly -- whether two boxes overlap depends on
    how each part is turned, so one pair reads as clashing while another pair
    that is genuinely closer reads clear. That inconsistency is the symptom
    that the measurement, not the threshold, is wrong.
    """
    page = browser.new_page()
    _load_part(page, server, hollow_stl)
    _set_quantity(page, 2)

    seen_clash = False
    for dx in (300, 250, 220, 200, 190, 185, 180, 178, 176, 174, 170, 160):
        page.evaluate(f"""()=>window.__setPlacements([
            {{x:0, y:0, z:0, rot:0}}, {{x:{dx}, y:0, z:0, rot:0}}])""")
        page.wait_for_timeout(250)
        clash = page.is_visible("#clash")
        if clash:
            seen_clash = True
        else:
            assert not seen_clash, (
                f"parts {dx} mm apart read as clear after a WIDER gap had "
                "already been flagged as clashing")
    assert seen_clash, "the parts never clashed even when overlapping"
    page.close()


def test_measured_distance_matches_the_real_geometry(server, browser, hollow_stl):
    """Near the threshold the reported distance must be the true one.

    Far apart the test may return a cheap lower bound, which is sound because
    the verdict is the same either way. Near the threshold it has to be exact,
    or the boundary lands in the wrong place.
    """
    import numpy as np
    import trimesh

    page = browser.new_page()
    _load_part(page, server, hollow_stl)
    _set_quantity(page, 2)
    mesh = trimesh.load(hollow_stl, force="mesh")

    for dx in (200, 190, 180, 175, 170):
        shown = page.evaluate(
            f"()=>window.__distance({{x:0,y:0,z:0,rot:0}},"
            f"{{x:{dx},y:0,z:0,rot:0}})")
        moved = mesh.copy()
        moved.apply_translation([dx, 0, 0])
        truth = float(np.abs(trimesh.proximity.ProximityQuery(mesh)
                             .signed_distance(moved.vertices)).min())
        assert abs(shown - truth) < 1.0, (
            f"at {dx} mm the viewer measured {shown:.1f} mm but the real "
            f"surface gap is {truth:.1f} mm")
    page.close()


def test_the_verdict_does_not_depend_on_how_parts_are_turned(server, browser,
                                                             hollow_stl):
    """Identical spacing must give an identical answer at any angle.

    Rotation-dependent verdicts are what produced the contradiction on screen:
    a part at one angle flagged while a closer part at another angle did not.
    """
    page = browser.new_page()
    _load_part(page, server, hollow_stl)
    _set_quantity(page, 2)
    for rot in (0, 45, 90, 137, 180):
        page.evaluate(f"""()=>window.__setPlacements([
            {{x:0, y:0, z:0, rot:{rot}}}, {{x:400, y:0, z:0, rot:{rot}}}])""")
        page.wait_for_timeout(300)
        assert not page.is_visible("#clash"), \
            f"two parts 400 mm apart were flagged when both were turned {rot}°"
    page.close()


def test_two_shell_walls_is_the_threshold(server, browser, hollow_stl):
    """The minimum is 2x shell thickness, not 1x.

    One shell thickness is the tempting answer and it is wrong: measured on the
    voxel grid, two castings 6 mm apart with a 6 mm shell have 6 mm of solid
    ceramic and 0 mm of air between them. The walls have fused and the castings
    come out as one block. Parting only appears past 2x.
    """
    page = browser.new_page()
    _load_part(page, server, hollow_stl)
    _set_quantity(page, 2)
    # shell 6 mm, pitch 1.5 mm -> minimum 15 mm
    page.fill("#shell", "6")
    page.dispatch_event("#shell", "input")
    page.wait_for_timeout(300)

    # 165 mm long part: centres 165 + gap apart gives exactly that gap
    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:185, y:0, z:0, rot:0}])""")   # 20 mm clear
    page.wait_for_timeout(300)
    assert not page.is_visible("#clash")

    page.evaluate("""()=>window.__setPlacements([
        {x:0, y:0, z:0, rot:0}, {x:175, y:0, z:0, rot:0}])""")   # 10 mm clear
    page.wait_for_timeout(300)
    assert page.is_visible("#clash"), \
        "10 mm between castings is under two 6 mm shell walls and must clash"
    page.close()


# -- the sidebar is tabbed -------------------------------------------------

def test_sidebar_has_four_tabs(server, browser, part_stl):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    names = page.evaluate(
        "()=>[...document.querySelectorAll('.tab')].map(b=>b.dataset.tab)")
    assert names == ["part", "process", "results", "exports"]
    page.close()


def test_only_one_panel_shows_at_a_time(server, browser, part_stl):
    page = browser.new_page()
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    assert page.is_visible("[data-panel=part]")

    page.click(".tab[data-tab=process]")
    page.wait_for_timeout(300)
    assert page.is_visible("[data-panel=process]")
    assert not page.is_visible("[data-panel=part]")
    page.close()


def test_generate_is_reachable_from_every_tab(server, browser, part_stl):
    """The action the whole sidebar exists to reach must not hide behind a tab.

    Putting it inside one panel means remembering which, and finding it again
    after every change made on another.
    """
    page = browser.new_page()
    page.goto(server, wait_until="load")
    page.wait_for_timeout(1500)
    for tab in ("part", "process"):
        page.click(f".tab[data-tab={tab}]")
        page.wait_for_timeout(250)
        assert page.is_visible("#go"), f"Generate is hidden on the {tab} tab"
    page.close()


def test_result_tabs_unlock_once_a_mould_exists(server, browser, part_stl):
    """A tab that opens onto an empty panel is worse than one visibly locked."""
    page = browser.new_page()
    _load_part(page, server, part_stl)
    assert page.is_disabled(".tab[data-tab=results]")
    assert page.is_disabled(".tab[data-tab=exports]")

    page.click("#go")
    page.wait_for_function("()=>window.__layers.shell !== null", timeout=300_000)
    page.wait_for_timeout(800)
    assert not page.is_disabled(".tab[data-tab=results]")
    assert not page.is_disabled(".tab[data-tab=exports]")
    # and the results are shown without hunting for them
    assert page.is_visible("[data-panel=results]")
    page.close()


# -- dragging stays responsive ---------------------------------------------

def test_a_drag_frame_is_cheap(server, browser, part_stl):
    """Moving a part must not run the exact clash test.

    The clash test is point-to-triangle over both parts' geometry. Running it
    on every drag frame made an 8-part cluster take three seconds per move,
    which is the lag this guards against. A drag frame updates transforms and
    schedules the check; the check itself runs once the pointer settles.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 8)

    per_frame = page.evaluate("""()=>{
        const t0 = performance.now();
        for(let f = 0; f < 60; f++){
            window.__placements()[0].x += 0.5;
            window.__dragFrame();
        }
        return (performance.now() - t0) / 60;
    }""")
    # 16 ms is one frame at 60 Hz; a drag update should be a fraction of that
    assert per_frame < 4.0, \
        f"a drag frame costs {per_frame:.1f} ms, which will feel laggy"
    page.close()


def test_the_clash_check_stays_interactive(server, browser, part_stl):
    """The deferred check still has to finish quickly enough not to freeze.

    Before the spatial grid this took over three seconds on eight parts.
    """
    page = browser.new_page()
    _load_part(page, server, part_stl)
    _set_quantity(page, 8)
    ms = page.evaluate("""()=>{
        const t0 = performance.now();
        for(let i = 0; i < 3; i++) window.__flagClashes();
        return (performance.now() - t0) / 3;
    }""")
    assert ms < 400, f"the clash check takes {ms:.0f} ms and will stutter"
    page.close()


# -- pour direction --------------------------------------------------------

def test_rotating_the_part_changes_which_face_is_up(server, browser, part_stl):
    """Up is the ground plane's positive side, and the X/Y/Z buttons turn the
    part relative to it.

    The first attempt captured the camera's screen-up instead, which cannot
    work: the viewer pins the scene to Z-up and orbiting walks the camera
    around a stationary part, so "screen up" describes where you are standing
    rather than which face of the casting should point at the sky. Every orbit
    returned a different tilted vector for the same, unmoved part.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    _load_part(page, server, part_stl)
    assert page.text_content("#upreadout").startswith("+Z")

    seen = []
    for _ in range(4):
        page.click("[data-rot='x']")
        page.wait_for_timeout(350)
        seen.append(page.text_content("#upreadout").split()[0])
    assert seen == ["+Y", "-Z", "-Y", "+Z"], f"got {seen}"
    page.close()


def test_orbiting_does_not_change_the_pour_direction(server, browser, part_stl):
    """Moving the camera must not change how the casting is gated."""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    _load_part(page, server, part_stl)
    before = page.text_content("#upreadout")

    page.mouse.move(900, 500)
    page.mouse.down()
    page.mouse.move(1200, 300)
    page.mouse.up()
    page.wait_for_timeout(500)

    assert page.text_content("#upreadout") == before, \
        "orbiting the camera changed the pour direction"
    page.close()


# ── the filesystem home page ────────────────────────────────────────────────
#
# `/` is a filesystem over saved projects; the editor sits at `/editor`. These
# drive the browser because the interesting half is the browser's: the store's
# own contract is covered in test_projects.py.


def _modified(page):
    """The title carries the unsaved marker now -- there is no status badge."""
    return page.eval_on_selector("#title", "e => e.classList.contains('modified')")


def _file_item_disabled(page, key):
    """Read a File-menu item's enabled state, opening the menu to do it."""
    page.click("#filebtn")
    page.wait_for_timeout(250)
    state = page.eval_on_selector(f'[data-file="{key}"]', "e => e.disabled")
    page.keyboard.press("Escape")
    page.wait_for_timeout(150)
    return state


def _drop_part(page, part_stl):
    """Put a part on the plate and wait for it to render."""
    page.set_input_files("#file", part_stl)
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(400)


def _save_as(page, name, folder_label=None):
    """Save through the File menu, which is where the action lives now."""
    page.click("#filebtn")
    page.wait_for_timeout(250)
    page.click('[data-file="save"]')
    page.wait_for_timeout(600)
    page.fill("#svname", name)
    if folder_label:
        page.select_option("#svfolder", label=folder_label)
    page.click("#svok")
    page.wait_for_timeout(1400)


def _new_folder(page, origin, name):
    """Make a folder from the home page and return to it. The store is shared
    across this module, so each test that counts rows works inside its own."""
    page.goto(origin, wait_until="load")
    page.wait_for_timeout(900)
    page.click("#newbtn")
    page.wait_for_timeout(200)
    page.click('[data-new="folder"]')
    page.wait_for_timeout(400)
    page.fill("#dlginput", name)
    page.click("#dlgok")
    page.wait_for_timeout(900)


def _open_folder(page, origin, name):
    """Open a named folder from the root and return its listing."""
    page.goto(origin, wait_until="load")
    page.wait_for_timeout(900)
    page.click(f'.rowitem:has-text("{name}")')
    page.wait_for_timeout(800)
    return page.evaluate("window.__fs()")


def test_the_home_page_is_the_filesystem(origin, browser):
    errors = []
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(origin, wait_until="load")
    page.wait_for_timeout(900)

    # It is the filesystem, not the viewer.
    assert page.query_selector("#crumbs") is not None
    assert page.query_selector("canvas") is None
    # Nothing saved yet, so it says so rather than showing an empty table.
    assert page.is_visible(".empty")
    assert not errors, f"page errors: {errors}"
    page.close()


def test_a_saved_project_appears_in_the_filesystem_and_reopens(origin, browser, part_stl):
    """The whole point: save from the editor, find it at home, open it back up."""
    errors = []
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    # Four parts, arranged by hand -- the arrangement is the part of the state
    # that no parameter can reconstruct, so it is what proves the save is real.
    page.fill("#qty", "4")
    page.dispatch_event("#qty", "input")
    page.wait_for_timeout(500)
    page.evaluate("window.__setPlacements("
                  "[{x:10,y:20,z:0,rot:45},{x:-10,y:20,z:0,rot:0},"
                  " {x:10,y:-20,z:0,rot:90},{x:-10,y:-20,z:0,rot:0}])")
    page.wait_for_timeout(400)
    before = page.evaluate("window.__placements()")

    _save_as(page, "Pump bracket")
    assert page.input_value("#title") == "Pump bracket"
    # A save is reopenable by URL, so a refresh does not land on a blank editor.
    assert "project=" in page.url

    page.goto(origin, wait_until="load")
    page.wait_for_timeout(900)
    listing = page.evaluate("window.__fs()")
    assert [i["name"] for i in listing["items"]] == ["Pump bracket"]

    page.click(".rowitem")
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(1200)

    assert page.input_value("#qty") == "4"
    after = page.evaluate("window.__placements()")
    assert len(after) == len(before)
    for a, b in zip(before, after):
        assert abs(a["x"] - b["x"]) < 0.01 and abs(a["rot"] - b["rot"]) < 0.01
    assert not errors, f"page errors: {errors}"
    page.close()


def test_the_pour_direction_survives_a_save(origin, browser, part_stl):
    """Which way is up is carried by the scene's rotation, not by a form field."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    page.click('#viewctl button:has-text("X")')
    page.wait_for_timeout(500)
    up_before = page.inner_text("#upreadout")
    assert up_before != "+Z (file)"      # the turn actually took

    _save_as(page, "Turned part")
    url = page.url

    page.goto(url, wait_until="load")
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(1200)
    assert page.inner_text("#upreadout") == up_before
    page.close()


def test_projects_can_be_filed_into_folders(origin, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    _new_folder(page, origin, "Brackets")
    names = [i["name"] for i in page.evaluate("window.__fs()")["items"]]
    assert "Brackets" in names

    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    _save_as(page, "Filed part", folder_label="↳ Brackets")

    # The project is in the folder, not at the root beside it.
    inside = _open_folder(page, origin, "Brackets")
    assert [i["name"] for i in inside["items"]] == ["Filed part"]
    assert [c["name"] for c in inside["crumbs"]] == ["Brackets"]
    page.close()


def test_saving_is_refused_until_there_is_a_part(origin, browser, part_stl):
    """A project with no mesh would reopen empty, so Save stays shut."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    assert _file_item_disabled(page, "save")
    _drop_part(page, part_stl)
    assert not _file_item_disabled(page, "save")
    page.close()


def test_unsaved_changes_are_flagged_after_a_save(origin, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    _new_folder(page, origin, "Watched")

    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    _save_as(page, "Watched part", folder_label="↳ Watched")
    assert not _modified(page)

    page.fill("#qty", "3")
    page.dispatch_event("#qty", "input")
    page.wait_for_timeout(400)
    assert _modified(page)

    # Saving again clears the flag, and does not fork a second project. A
    # project that already has a name and a home needs no dialog to re-save.
    page.click("#filebtn")
    page.wait_for_timeout(250)
    page.click('[data-file="save"]')
    page.wait_for_timeout(1400)
    assert not page.is_visible("#veil")
    assert not _modified(page)

    # Re-saving overwrites; it does not fork a second project beside the first.
    inside = _open_folder(page, origin, "Watched")
    assert [i["name"] for i in inside["items"]] == ["Watched part"]
    page.close()


def test_a_project_is_renamed_from_its_own_header(origin, browser, part_stl):
    """The title in the toolbar is the name -- there is no rename dialog."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    _new_folder(page, origin, "Renames")

    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    _save_as(page, "First name", folder_label="↳ Renames")

    page.fill("#title", "Second name")
    page.dispatch_event("#title", "blur")
    page.wait_for_timeout(1200)

    inside = _open_folder(page, origin, "Renames")
    assert [i["name"] for i in inside["items"]] == ["Second name"]
    page.close()


def test_the_toolbar_waits_for_a_part(origin, browser, part_stl):
    """Naming a project before there is a part promises a save that cannot
    happen, so the field and the button stay shut until one is loaded."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    assert page.is_disabled("#title")
    assert _file_item_disabled(page, "save")
    assert _file_item_disabled(page, "saveas")

    _drop_part(page, part_stl)
    assert not page.is_disabled("#title")
    assert not _file_item_disabled(page, "save")
    assert _modified(page)      # loaded but never saved
    page.close()


# ── the edit bar and the File menu ──────────────────────────────────────────


def _set_qty(page, value):
    """Type into the quantity field the way a person does.

    `fill()` sets the value without ever focusing the field, and the history
    takes its "before" snapshot on focus -- so filling would record the edit
    rather than the state preceding it.
    """
    page.click("#qty")
    page.keyboard.press("Meta+a")
    page.keyboard.type(str(value))
    page.keyboard.press("Tab")
    page.wait_for_timeout(600)


def test_undo_and_redo_walk_the_arrangement_back_and_forward(origin, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    # Nothing has happened yet, so there is nothing to step back to.
    assert page.is_disabled("#undo")
    assert page.is_disabled("#redo")

    _set_qty(page, 4)
    _set_qty(page, 6)
    assert len(page.evaluate("window.__placements()")) == 6
    assert not page.is_disabled("#undo")

    # Each typed value is ONE step, not one per keystroke.
    page.click("#undo")
    page.wait_for_timeout(700)
    assert page.input_value("#qty") == "4"
    assert len(page.evaluate("window.__placements()")) == 4

    page.click("#undo")
    page.wait_for_timeout(700)
    assert page.input_value("#qty") == "1"
    assert page.is_disabled("#undo")

    page.click("#redo")
    page.wait_for_timeout(700)
    assert page.input_value("#qty") == "4"
    page.click("#redo")
    page.wait_for_timeout(700)
    assert page.input_value("#qty") == "6"
    assert page.is_disabled("#redo")
    assert not errors, f"page errors: {errors}"
    page.close()


def test_a_new_edit_discards_the_redo_branch(origin, browser, part_stl):
    """Stepping back and then editing forks the timeline; redo cannot rejoin."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    _set_qty(page, 4)
    page.click("#undo")
    page.wait_for_timeout(700)
    assert not page.is_disabled("#redo")

    _set_qty(page, 3)
    assert page.is_disabled("#redo")
    page.close()


def test_a_hand_arrangement_is_one_undo_step(origin, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    _set_qty(page, 2)
    before = page.evaluate("window.__placements()")

    page.evaluate("window.__setPlacements("
                  "[{x:40,y:40,z:0,rot:30},{x:-40,y:-40,z:0,rot:0}])")
    page.wait_for_timeout(600)
    assert page.evaluate("window.__placements()")[0]["x"] == 40

    page.click("#undo")
    page.wait_for_timeout(700)
    after = page.evaluate("window.__placements()")
    assert len(after) == len(before)
    assert abs(after[0]["x"] - before[0]["x"]) < 0.01
    page.close()


def test_save_as_forks_a_second_project(origin, browser, part_stl):
    """Save as copies; it must not write over the project it came from."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    _new_folder(page, origin, "Forks")

    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    _save_as(page, "Original", folder_label="↳ Forks")
    first_url = page.url

    _set_qty(page, 5)
    page.click("#filebtn")
    page.wait_for_timeout(250)
    page.click('[data-file="saveas"]')
    page.wait_for_timeout(700)
    page.fill("#svname", "Variant")
    page.select_option("#svfolder", label="↳ Forks")
    page.click("#svok")
    page.wait_for_timeout(1600)
    assert page.url != first_url

    inside = _open_folder(page, origin, "Forks")
    assert sorted(i["name"] for i in inside["items"]) == ["Original", "Variant"]

    # The original keeps the state it was saved with.
    page.goto(first_url, wait_until="load")
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(1200)
    assert page.input_value("#qty") == "1"
    page.close()


def test_exports_are_chosen_from_the_file_menu(origin, browser, part_stl):
    """The Exports tab is gone; downloads are picked in a dialog instead."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    assert page.query_selector('.tab[data-tab="exports"]') is None

    _drop_part(page, part_stl)
    # Nothing to export until a mould exists.
    assert _file_item_disabled(page, "export")

    page.click("#go")
    _wait_done(page)
    assert not _file_item_disabled(page, "export")

    page.click("#filebtn")
    page.wait_for_timeout(250)
    page.click('[data-file="export"]')
    page.wait_for_timeout(600)
    names = page.eval_on_selector_all(".exrow .exname", "e => e.map(x => x.textContent)")
    assert any(n.endswith("_shell.stl") for n in names)
    assert any(n.endswith("_section.png") for n in names)
    assert "5 files" in page.inner_text("#exok")

    # Unpicking a row takes it out of the download.
    page.eval_on_selector_all("#exlist input", """els => {
      els[0].checked = false;
      els[0].dispatchEvent(new Event('change', {bubbles: true}));
    }""")
    page.wait_for_timeout(300)
    assert "4 files" in page.inner_text("#exok")

    with page.expect_download(timeout=30000) as dl:
        page.click("#exok")
    assert dl.value.suggested_filename.endswith(".stl")
    page.close()


def test_the_zoom_picker_is_relative_to_the_fitted_view(origin, browser, part_stl):
    """100% is whatever `fit()` produced, so a percentage means the same thing
    for a 20 mm part and a 400 mm one."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    page.wait_for_timeout(500)

    assert page.inner_text("#zoomval") == "100%"
    fitted = page.evaluate("window.__probe().dist")

    page.click("#zoombtn")
    page.wait_for_timeout(250)
    page.click('[data-zoom="200"]')
    page.wait_for_timeout(600)
    assert page.inner_text("#zoomval") == "200%"
    # Twice the magnification is half the distance.
    assert abs(page.evaluate("window.__probe().dist") - fitted / 2) < fitted * 0.05

    page.click("#zoombtn")
    page.wait_for_timeout(250)
    page.click('[data-zoom="fit"]')
    page.wait_for_timeout(700)
    assert page.inner_text("#zoomval") == "100%"
    assert not errors, f"page errors: {errors}"
    page.close()


def test_a_built_mould_is_saved_and_comes_back(origin, browser, part_stl):
    """Saving after a build must keep the build.

    The mould was the half that went missing: the project reopened with its
    part and its parameters but claimed it had never been built, because the
    meshes lived in a job directory that nothing pointed at any more.
    """
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    page.click("#go")
    _wait_done(page)
    built = page.evaluate("window.__probe().layers")
    assert built["shell"] and built["fired"]

    _save_as(page, "Built project")
    url = page.url

    page.goto(url, wait_until="load")
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(2500)

    back = page.evaluate("window.__probe().layers")
    assert back["shell"], "the shell did not come back with the project"
    assert back["fired"], "the fired shell did not come back with the project"
    # The results table is repopulated, and exporting is possible again.
    assert page.eval_on_selector_all("#stats tr", "e => e.length") > 0
    assert not _file_item_disabled(page, "export")
    # Reopening lands on the part, not on a table of numbers already seen.
    # (the tabs are uppercased in CSS, so the DOM text is "Part")
    assert page.eval_on_selector(".tab.on", "e => e.textContent").strip() == "Part"
    assert not errors, f"page errors: {errors}"
    page.close()


def test_editing_the_setup_drops_the_saved_mould(origin, browser, part_stl):
    """A project must never reopen showing a mould built for other parameters."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    page.click("#go")
    _wait_done(page)
    _save_as(page, "Stale check")
    url = page.url

    # The build switches to Results; the quantity lives on Part.
    page.click('.tab[data-tab="part"]')
    page.wait_for_timeout(300)
    _set_qty(page, 3)
    assert not page.evaluate("window.__probe().layers")["shell"]
    assert _file_item_disabled(page, "export")

    page.click("#filebtn")
    page.wait_for_timeout(250)
    page.click('[data-file="save"]')
    page.wait_for_timeout(2200)

    page.goto(url, wait_until="load")
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(2200)
    assert page.input_value("#qty") == "3"
    assert not page.evaluate("window.__probe().layers")["shell"]
    page.close()


def test_save_stays_available_after_a_build(origin, browser, part_stl):
    """Generating a mould does not touch the parameters, so a project that was
    already saved stayed "clean" -- and Save greyed out with the mould, the
    thing most worth keeping, sitting unsaved on screen."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)
    _save_as(page, "Clean then built")
    url = page.url

    # Saved and untouched: Save is still offered, not greyed out.
    assert not _file_item_disabled(page, "save")

    page.click("#go")
    _wait_done(page)
    assert not _file_item_disabled(page, "save")

    page.click("#filebtn")
    page.wait_for_timeout(250)
    page.click('[data-file="save"]')
    page.wait_for_timeout(2200)

    page.goto(url, wait_until="load")
    page.wait_for_function("()=>window.__probe().layers.part === true", timeout=40000)
    page.wait_for_timeout(2500)
    assert page.evaluate("window.__probe().layers")["shell"]
    page.close()


def test_file_menu_items_each_get_their_own_row(origin, browser):
    """An item with no keyboard shortcut had nothing to push against, so it
    shrank to its text and the next item sat down beside it."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    page.click("#filebtn")
    page.wait_for_timeout(300)

    boxes = page.eval_on_selector_all("#filemenu .mi", """els => els.map(e => {
      const r = e.getBoundingClientRect();
      return {top: Math.round(r.top), width: Math.round(r.width)};
    })""")
    assert len(boxes) == 8
    assert len({b["top"] for b in boxes}) == len(boxes), "items share a row"
    assert len({b["width"] for b in boxes}) == 1, "items are not full width"
    page.close()


# ── the Docs-style header ───────────────────────────────────────────────────


def test_the_header_is_two_rows_under_a_logo(origin, browser):
    """Name on top, menus beneath, mark on the left — and no wordmark text."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)

    assert "Ferrum" not in page.eval_on_selector("#toolbar", "e => e.innerText")
    assert page.eval_on_selector("#logo img", "e => e.complete && e.naturalWidth > 0")
    assert page.eval_on_selector("#logo", "e => e.getAttribute('href')") == "/"

    title_top = page.eval_on_selector("#title", "e => e.getBoundingClientRect().top")
    menu_top = page.eval_on_selector("#menubar", "e => e.getBoundingClientRect().top")
    assert title_top < menu_top, "the menus should sit under the name"

    assert page.eval_on_selector_all(
        "#menubar .menubtn", "e => e.map(x => x.textContent.trim())"
    )[:3] == ["File", "Edit", "View"]
    assert not errors, f"page errors: {errors}"
    page.close()


def test_the_logo_goes_back_to_the_filesystem(origin, browser):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    page.click("#logo")
    page.wait_for_timeout(1200)
    assert page.query_selector("#crumbs") is not None
    page.close()


def test_the_edit_menu_drives_undo(origin, browser, part_stl):
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    # Nothing to undo yet.
    page.click("#editbtn")
    page.wait_for_timeout(250)
    assert page.eval_on_selector('[data-edit="undo"]', "e => e.disabled")
    page.keyboard.press("Escape")
    page.wait_for_timeout(150)

    _set_qty(page, 4)
    page.click("#editbtn")
    page.wait_for_timeout(250)
    assert not page.eval_on_selector('[data-edit="undo"]', "e => e.disabled")
    page.click('[data-edit="undo"]')
    page.wait_for_timeout(700)
    assert page.input_value("#qty") == "1"
    page.close()


def test_the_view_menu_toggles_the_ground_plane(origin, browser, part_stl):
    """The plane is the surface the casting stands on, so it starts on and the
    menu shows which way the switch is set."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    assert page.evaluate("window.__grid.visible")
    page.click("#viewbtn")
    page.wait_for_timeout(250)
    assert page.eval_on_selector('[data-view="plane"]', "e => e.classList.contains('on')")

    page.click('[data-view="plane"]')
    page.wait_for_timeout(500)
    assert not page.evaluate("window.__grid.visible")

    # Shift+G is the same switch.
    page.click("body")
    page.keyboard.press("Shift+G")
    page.wait_for_timeout(400)
    assert page.evaluate("window.__grid.visible")
    page.close()


def test_view_menu_rows_do_not_wrap(origin, browser):
    """The tick and the shortcut both sit on the row without pushing it open."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    page.click("#viewbtn")
    page.wait_for_timeout(300)
    heights = page.eval_on_selector_all(
        "#viewmenu .mi", "e => e.map(x => Math.round(x.getBoundingClientRect().height))")
    assert len(set(heights)) == 1, f"rows are uneven: {heights}"
    page.close()


def test_the_actions_sit_together_in_the_middle_of_the_bar(origin, browser, part_stl):
    """Undo/redo, zoom and Generate are one inline group on the bar's centre
    line, kept clear of the File/Edit/View menus on the left."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)

    def geometry():
        return page.evaluate("""() => {
          const g = s => { const r = document.querySelector(s).getBoundingClientRect();
            return {x: r.x, right: r.right, cy: Math.round(r.y + r.height / 2)}; };
          return {bar: g('#toolbar'), menubar: g('#menubar'), actions: g('#actions'),
                  undo: g('#undo'), zoom: g('#zoombtn'), go: g('#go')};
        }""")

    g = geometry()
    # All three on one line.
    assert len({g["undo"]["cy"], g["zoom"]["cy"], g["go"]["cy"]}) == 1
    # Clear of the text menus.
    assert g["actions"]["x"] > g["menubar"]["right"]
    # On the bar's centre.
    bar_mid = (g["bar"]["x"] + g["bar"]["right"]) / 2
    act_mid = (g["actions"]["x"] + g["actions"]["right"]) / 2
    assert abs(bar_mid - act_mid) < 4, f"group is off centre: {bar_mid} vs {act_mid}"

    # A long name must not shove the group off the centre line.
    _drop_part(page, part_stl)
    page.fill("#title", "A very long project name that runs on and on")
    page.click("body")
    page.wait_for_timeout(400)
    g = geometry()
    act_mid = (g["actions"]["x"] + g["actions"]["right"]) / 2
    assert abs((g["bar"]["x"] + g["bar"]["right"]) / 2 - act_mid) < 4
    page.close()


def test_menus_open_above_the_viewport(origin, browser, part_stl):
    """Positioning the header created a stacking context that trapped the
    menus behind the canvas, which then swallowed every click in them."""
    page = browser.new_page(viewport={"width": 1400, "height": 800})
    page.goto(origin + "/editor", wait_until="load")
    page.wait_for_timeout(1800)
    _drop_part(page, part_stl)

    page.click("#zoombtn")
    page.wait_for_timeout(300)
    # The click has to actually land on the item, not on the canvas over it.
    page.click('[data-zoom="200"]')
    page.wait_for_timeout(600)
    assert page.inner_text("#zoomval") == "200%"
    page.close()
