"""Figures fig:layout and fig:calibration (Chapter 3): screenshots of the speller and
the calibration GUI, rendered with the GUIs' own scene classes at the size of the
laptop's display (1920x1200) and cropped to the ring.

The speller copies the first phrase of the MacKenzie & Soukoreff set with the gpt2
suggestions, following the cue (Speller.next_target) as a perfect user would, and is
captured at two consecutive selections: level 1 with the cue on the suggestion group,
then that group opened with the cue on the word.

Run from the repository root on a virtual display (needs the nlp extra):
    xvfb-run -s "-screen 0 1920x1200x24" .venv/bin/python thesis/figures/make_screenshots.py
Writes thesis/figures/speller_level1.png, speller_level2.png and calibration_cue.png.
"""

import sys
from pathlib import Path

import numpy as np
import pyglet
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from calibration import CUE_GAP, CUE_THICKNESS, FlickerScene  # noqa: E402
from speller_gui import SpellerScene  # noqa: E402
from ssvep import config, language  # noqa: E402
from ssvep.speller import Speller  # noqa: E402

OUT = Path(__file__).parent
WIDTH, HEIGHT = 1920, 1200  # the laptop's display
PHRASE = "MY WATCH FELL IN THE WATER"  # first phrase of the MacKenzie & Soukoreff set
CAPTURE_AT = "MY WATCH FEL"  # typed text of the two speller screenshots
CALIBRATION_CUE = 2  # target with the cue frame in the calibration screenshot
MARGIN = 20  # px of black kept outside the cue frame


def render(window: pyglet.window.Window, scene, path: Path) -> None:
    """Draw ``scene`` once (targets at rest) and save the part of the screen that holds
    the ring, with room for a cue frame around any target, so every image has the
    same size and scale."""
    scene.set_frame(0)
    window.switch_to()
    window.clear()
    scene.draw()
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    data = buffer.get_image_data()
    rgb = np.frombuffer(data.get_data("RGBA", WIDTH * 4), np.uint8)
    rgb = rgb.reshape(HEIGHT, WIDTH, 4)[::-1, :, :3]  # OpenGL rows run bottom-up
    reach = scene.size / 2 + CUE_GAP + CUE_THICKNESS + MARGIN  # px from a centre
    xs, ys = scene.centres[:, 0], HEIGHT - scene.centres[:, 1]  # image rows top-down
    left, right = int(max(xs.min() - reach, 0)), int(min(xs.max() + reach, WIDTH))
    top, bottom = int(max(ys.min() - reach, 0)), int(min(ys.max() + reach, HEIGHT))
    Image.fromarray(np.ascontiguousarray(rgb[top:bottom, left:right])).save(path)
    print(f"Saved {path} ({right - left}x{bottom - top} px of {WIDTH}x{HEIGHT})")


def main():
    window = pyglet.window.Window(WIDTH, HEIGHT, caption="screenshots")
    freqs, phases = config.TARGET_FREQS, config.TARGET_PHASES
    slots, rate = config.TARGET_SLOTS, config.REFRESH_RATE

    calibration = FlickerScene(window, freqs, phases, slots, rate)
    calibration.set_cue(CALIBRATION_CUE)
    render(window, calibration, OUT / "calibration_cue.png")

    speller = Speller(suggest=language.make_suggester("gpt2"))
    speller.set_phrase(PHRASE)
    while not (speller.text == CAPTURE_AT and speller.group is None):
        speller.select(speller.next_target())
    scene = SpellerScene(window, freqs, phases, slots, rate)
    for level in (1, 2):
        target = speller.next_target()
        scene.set_labels(speller.labels())
        scene.set_text(speller.phrase, speller.text, "")
        scene.set_cue(target)
        render(window, scene, OUT / f"speller_level{level}.png")
        speller.select(target)
    window.close()


if __name__ == "__main__":
    main()
