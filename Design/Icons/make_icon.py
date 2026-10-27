#!/usr/bin/env python3
"""DYTM's Liquid Glass app icon, written as an Icon Composer bundle.

    python3 Design/Icons/make_icon.py      # rewrites Design/Icons/AppIcon.icon

A tilted die showing six, its pips laid out as a check (probabilistic agents, deterministic verification),
as one white glass glyph on Apple's developer-tool blueprint background (blue gradient, fine grid, its own
Dark gradient). AppIcon is Xcode's default app icon name: move the bundle next to the iOS app's
Assets.xcassets once that app exists.

Geometry is on Icon Composer's 1024 pt canvas: solid shapes clockwise, holes counter-clockwise, nonzero fill.
Requires Pillow, numpy and iconkit from the liquid-glass-icons skill (point ICONKIT_DIR at it if it moved).
"""
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.environ.get("ICONKIT_DIR", os.path.expanduser("~/.claude/skills/liquid-glass-icons/scripts")))
from iconkit import C, blueprint_icon, circle, polygon, rotate, write_svg  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AppIcon.icon")


def arc(cx, cy, r, a0, a1, step=2.0):
    """Points on a circle from angle a0 to a1, in degrees on screen (0 = right, 90 = down)."""
    n = max(2, math.ceil(abs(a1 - a0) / step) + 1)
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / (n - 1))),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / (n - 1)))) for i in range(n)]


def rounded_square(cx, cy, side, r):
    h = side / 2
    return (arc(cx + h - r, cy - h + r, r, -90, 0) + arc(cx + h - r, cy + h - r, r, 0, 90) +
            arc(cx - h + r, cy + h - r, r, 90, 180) + arc(cx - h + r, cy - h + r, r, 180, 270))


def dice():
    """The die, tilted, with six pip holes forming an upright check: two on the short arm, the elbow, three on the
    long arm. Pips are circles, so they are placed straight on screen while the die body rotates."""
    tilt, side, corner, pip, step = -10, 620, 146, 44, 114
    down, up = math.radians(42), math.radians(-56)
    pips = [(-k * step * math.cos(down), -k * step * math.sin(down)) for k in (2, 1)] + [(0, 0)]
    pips += [(k * step * math.cos(up), k * step * math.sin(up)) for k in (1, 2, 3)]
    xs, ys = zip(*pips)
    dx, dy = C - (min(xs) + max(xs)) / 2, C - (min(ys) + max(ys)) / 2
    return " ".join([polygon(rotate(rounded_square(C, C, side, corner), tilt))] +
                    [circle(x + dx, y + dy, pip, clockwise=False) for x, y in pips])


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="dytm-glyph-")
    shutil.rmtree(OUT, ignore_errors=True)
    blueprint_icon(OUT, write_svg(os.path.join(tmp, "dice.svg"), dice()), glyph_name="Dice")
    shutil.rmtree(tmp)
    print("wrote", OUT)
