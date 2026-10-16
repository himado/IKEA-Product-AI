"""
Step 13 (first version) - placement engine for a scanned room.

Step 7 puts each product in some free zone on its own; in a real room things belong together and
must not block anything. Rules:
  - a reading group (armchair + floor lamp + side/coffee table) is placed together along a wall,
    lamp on the corner side, table on the other side, armchair facing into the room;
  - nothing tall goes into the TV <-> sofa viewing area (a coffee table may);
  - the door swing stays clear (already removed from the free grid in step 12);
  - every footprint + its usage clearance must lie on free cells of free_grid.npy.
Scores: close to the window (reading light), in a corner, far from the door.
Items not in a group keep the zone placement step 7 gave them.
"""

import numpy as np

GROUP = {"armchair", "floor_lamp", "coffee_side_table"}
FRONT_CLEAR = {"armchair": 0.5, "floor_lamp": 0.0, "coffee_side_table": 0.0}


class Grid:
    def __init__(self, prof, free):
        self.free, self.res = free, prof["free_grid"]["res"]
        self.W, self.D = prof["size_m"]
        self.block = np.zeros_like(free)
        self.notes = []
        tv = [o for o in prof["objects"] if o["category"] == "tv_bench"]
        sofa = [o for o in prof["objects"] if o["category"] == "sofa"]
        for a in tv:
            for b in sofa:
                # facing each other across x or z: block the space between them over their overlap
                oz0, oz1 = max(a["z0"], b["z0"]), min(a["z1"], b["z1"])
                ox0, ox1 = max(a["x0"], b["x0"]), min(a["x1"], b["x1"])
                if oz1 - oz0 > 0.4:
                    x0, x1 = sorted([a["x1"], b["x0"]]) if a["x1"] <= b["x0"] else sorted([b["x1"], a["x0"]])
                    self.mark(x0, oz0, x1, oz1)
                    self.notes.append(f"giữ trống tầm nhìn TV ↔ sofa (x {x0:.1f}–{x1:.1f} m)")
                elif ox1 - ox0 > 0.4:
                    z0, z1 = sorted([a["z1"], b["z0"]]) if a["z1"] <= b["z0"] else sorted([b["z1"], a["z0"]])
                    self.mark(ox0, z0, ox1, z1)
                    self.notes.append(f"giữ trống tầm nhìn TV ↔ sofa (z {z0:.1f}–{z1:.1f} m)")

    def idx(self, x0, z0, x1, z1):
        r = self.res
        return slice(int(np.floor(x0 / r)), int(np.ceil(x1 / r))), slice(int(np.floor(z0 / r)), int(np.ceil(z1 / r)))

    def mark(self, x0, z0, x1, z1):
        self.block[self.idx(x0, z0, x1, z1)] = True

    def ok(self, x0, z0, x1, z1, view_ok=False):
        """All cells free; view_ok=True allows the TV <-> sofa view area (legroom is low, it does not
        block the view - the footprint itself must stay out)."""
        if x0 < -1e-6 or z0 < -1e-6 or x1 > self.W + 1e-6 or z1 > self.D + 1e-6:
            return False
        sx, sz = self.idx(x0 + 0.01, z0 + 0.01, x1 - 0.01, z1 - 0.01)
        f = self.free[sx, sz]
        return f.size > 0 and f.all() and (view_ok or not self.block[sx, sz].any())


def wall_frames(W, D):
    """For each wall: origin, along-wall unit vector u, inward normal n (room metres), wall length,
    and the step-7 'front' name of n."""
    return {"bottom": ((0, 0), (1, 0), (0, 1), W, "up"), "top": ((0, D), (1, 0), (0, -1), W, "down"),
            "left": ((0, 0), (0, 1), (1, 0), D, "right"), "right": ((W, 0), (0, 1), (-1, 0), D, "left")}


def rect_on_wall(o, u, n, along0, w, d):
    """Footprint of a w (along wall) x d (from wall) item starting at `along0` -> x0, z0, x1, z1."""
    p0 = np.array(o) + np.array(u) * along0
    p1 = p0 + np.array(u) * w + np.array(n) * d
    return min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])


def place_group(prof, free, items):
    """items: {category: (w_cm, d_cm)} of the group members present. -> (placements, notes) or None"""
    g = Grid(prof, free)
    W, D = prof["size_m"]
    win = [op for op in prof["openings"] if op["type"] == "window"]
    door = [op for op in prof["openings"] if op["type"] == "door"]
    frames = wall_frames(W, D)

    def center_of(op):
        o, u, _, _, _ = frames[op["wall"]]
        m = (op["u0"] + op["u1"]) / 2
        return np.array(o) + np.array(u) * m

    order = [c for c in ("floor_lamp", "armchair", "coffee_side_table") if c in items]
    best = None
    for wall, (o, u, n, L, front) in frames.items():
        for start in np.arange(0, L, 0.05):
            for flip in (False, True):
                seq = order[::-1] if flip else order
                rects, pos, fine = {}, start, True
                for c in seq:
                    w, d = items[c][0] / 100, items[c][1] / 100
                    r = rect_on_wall(o, u, n, pos, w, d)
                    clear = rect_on_wall(o, u, n, pos, w, d + FRONT_CLEAR[c])
                    if not g.ok(*r) or not g.ok(*clear, view_ok=True):
                        fine = False
                        break
                    rects[c] = r
                    pos += w + 0.05
                if not fine:
                    continue
                arm = rects.get("armchair") or next(iter(rects.values()))
                ac = np.array([(arm[0] + arm[2]) / 2, (arm[1] + arm[3]) / 2])
                score = 0.0
                if win:
                    score -= min(np.linalg.norm(ac - center_of(w_)) for w_ in win)          # reading light
                if door:
                    score -= 1.5 * max(0.0, 1.5 - min(np.linalg.norm(ac - center_of(d_)) for d_ in door))
                ends = [start, pos - 0.05]
                if min(ends[0], L - ends[1]) < 0.15:
                    score += 0.6                                                             # tucked in a corner
                if best is None or score > best[0]:
                    best = (score, wall, front, rects)
    if best is None:
        return None, g.notes
    _, wall, front, rects = best
    return {c: {"wall": wall, "front": front, "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3]} for c, r in rects.items()}, \
        g.notes + [f"góc đọc sách dọc tường {wall}"]
