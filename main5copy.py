"""
BRAITENBERG VEHICLE 5 -- Threshold-Brain Simulation
=====================================================
Based on Valentino Braitenberg's "Vehicles: Experiments in Synthetic
Psychology" (Vehicle 5 chapter). This is a from-scratch, single-file
Pygame simulation. No game engine, no external assets.

WHAT'S BEING MODELED (straight from the text)
-----------------------------------------------
1. THRESHOLD DEVICE
   "gives no output if its input line carries a signal below the
   threshold, and it gives full output beyond the threshold."
   -> class ThresholdNode

2. EXCITATORY / INHIBITORY CONNECTIONS
   "activation... inhibitory... connections... counteract the
   activation that comes from other sources."
   -> Brain.connect(src, dst, weight)   weight > 0 = excitatory
                                         weight < 0 = inhibitory

3. DELAY ("a short delay... during this time the gadget performs
   its little calculation")
   -> Brain.step() computes every node's NEW output from the OLD
   (previous frame's) outputs of its inputs, then commits them all
   at once. This one-frame lag *is* the delay.

4. NAMES / SELECTIVE RECOGNITION
   "the olive green vehicle is its special friend... something
   like proper nouns... NAMES that refer to very particular
   objects."
   -> Each sensor is "tuned" (like a receptor tuned to a frequency)
   to one color only. A gray/neutral source is invisible to the
   friend-sensor; only the olive-green FRIEND registers. That is
   the vehicle's one and only "name" in its little world.

5. MEMORY
   "activates another threshold device which in turn is connected
   back to the first device. Once a red light is sighted, the two
   devices will activate one another forever."
   -> Two nodes, MEM_A and MEM_B, are wired into a reciprocal
   (self-sustaining) loop. One pulse from the red-light sensor
   latches them on permanently -- this is literally the "bell that
   rings forever" example from the book.

6. INHIBITION SHAPING FUTURE BEHAVIOR
   Once the memory latch is on, it sends an inhibitory connection
   into the APPROACH node, making the vehicle more hesitant/slower
   to approach even its friend -- a small demonstration of how a
   stored memory can permanently bias later decisions.

CONTROLS
--------
  Left click        : drop a new source of the currently selected color
  1                  : select FRIEND (olive) color to place
  2                  : select DANGER (red) color to place
  3                  : select NEUTRAL (gray) color to place  (ignored by vehicle)
  [ / ]              : decrease / increase the APPROACH threshold live
  R                  : reset the whole simulation
  ESC / close window : quit
"""

import math
import random
import sys

import pygame

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
WORLD_W, HEIGHT = 760, 700
PANEL_W = 420
WIDTH = WORLD_W + PANEL_W
FPS = 60

BG = (18, 20, 24)
PANEL_BG = (24, 26, 32)
GRID = (30, 33, 40)
TEXT = (225, 227, 232)
DIM = (130, 134, 145)

FRIEND_COLOR = (154, 176, 44)   # "olive green" special friend
DANGER_COLOR = (206, 66, 66)    # red -> triggers memory
NEUTRAL_COLOR = (110, 112, 118) # gray -> vehicle is blind to these (no name)

SENSOR_RANGE = 260.0
VEHICLE_RADIUS = 14
SENSOR_SPREAD_DEG = 40  # half-angle between the two sensors

random.seed()


# --------------------------------------------------------------------------
# 1 & 2 & 3.  THE THRESHOLD BRAIN
# --------------------------------------------------------------------------
class ThresholdNode:
    """A single threshold device. See docstring point 1."""

    def __init__(self, name, threshold, is_sensor=False):
        self.name = name
        self.threshold = threshold
        self.is_sensor = is_sensor
        self.output = 0.0   # current (already "delayed") output
        self._pending = 0.0 # next output, computed but not yet committed


class Brain:
    """A little network of ThresholdNodes connected by weighted wires.

    Positive weight  = excitatory  ("activation" in the book)
    Negative weight  = inhibitory  ("counteracts the activation")
    """

    def __init__(self):
        self.nodes: dict[str, ThresholdNode] = {}
        self.connections: list[tuple[str, str, float]] = []

    def add(self, name, threshold=0.0, is_sensor=False):
        self.nodes[name] = ThresholdNode(name, threshold, is_sensor)
        return self.nodes[name]

    def connect(self, src, dst, weight):
        self.connections.append((src, dst, weight))

    def set_sensor(self, name, value):
        """Sensors bypass thresholding -- they just report a raw reading."""
        self.nodes[name].output = value

    def step(self):
        """Compute every non-sensor node's new state from last frame's
        outputs, then commit everything simultaneously. This simultaneity
        is what creates the book's "short delay" of one calculation step,
        and it's also what makes a reciprocal MEM_A<->MEM_B loop stable
        instead of exploding or racing.
        """
        totals = {n: 0.0 for n in self.nodes if not self.nodes[n].is_sensor}
        for src, dst, w in self.connections:
            totals[dst] += w * self.nodes[src].output

        for name, total in totals.items():
            node = self.nodes[name]
            node._pending = 1.0 if total >= node.threshold else 0.0

        for name in totals:
            self.nodes[name].output = self.nodes[name]._pending

    def raw_input_sum(self, dst):
        """Helper purely for the UI: what is currently feeding a node,
        before thresholding? Useful to display on screen."""
        total = 0.0
        for src, d, w in self.connections:
            if d == dst:
                total += w * self.nodes[src].output
        return total


# --------------------------------------------------------------------------
# WORLD OBJECTS
# --------------------------------------------------------------------------
class Source:
    """A stationary light/signal source of a given color."""

    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color


class Vehicle:
    """Vehicle 5: two sensors, a threshold brain, two motors."""

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.angle = random.uniform(0, 2 * math.pi)
        self.left_speed = 0.0
        self.right_speed = 0.0
        self._wander_bias = 0.0  # slow-drifting turn bias for calm wandering

        self.brain = self._build_brain()

    # ---- brain wiring -----------------------------------------------
    def _build_brain(self):
        b = Brain()

        # --- sensors (raw analog readings, tuned by color) ---
        b.add("S_L_friend", is_sensor=True)
        b.add("S_R_friend", is_sensor=True)
        b.add("S_L_danger", is_sensor=True)
        b.add("S_R_danger", is_sensor=True)

        # --- "I recognise my friend" threshold devices ---
        # Fires only once the tuned friend-sensor is strong enough.
        b.add("RECOG_L", threshold=0.15)
        b.add("RECOG_R", threshold=0.15)
        b.connect("S_L_friend", "RECOG_L", 1.0)
        b.connect("S_R_friend", "RECOG_R", 1.0)

        # --- APPROACH: fires when either recognizer fires. This is the
        # decision node whose threshold you can tweak live with [ and ]. ---
        b.add("APPROACH", threshold=2.0)
        b.connect("RECOG_L", "APPROACH", 1.0)
        b.connect("RECOG_R", "APPROACH", 1.0)

        # --- MEMORY: reciprocal self-sustaining pair, "rings forever" ---
        b.add("MEM_A", threshold=1.0)
        b.add("MEM_B", threshold=1.0)
        b.connect("S_L_danger", "MEM_A", 1.0)
        b.connect("S_R_danger", "MEM_A", 1.0)
        b.connect("MEM_B", "MEM_A", 1.0)   # keeps A alive
        b.connect("MEM_A", "MEM_B", 1.0)   # keeps B alive -> the "loop"

        # --- Memory makes the vehicle more cautious about approaching:
        # an inhibitory wire raises the effective bar for APPROACH. ---
        b.connect("MEM_A", "APPROACH", -0.5)

        return b

    # ---- perception ----------------------------------------------------
    def _sense(self, sources):
        """Each of the two sensors points slightly left/right of heading
        and only 'hears' sources whose color matches what it's tuned for.
        This is the physical basis of the vehicle's one and only NAME."""
        left_dir = self.angle - math.radians(SENSOR_SPREAD_DEG)
        right_dir = self.angle + math.radians(SENSOR_SPREAD_DEG)

        readings = {"friend_L": 0.0, "friend_R": 0.0,
                    "danger_L": 0.0, "danger_R": 0.0}

        for src in sources:
            dx, dy = src.x - self.x, src.y - self.y
            dist = math.hypot(dx, dy)
            if dist > SENSOR_RANGE or dist < 1e-6:
                continue
            intensity = 1.0 - (dist / SENSOR_RANGE)
            ang_to_src = math.atan2(dy, dx)

            def align(sensor_dir):
                d = abs((ang_to_src - sensor_dir + math.pi) % (2 * math.pi) - math.pi)
                return max(0.0, math.cos(d))  # 1.0 facing straight at it

            wL, wR = align(left_dir), align(right_dir)

            if src.color == FRIEND_COLOR:
                readings["friend_L"] += intensity * wL
                readings["friend_R"] += intensity * wR
            elif src.color == DANGER_COLOR:
                readings["danger_L"] += intensity * wL
                readings["danger_R"] += intensity * wR
            # NEUTRAL_COLOR sources produce no reading at all: no name, no reaction.

        return readings

    # ---- think + act -----------------------------------------------------
    def update(self, sources, dt):
        r = self._sense(sources)
        self.brain.set_sensor("S_L_friend", r["friend_L"])
        self.brain.set_sensor("S_R_friend", r["friend_R"])
        self.brain.set_sensor("S_L_danger", r["danger_L"])
        self.brain.set_sensor("S_R_danger", r["danger_R"])
        self.brain.step()

        approaching = self.brain.nodes["APPROACH"].output > 0.5

        base = 8.0  # calm wander speed (was 22 -- too jittery/fast)
        if approaching:
            # "love" wiring (crossed): a stimulus on one side speeds up
            # the OPPOSITE wheel, steering the vehicle toward the source.
            gain = 140.0
            target_left = base + gain * r["friend_R"]
            target_right = base + gain * r["friend_L"]
        else:
            # idle wander: a slow, smoothly drifting turn bias instead of
            # per-frame random jitter, so the vehicle glides and curves
            # gently rather than twitching side to side.
            self._wander_bias += random.uniform(-0.06, 0.06)
            self._wander_bias = max(-1.2, min(1.2, self._wander_bias))
            target_left = base - self._wander_bias * 2.0
            target_right = base + self._wander_bias * 2.0

        # smooth (low-pass filter) the wheel speeds every frame instead of
        # snapping to the target -- this removes the "twitchy idiot" motion
        # and gives calm acceleration/deceleration instead.
        smoothing = 0.08
        self.left_speed += (target_left - self.left_speed) * smoothing
        self.right_speed += (target_right - self.right_speed) * smoothing

        linear = (self.left_speed + self.right_speed) * 0.5 * dt
        angular = (self.right_speed - self.left_speed) / 34.0 * dt

        self.angle += angular
        self.x += math.cos(self.angle) * linear
        self.y += math.sin(self.angle) * linear

        self.x = max(20, min(WORLD_W - 20, self.x))
        self.y = max(20, min(HEIGHT - 20, self.y))

    @property
    def remembers_danger(self):
        return self.brain.nodes["MEM_A"].output > 0.5


# --------------------------------------------------------------------------
# DRAWING
# --------------------------------------------------------------------------
FONT = None
FONT_SM = None
FONT_TITLE = None


def draw_world(screen, sources, vehicles):
    screen.fill(BG, (0, 0, WORLD_W, HEIGHT))
    for gx in range(0, WORLD_W, 40):
        pygame.draw.line(screen, GRID, (gx, 0), (gx, HEIGHT))
    for gy in range(0, HEIGHT, 40):
        pygame.draw.line(screen, GRID, (0, gy), (WORLD_W, gy))

    for src in sources:
        pygame.draw.circle(screen, src.color, (int(src.x), int(src.y)), 10)
        pygame.draw.circle(screen, (0, 0, 0), (int(src.x), int(src.y)), 10, 1)

    # draw each vehicle (and their sensor rays)
    for vehicle in vehicles:
        # sensor rays
        left_dir = vehicle.angle - math.radians(SENSOR_SPREAD_DEG)
        right_dir = vehicle.angle + math.radians(SENSOR_SPREAD_DEG)
        for d, col in ((left_dir, (70, 90, 70)), (right_dir, (90, 70, 70))):
            ex = vehicle.x + math.cos(d) * SENSOR_RANGE
            ey = vehicle.y + math.sin(d) * SENSOR_RANGE
            pygame.draw.line(screen, col, (vehicle.x, vehicle.y), (ex, ey), 1)

        # vehicle body (triangle pointing heading)
        tip = (vehicle.x + math.cos(vehicle.angle) * VEHICLE_RADIUS * 1.6,
               vehicle.y + math.sin(vehicle.angle) * VEHICLE_RADIUS * 1.6)
        back1 = (vehicle.x + math.cos(vehicle.angle + 2.5) * VEHICLE_RADIUS,
                 vehicle.y + math.sin(vehicle.angle + 2.5) * VEHICLE_RADIUS)
        back2 = (vehicle.x + math.cos(vehicle.angle - 2.5) * VEHICLE_RADIUS,
                 vehicle.y + math.sin(vehicle.angle - 2.5) * VEHICLE_RADIUS)
        color = (240, 200, 90) if vehicle.brain.nodes["APPROACH"].output > 0.5 else (150, 160, 200)
        pygame.draw.polygon(screen, color, [tip, back1, back2])

        if vehicle.remembers_danger:
            pygame.draw.circle(screen, DANGER_COLOR, (int(vehicle.x), int(vehicle.y)),
                                VEHICLE_RADIUS + 8, 2)

    hud = [
        "1: place FRIEND (olive)   2: place DANGER (red)   3: place neutral (gray)",
        "click: drop source        [ / ]: APPROACH threshold        R: reset",
    ]
    for i, line in enumerate(hud):
        screen.blit(FONT_SM.render(line, True, DIM), (10, HEIGHT - 40 + i * 18))


NODE_LAYOUT = {
    "S_L_friend": (60, 70),   "S_R_friend": (60, 130),
    "S_L_danger": (60, 300),  "S_R_danger": (60, 360),
    "RECOG_L": (190, 70),     "RECOG_R": (190, 130),
    "APPROACH": (320, 100),
    "MEM_A": (190, 300),      "MEM_B": (320, 330),
}
NODE_LABEL = {
    "S_L_friend": "sensor L\n(friend)", "S_R_friend": "sensor R\n(friend)",
    "S_L_danger": "sensor L\n(red)",    "S_R_danger": "sensor R\n(red)",
    "RECOG_L": "RECOG L", "RECOG_R": "RECOG R",
    "APPROACH": "APPROACH",
    "MEM_A": "MEM A", "MEM_B": "MEM B",
}


def draw_brain_panel(screen, vehicle, selected_color):
    ox = WORLD_W
    screen.fill(PANEL_BG, (ox, 0, PANEL_W, HEIGHT))
    screen.blit(FONT_TITLE.render("Threshold Brain (live)", True, TEXT), (ox + 16, 14))

    b = vehicle.brain
    pts = {name: (ox + 40 + p[0] * 0.85, 60 + p[1]) for name, p in NODE_LAYOUT.items()}

    # connections
    for src, dst, w in b.connections:
        if src not in pts or dst not in pts:
            continue
        col = (90, 200, 110) if w > 0 else (210, 90, 90)
        pygame.draw.line(screen, col, pts[src], pts[dst], 2 if abs(w) >= 1 else 1)

    # nodes
    for name, node in b.nodes.items():
        if name not in pts:
            continue
        x, y = pts[name]
        active = node.output > 0.5
        base_col = (60, 65, 75)
        glow = (240, 200, 90) if active else base_col
        r = 22 if not node.is_sensor else 16
        pygame.draw.circle(screen, glow, (int(x), int(y)), r)
        pygame.draw.circle(screen, (10, 10, 12), (int(x), int(y)), r, 2)

        label = NODE_LABEL.get(name, name)
        for i, ln in enumerate(label.split("\n")):
            surf = FONT_SM.render(ln, True, TEXT)
            screen.blit(surf, (x - surf.get_width() / 2, y - r - 30 + i * 14))

        if not node.is_sensor:
            val = FONT_SM.render(f"th={node.threshold:.2f}", True, DIM)
            screen.blit(val, (x - val.get_width() / 2, y + r + 4))
        else:
            val = FONT_SM.render(f"{node.output:.2f}", True, DIM)
            screen.blit(val, (x - val.get_width() / 2, y + r + 4))

    legend_y = 470
    screen.blit(FONT.render("green line = excitatory", True, (90, 200, 110)), (ox + 16, legend_y))
    screen.blit(FONT.render("red line   = inhibitory", True, (210, 90, 90)), (ox + 16, legend_y + 22))
    screen.blit(FONT.render("glowing node = active (output=1)", True, (240, 200, 90)), (ox + 16, legend_y + 44))

    status_y = 550
    approach_val = b.raw_input_sum("APPROACH")
    lines = [
        f"APPROACH threshold: {b.nodes['APPROACH'].threshold:.2f}   (current input sum: {approach_val:.2f})",
        f"State: {'APPROACHING FRIEND' if b.nodes['APPROACH'].output > 0.5 else 'wandering / searching'}",
        f"Memory of danger latched: {'YES -- rings forever' if vehicle.remembers_danger else 'no'}",
        "",
        f"Now placing: {'FRIEND' if selected_color==FRIEND_COLOR else 'DANGER' if selected_color==DANGER_COLOR else 'neutral'}",
    ]
    for i, ln in enumerate(lines):
        screen.blit(FONT.render(ln, True, TEXT), (ox + 16, status_y + i * 22))


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def make_default_sources():
    return [
        Source(600, 150, FRIEND_COLOR),
        Source(150, 550, DANGER_COLOR),
        Source(400, 500, NEUTRAL_COLOR),
        Source(650, 600, NEUTRAL_COLOR),
        Source(120, 150, NEUTRAL_COLOR),
    ]


def main():
    global FONT, FONT_SM, FONT_TITLE
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Braitenberg Vehicle 5 -- Threshold Brain Simulation")
    clock = pygame.time.Clock()

    FONT = pygame.font.SysFont("consolas", 15)
    FONT_SM = pygame.font.SysFont("consolas", 12)
    FONT_TITLE = pygame.font.SysFont("consolas", 18, bold=True)

    sources = make_default_sources()
    vehicles = [Vehicle(WORLD_W / 2 - 60, HEIGHT / 2), Vehicle(WORLD_W / 2 + 60, HEIGHT / 2)]
    selected_idx = 0
    selected_color = FRIEND_COLOR

    running = True
    while running:
        dt = clock.tick(FPS) / 16.6667  # normalize roughly to "frames"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    selected_color = FRIEND_COLOR
                elif event.key == pygame.K_2:
                    selected_color = DANGER_COLOR
                elif event.key == pygame.K_3:
                    selected_color = NEUTRAL_COLOR
                elif event.key == pygame.K_r:
                    sources = make_default_sources()
                    vehicles = [Vehicle(WORLD_W / 2 - 60, HEIGHT / 2), Vehicle(WORLD_W / 2 + 60, HEIGHT / 2)]
                    selected_idx = 0
                elif event.key == pygame.K_LEFTBRACKET:
                    node = vehicles[selected_idx].brain.nodes["APPROACH"]
                    node.threshold = max(0.1, node.threshold - 0.1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    node = vehicles[selected_idx].brain.nodes["APPROACH"]
                    node.threshold = min(3.0, node.threshold + 0.1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < WORLD_W:
                    sources.append(Source(mx, my, selected_color))

        # update all vehicles
        for v in vehicles:
            v.update(sources, dt)

        draw_world(screen, sources, vehicles)
        # show the brain panel for the selected vehicle
        draw_brain_panel(screen, vehicles[selected_idx], selected_color)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
