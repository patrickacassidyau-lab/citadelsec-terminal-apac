import gamelib
import random
from sys import maxsize
import json


"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips:

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical
  board states. Though, we recommended making a copy of the map to preserve
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    SUPPORT_LOCATIONS = [[13, 12], [14, 12]]
    CORE_TURRET_LOCATIONS = [[11, 11], [12, 12], [13, 13], [14, 13], [15, 12], [16, 11]]
    # Subsets for destruction-triggered wall reactions (strict left vs strict right).
    CORE_LEFT_TURRETS = [[11, 11], [12, 12], [13, 13]]
    CORE_RIGHT_TURRETS = [[14, 13], [15, 12], [16, 11]]
    DAMAGE_REACTION_WALLS_LEFT = [[10, 11], [11, 12], [12, 13]]
    # Mirror across x + x' = 27 (arena x in [0, 27]).
    DAMAGE_REACTION_WALLS_RIGHT = [[17, 11], [16, 12], [15, 13]]
    # Opening turret wedge before the generic fill; order is placement priority.
    INITIAL_TURRET_LOCATIONS = [
        [0, 13],
        [27, 13],
        [11, 11],
        [12, 12],
        [13, 13],
        [14, 13],
        [15, 12],
        [16, 11],
    ]
    # Corner-entry wall funnels at [0,13] and [27,13].
    # Forces scouts entering the corners to travel 6+ extra tiles under turret fire.
    # Each funnel is a 3-tile wall arm extending inward from the corner entry.
    FUNNEL_WALL_LEFT  = [[1, 13], [2, 13], [3, 12]]   # left corner [0,13] funnel
    FUNNEL_WALL_RIGHT = [[26, 13], [25, 13], [24, 12]] # right corner [27,13] funnel

    # Prevent turret clumping during the fill phase.
    # r=1 uses Chebyshev distance (blocks adjacent + diagonal adjacency).
    TURRET_SPACING_RADIUS = 1
    SCOUT_SPLIT_MIN_COUNT = 10
    SCOUT_SPLIT_RATIO = 0.2
    SCOUT_SPLIT_MIN_X_DISTANCE = 8
    CENTER_PRESSURE_Y_MIN = 14
    CENTER_PRESSURE_Y_MAX = 21
    CENTER_PRESSURE_DURATION_TURNS = 1
    DEEP_LANE_Y_MIN = 22
    DEEP_LANE_Y_MAX = 24
    DEEP_LANE_DURATION_TURNS = 1

    # --- Simulator constants ---
    # Number of attack scenarios simulated per turn to pick the optimal spawn.
    SIM_SCENARIOS = 120
    # Weight recharge-rate efficiency vs raw damage. Higher = prefer faster recharge.
    # Objective: maximise MP/SP recharge rate, not just raw damage (Smite principle).
    SIM_RECHARGE_WEIGHT = 1.5

    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """
        Read in config and perform any initial setup here
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        self.scored_on_locations = []
        self.enemy_spawn_locations = []
        self.center_pressure_side = None
        self.center_pressure_until_turn = -1
        self.deep_lane_place_side = None
        self.deep_lane_until_turn = -1
        # Simulator: track last N spawn locations to avoid predictable patterns.
        # Stores (turn, location) tuples so we can rotate across the spawn pool.
        self._spawn_history = []          # list of [x, y] used historically
        self._last_sim_result = None      # cached best location from last simulation

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  # Comment or remove this line to enable warnings.
        self._update_center_pressure_state(game_state.turn_number)
        self._update_deep_lane_state(game_state.turn_number)

        self.scout_support_turret_turn(game_state)

        game_state.submit_turn()

    def scout_support_turret_turn(self, game_state):
        core_missing_before = self._core_is_missing(game_state)
        # When rebuilding core, place turrets first then supports so SP goes to defense anchors first.
        if core_missing_before:
            for location in self.INITIAL_TURRET_LOCATIONS:
                game_state.attempt_spawn(TURRET, location)
            game_state.attempt_spawn(SUPPORT, self.SUPPORT_LOCATIONS)
        else:
            game_state.attempt_spawn(SUPPORT, self.SUPPORT_LOCATIONS)
            for location in self.INITIAL_TURRET_LOCATIONS:
                game_state.attempt_spawn(TURRET, location)
        self.build_reactive_defense(game_state)
        self.build_center_pressure_turret(game_state)
        self.build_deep_lane_counter_turret(game_state)
        core_missing = self._core_is_missing(game_state)
        self.build_damage_reaction_walls(game_state)
        self.build_corner_funnels(game_state)
        # Turrets are never upgraded (only spawned/rebuilt). Supports only for upgrades.
        game_state.attempt_upgrade(self.SUPPORT_LOCATIONS)

        if not core_missing:
            planned_turrets = self._planned_turret_locations(game_state)
            for location in self._turret_fill_locations(game_state):
                if self._violates_turret_spacing(location, planned_turrets, r=self.TURRET_SPACING_RADIUS):
                    continue
                placed = game_state.attempt_spawn(TURRET, location)
                if placed:
                    planned_turrets.add((location[0], location[1]))

        # Place early-game interceptors to counter opponent scout spam
        # (especially turn-0 [13,0]/[14,0] hard-coded rushes).
        self.spawn_early_interceptors(game_state)
        self.spawn_scouts(game_state)
        self.enemy_spawn_locations = []

    def _friendly_structure_at(self, game_state, location, unit_type):
        if not game_state.game_map.in_arena_bounds(location):
            return False
        units = game_state.game_map[location[0], location[1]]
        if not units:
            return False
        for unit in units:
            if unit.stationary and unit.player_index == 0 and unit.unit_type == unit_type:
                return True
        return False

    def _core_is_missing(self, game_state):
        for loc in self.SUPPORT_LOCATIONS:
            if not self._friendly_structure_at(game_state, loc, SUPPORT):
                return True
        for loc in self.CORE_TURRET_LOCATIONS:
            if not self._friendly_structure_at(game_state, loc, TURRET):
                return True
        return False

    def _planned_turret_locations(self, game_state):
        """
        A set of friendly turret coordinates currently on the map plus our opening formation.
        Used to avoid placing fill turrets too close together.
        """
        planned = set()

        # Existing friendly turrets.
        for location in game_state.game_map:
            if self._is_friendly_turret_at(game_state, location):
                planned.add((location[0], location[1]))

        # Opening formation turrets (even if not yet placed this turn).
        for loc in self.INITIAL_TURRET_LOCATIONS:
            planned.add((loc[0], loc[1]))

        return planned

    def _is_friendly_turret_at(self, game_state, location):
        if not game_state.game_map.in_arena_bounds(location):
            return False
        units = game_state.game_map[location[0], location[1]]
        if not units:
            return False
        for unit in units:
            if unit.stationary and unit.player_index == 0 and unit.unit_type == TURRET:
                return True
        return False

    def _any_core_turret_present(self, game_state):
        return any(self._friendly_structure_at(game_state, loc, TURRET) for loc in self.CORE_TURRET_LOCATIONS)

    def _left_core_turret_destroyed(self, game_state):
        return any(not self._friendly_structure_at(game_state, loc, TURRET) for loc in self.CORE_LEFT_TURRETS)

    def _right_core_turret_destroyed(self, game_state):
        return any(not self._friendly_structure_at(game_state, loc, TURRET) for loc in self.CORE_RIGHT_TURRETS)

    def build_damage_reaction_walls(self, game_state):
        """
        If any left/right core turret slot has no turret (destroyed), try to build that
        side's emergency wall line. Skips until at least one core turret exists anywhere
        so opening turns before the wedge is placed do not spam walls. Runs before
        support upgrades so walls win on SP priority.
        """
        if not self._any_core_turret_present(game_state):
            return
        if self._left_core_turret_destroyed(game_state):
            for loc in self.DAMAGE_REACTION_WALLS_LEFT:
                game_state.attempt_spawn(WALL, loc)
        if self._right_core_turret_destroyed(game_state):
            for loc in self.DAMAGE_REACTION_WALLS_RIGHT:
                game_state.attempt_spawn(WALL, loc)

    def build_corner_funnels(self, game_state):
        """
        Build wall funnels at corner entries [0,13] and [27,13].
        Forces scouts entering those corners to travel 6+ extra tiles under turret fire
        before reaching our interior, significantly increasing damage dealt per scout.
        Only placed when SP is available after core structures.
        """
        for loc in self.FUNNEL_WALL_LEFT:
            game_state.attempt_spawn(WALL, loc)
        for loc in self.FUNNEL_WALL_RIGHT:
            game_state.attempt_spawn(WALL, loc)

    def spawn_early_interceptors(self, game_state):
        """
        Counter opponent scout-spam, especially hard-coded turn-0 spawns from [13,0]/[14,0].
        On turn 0-2, place 1 interceptor at a position that intercepts the central bottom
        corridor before we've built up enough turrets to stop a wave.
        Uses remaining MP only (interceptor costs 1 MP); won't fire if MP is being saved
        for a scout wave.

        Interceptors placed at [13,1] and [14,1] cover the [13,0] and [14,0] spawn lanes
        with minimal MP spend (1 MP each).
        """
        turn = game_state.turn_number
        # Only deploy early-game interceptors on turns 0-4 or whenever enemy scout-spam
        # is detected from the common central spawn tiles.
        enemy_central_scouts = sum(
            1 for loc in self.enemy_spawn_locations
            if isinstance(loc, (list, tuple)) and len(loc) >= 2
            and 12 <= loc[0] <= 15 and loc[1] >= 24
        )
        deploy_intercept = (turn <= 4) or (enemy_central_scouts >= 3)
        if not deploy_intercept:
            return

        available_mp = game_state.get_resource(MP)
        # Reserve MP for scouts unless early game. Use at most 2 MP for interceptors.
        # On turns 0-2 always place them regardless; later only if MP is plentiful.
        if turn > 2 and available_mp < 8:
            return

        # Place interceptors to cover [13,0]/[14,0] approach path.
        # [13,1] and [14,1] are in our territory and intercept scouts running that lane.
        intercept_spots = [[13, 1], [14, 1]]
        for spot in intercept_spots:
            if game_state.can_spawn(INTERCEPTOR, spot, 1):
                game_state.attempt_spawn(INTERCEPTOR, spot, 1)
                available_mp -= 1
                if available_mp < 1:
                    break

    def _nearby_cells(self, location, r=1):
        x0, y0 = location
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                yield [x0 + dx, y0 + dy]

    def _violates_turret_spacing(self, location, planned_turrets, r=1):
        for n in self._nearby_cells(location, r=r):
            if (n[0], n[1]) in planned_turrets:
                return True
        return False

    def _turret_reserved_cells(self):
        reserved = {(13, 12), (14, 12)}
        for loc in self.INITIAL_TURRET_LOCATIONS:
            reserved.add((loc[0], loc[1]))
        return reserved

    def _turret_fill_locations(self, game_state):
        exclude = self._turret_reserved_cells()
        gmap = game_state.game_map
        half = game_state.HALF_ARENA
        candidates = []
        for y in range(0, half):
            for x in range(0, game_state.ARENA_SIZE):
                loc = [x, y]
                if (x, y) in exclude:
                    continue
                if gmap.in_arena_bounds(loc):
                    candidates.append(loc)
        # Fill front rows first. Within each row, build from both sides inward and
        # bias toward the most recently breached side.
        candidates.sort(key=lambda L: self._turret_fill_priority(L, game_state))
        return candidates

    def _recent_hit_side(self):
        """
        Returns 'left' or 'right' from the most recent enemy breach, or None.
        """
        if not self.scored_on_locations:
            return None
        x, _ = self.scored_on_locations[-1]
        return "left" if x <= 13 else "right"

    def _update_center_pressure_state(self, turn_number):
        """
        Detect enemy mobile spawns in y-band [14, 21] and temporarily bias fill
        to the center-front of the pressured side.
        """
        pressure_spawns = []
        for loc in self.enemy_spawn_locations:
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue
            if self.CENTER_PRESSURE_Y_MIN <= loc[1] <= self.CENTER_PRESSURE_Y_MAX:
                pressure_spawns.append(loc)
        if not pressure_spawns:
            return

        left = sum(1 for x, _ in pressure_spawns if x <= 13)
        right = len(pressure_spawns) - left
        if left == right:
            avg_x = sum(x for x, _ in pressure_spawns) / len(pressure_spawns)
            self.center_pressure_side = "left" if avg_x <= 13.5 else "right"
        else:
            self.center_pressure_side = "left" if left > right else "right"
        self.center_pressure_until_turn = turn_number + self.CENTER_PRESSURE_DURATION_TURNS

    def _update_deep_lane_state(self, turn_number):
        """
        Enemy mobile spawns in y-band [22, 24]: next turn place one turret on the
        opposite side center (mirror of center-pressure placement).
        """
        deep_spawns = []
        for loc in self.enemy_spawn_locations:
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue
            if self.DEEP_LANE_Y_MIN <= loc[1] <= self.DEEP_LANE_Y_MAX:
                deep_spawns.append(loc)
        if not deep_spawns:
            return

        left = sum(1 for x, _ in deep_spawns if x <= 13)
        right = len(deep_spawns) - left
        if left == right:
            avg_x = sum(x for x, _ in deep_spawns) / len(deep_spawns)
            attack_side = "left" if avg_x <= 13.5 else "right"
        else:
            attack_side = "left" if left > right else "right"
        self.deep_lane_place_side = "right" if attack_side == "left" else "left"
        self.deep_lane_until_turn = turn_number + self.DEEP_LANE_DURATION_TURNS

    def _center_pressure_active(self, turn_number):
        return self.center_pressure_side is not None and turn_number <= self.center_pressure_until_turn

    def _deep_lane_active(self, turn_number):
        return self.deep_lane_place_side is not None and turn_number <= self.deep_lane_until_turn

    def _friendly_turret_positions(self, game_state):
        turrets = []
        for location in game_state.game_map:
            units = game_state.game_map[location[0], location[1]]
            if units:
                if any(unit.stationary and unit.player_index == 0 and unit.unit_type == TURRET for unit in units):
                    turrets.append([location[0], location[1]])
        return turrets

    def _best_side_gap_target(self, game_state, side):
        """
        Select one central gap tile on the given side ('left' or 'right') and
        maximize spacing from existing friendly turrets.
        """
        if side == "left":
            candidates = [[10, 12], [9, 12], [11, 12], [10, 11], [9, 11]]
        else:
            candidates = [[17, 12], [18, 12], [16, 12], [17, 11], [18, 11]]

        existing_turrets = self._friendly_turret_positions(game_state)
        viable = []
        for loc in candidates:
            if not game_state.game_map.in_arena_bounds(loc):
                continue
            if game_state.contains_stationary_unit(loc):
                continue
            if loc[1] >= game_state.HALF_ARENA:
                continue

            if not existing_turrets:
                min_chebyshev = 99
            else:
                min_chebyshev = min(
                    max(abs(loc[0] - t[0]), abs(loc[1] - t[1])) for t in existing_turrets
                )

            viable.append((loc, min_chebyshev, abs(loc[0] - 13.5)))

        if not viable:
            return None
        viable.sort(key=lambda item: (-item[1], item[2]))
        return viable[0][0]

    def build_center_pressure_turret(self, game_state):
        """
        On center-pressure signal, place exactly one central turret on the pressured side.
        Placement is chosen to be spaced relative to our existing turret network.
        """
        if not self._center_pressure_active(game_state.turn_number):
            return
        target = self._best_side_gap_target(game_state, self.center_pressure_side)
        if target is not None:
            game_state.attempt_spawn(TURRET, target, 1)

    def build_deep_lane_counter_turret(self, game_state):
        """
        After enemy deep-lane spawns (y in [22,24]), place one turret on the
        opposite side center gap (same placement logic as center pressure).
        """
        if not self._deep_lane_active(game_state.turn_number):
            return
        target = self._best_side_gap_target(game_state, self.deep_lane_place_side)
        if target is not None:
            game_state.attempt_spawn(TURRET, target, 1)

    def _turret_fill_priority(self, location, game_state):
        x, y = location
        arena_max_x = game_state.ARENA_SIZE - 1
        edge_distance = min(x, arena_max_x - x)
        hit_side = self._recent_hit_side()
        is_left = x <= 13

        if hit_side is None:
            side_bias = 0
        elif (hit_side == "left" and is_left) or (hit_side == "right" and not is_left):
            side_bias = 0
        else:
            side_bias = 1

        # Alternate sides within the same row/ring for more even parallel flank fill.
        # Lower value is preferred.
        side_alternation = 0 if is_left else 1

        return (-y, edge_distance, side_bias, side_alternation, x)

    def _reactive_turret_two_steps_toward_center(self, bx, by, half):
        dx = 2 if bx < half else -2
        dy = 2 if by < half else -2
        return [bx + dx, by + dy]

    def _reactive_turret_one_step_toward_center_bottom(self, bx, by, half):
        """One step toward map center without leaving bottom territory (y < half)."""
        dx = 1 if bx < half else -1
        dy = 1 if by + 1 < half else 0
        return [bx + dx, by + dy]

    def build_reactive_defense(self, game_state):
        """
        After a breach, place a reactive turret.
        - Breach 0 <= y <= 8: two tiles toward center (x and y), as before.
        - Breach 9 <= y <= 13: try breach tile, else one step toward center (stays in our half).
        """
        half = game_state.HALF_ARENA
        for location in self.scored_on_locations:
            if not isinstance(location, (list, tuple)) or len(location) < 2:
                continue
            bx, by = int(location[0]), int(location[1])
            if 0 <= by <= 8:
                build_location = self._reactive_turret_two_steps_toward_center(bx, by, half)
                if not game_state.game_map.in_arena_bounds(build_location):
                    continue
                game_state.attempt_spawn(TURRET, build_location)
            elif 9 <= by <= 13:
                on_breach = [bx, by]
                one_in = self._reactive_turret_one_step_toward_center_bottom(bx, by, half)
                for candidate in (on_breach, one_in):
                    if not game_state.game_map.in_arena_bounds(candidate):
                        continue
                    if candidate[1] >= half:
                        continue
                    if game_state.attempt_spawn(TURRET, candidate):
                        break
            else:
                continue

    def spawn_scouts(self, game_state):
        """
        Simulator-first spawn selection.
        Simulates SIM_SCENARIOS attack scenarios each turn, scoring each by a
        composite objective that maximises the MP/SP recharge rate (Smite principle),
        not just raw damage.

        Score = expected_damage * (1 + SIM_RECHARGE_WEIGHT / (path_length + 1))
        This rewards shorter paths (faster recharge) and punishes expensive defences.

        Also randomises spawn location across viable candidates to avoid predictable
        patterns — prevents opponents hard-coding interceptors vs our turn-0 spawn.
        """
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT)
            + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        deploy_locations = self.filter_trapped_scout_spawns(deploy_locations, game_state)
        if not deploy_locations:
            return

        available_scouts = int(game_state.number_affordable(SCOUT))
        if available_scouts < 1:
            return

        # Run simulator to pick best spawn location.
        best, second = self._simulate_attack_scenarios(game_state, deploy_locations, available_scouts)
        if best is None:
            return

        # Use a small split when we have enough scouts and a viable second location.
        if second is not None and available_scouts >= self.SCOUT_SPLIT_MIN_COUNT:
            secondary_count = max(1, int(available_scouts * self.SCOUT_SPLIT_RATIO))
            primary_count = max(1, available_scouts - secondary_count)
            game_state.attempt_spawn(SCOUT, best, primary_count)
            game_state.attempt_spawn(SCOUT, second, secondary_count)
        else:
            game_state.attempt_spawn(SCOUT, best, available_scouts)

        # Track spawn history to detect if we're being too predictable.
        self._spawn_history.append(best[:])
        if len(self._spawn_history) > 10:
            self._spawn_history.pop(0)

    def _simulate_attack_scenarios(self, game_state, deploy_locations, num_scouts):
        """
        Core simulator: evaluates up to SIM_SCENARIOS spawn configurations and scores
        each by the Smite objective — maximise recharge rate efficiency, not raw damage.

        Objective per scenario:
          score = expected_damage * (1 + RECHARGE_WEIGHT / (path_length + 1))
            - expected_damage: sum of (turret_damage * attackers) along path
            - path_length: number of tiles in path (shorter = faster recharge cycle)
            - RECHARGE_WEIGHT: tunes how much we value speed over raw damage

        Additionally applies a diversity penalty to locations we've recently spawned at,
        to avoid being predictable to opponents who hard-code counters.

        Returns (best_location, second_best_location_or_None).
        """
        turret_damage = gamelib.GameUnit(TURRET, game_state.config).damage_i
        # Paths are expensive; cache per location.
        path_cache = {}
        scored = []

        # Determine diversity penalty tiles (recently used spawns).
        recent_set = set(tuple(s) for s in self._spawn_history[-5:])

        # Sample scenarios: try each location, plus random jitter up to SIM_SCENARIOS total.
        candidates = list(deploy_locations)
        # If fewer candidates than SIM_SCENARIOS, re-use with small perturbations isn't
        # applicable (edge tiles are fixed), so just evaluate all candidates.
        num_to_eval = min(len(candidates), self.SIM_SCENARIOS)
        # Shuffle to avoid always evaluating same ordering when we truncate.
        random.shuffle(candidates)
        eval_candidates = candidates[:num_to_eval]

        for location in eval_candidates:
            loc_key = (location[0], location[1])
            if loc_key not in path_cache:
                path = game_state.find_path_to_edge(location)
                path_cache[loc_key] = path
            path = path_cache[loc_key]
            if path is None:
                continue

            path_length = len(path)
            # Estimate damage taken along path (lower = better spawn tile for us).
            damage_taken = sum(
                len(game_state.get_attackers(tile, 0)) * turret_damage
                for tile in path
            )
            # Estimate damage dealt: scouts hitting structures along path.
            # Approximated as: scouts × tiles_that_have_enemy_structures_in_range.
            damage_dealt = self._estimate_damage_dealt(game_state, path, num_scouts)

            # Composite score: reward dealt damage, reward shorter paths (fast recharge),
            # penalise taken damage (costly defence rebuild from SP).
            recharge_bonus = self.SIM_RECHARGE_WEIGHT / (path_length + 1)
            raw_score = (damage_dealt + 1) * (1.0 + recharge_bonus) - damage_taken * 0.5

            # Diversity penalty: slightly discourage recently used spawn tiles so we
            # don't become predictable across turns.
            if loc_key in recent_set:
                raw_score *= 0.85

            scored.append((location, raw_score))

        if not scored:
            return None, None

        scored.sort(key=lambda item: -item[1])
        best = scored[0][0]
        # Find second-best that is far enough away for a useful split.
        second = None
        for loc, _ in scored[1:]:
            if abs(loc[0] - best[0]) >= self.SCOUT_SPLIT_MIN_X_DISTANCE:
                second = loc
                break
        return best, second

    def _estimate_damage_dealt(self, game_state, path, num_scouts):
        """
        Estimate how much damage our scouts deal following this path.
        Scouts attack the nearest enemy unit in range each step (range 3.5 for scouts).
        We approximate by counting enemy stationary units within scout attack range
        of any tile on the path, then multiply by scout attack damage.
        Scouts have damage_i (damage to structures) from config.
        """
        scout_damage = gamelib.GameUnit(SCOUT, game_state.config).damage_i
        scout_range_sq = 3.5 ** 2  # range^2 for distance comparison
        counted = set()
        for tile in path:
            tx, ty = tile[0], tile[1]
            # Check tiles within range for enemy stationary units.
            for r in range(-4, 5):
                for c in range(-4, 5):
                    ex, ey = tx + c, ty + r
                    if (ex, ey) in counted:
                        continue
                    if (c * c + r * r) > scout_range_sq:
                        continue
                    eloc = [ex, ey]
                    if not game_state.game_map.in_arena_bounds(eloc):
                        continue
                    units = game_state.game_map[ex, ey]
                    if units:
                        for unit in units:
                            if unit.stationary and unit.player_index == 1:
                                counted.add((ex, ey))
                                break
        return len(counted) * scout_damage * num_scouts



    def least_damage_spawn_location(self, game_state, location_options):
        """
        Prefer the spawn whose path to the enemy edge estimates lowest turret damage.
        """
        ranked = self.rank_scout_spawn_locations(game_state, location_options)
        if not ranked:
            return location_options[0]
        return ranked[0][0]

    def rank_scout_spawn_locations(self, game_state, location_options):
        """
        Returns list of (location, estimated_damage) sorted by lowest damage.
        """
        ranked = []
        turret_damage = gamelib.GameUnit(TURRET, game_state.config).damage_i
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            if path is None:
                continue
            damage = 0
            for path_location in path:
                damage += len(game_state.get_attackers(path_location, 0)) * turret_damage
            ranked.append((location, damage))
        ranked.sort(key=lambda item: item[1])
        return ranked

    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def _is_own_stationary_at(self, game_state, location):
        if not game_state.game_map.in_arena_bounds(location):
            return False
        units = game_state.game_map[location[0], location[1]]
        if not units:
            return False
        return any(unit.stationary and unit.player_index == 0 for unit in units)

    def _is_trapped_scout_spawn(self, location, game_state):
        """
        Reject spawn tiles where scouts can get immediately pinned:
        our stationary unit directly above AND our stationary unit directly left or right.
        """
        x, y = location
        above = [x, y + 1]
        left = [x - 1, y]
        right = [x + 1, y]
        return self._is_own_stationary_at(game_state, above) and (
            self._is_own_stationary_at(game_state, left)
            or self._is_own_stationary_at(game_state, right)
        )

    def filter_trapped_scout_spawns(self, locations, game_state):
        return [loc for loc in locations if not self._is_trapped_scout_spawn(loc, game_state)]

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        try:
            state = json.loads(turn_string)
            events = state.get("events", {}) if isinstance(state, dict) else {}
            breaches = events.get("breach", []) if isinstance(events, dict) else []
            spawns = events.get("spawn", []) if isinstance(events, dict) else []
        except Exception:
            return
        for breach in breaches:
            if not isinstance(breach, (list, tuple)) or len(breach) < 5:
                continue
            location = breach[0]
            if not isinstance(location, (list, tuple)) or len(location) < 2:
                continue
            unit_owner_self = True if breach[4] == 1 else False
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))
        for spawn in spawns:
            if not isinstance(spawn, (list, tuple)) or len(spawn) < 4:
                continue
            location = spawn[0]
            if not isinstance(location, (list, tuple)) or len(location) < 2:
                continue
            unit_type = spawn[1]
            owner = spawn[3]
            if owner == 2 and unit_type in (SCOUT, DEMOLISHER, INTERCEPTOR):
                self.enemy_spawn_locations.append(location)


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
