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
    # Core supports: inner pair upgraded from the start; outer pair upgrade one per turn (see _attempt_core_support_upgrades).
    CORE_SUPPORT_INNER = [[13, 11], [14, 11]]
    CORE_SUPPORT_OUTER = [[12, 11], [15, 11]]
    CORE_SUPPORT_LOCATIONS = [[13, 11], [14, 11], [12, 11], [15, 11]]
    CORE_TURRET_LOCATIONS = [[11, 12], [16, 12], [11, 10], [16, 10]]
    # Subsets for destruction-triggered wall reactions (strict left vs strict right).
    CORE_LEFT_TURRETS = [[11, 12], [11, 10]]
    CORE_RIGHT_TURRETS = [[16, 12], [16, 10]]
    DAMAGE_REACTION_WALLS_LEFT = [[10, 11], [10, 12], [11, 11]]
    DAMAGE_REACTION_WALLS_RIGHT = [[17, 11], [17, 12], [16, 11]]
    # Opening turret box before the generic fill; order is placement priority.
    INITIAL_TURRET_LOCATIONS = [
        [11, 12],
        [16, 12],
        [11, 10],
        [16, 10],
    ]
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
            game_state.attempt_spawn(SUPPORT, self.CORE_SUPPORT_LOCATIONS)
        else:
            game_state.attempt_spawn(SUPPORT, self.CORE_SUPPORT_LOCATIONS)
            for location in self.INITIAL_TURRET_LOCATIONS:
                game_state.attempt_spawn(TURRET, location)
        self.build_reactive_defense(game_state)
        self.build_center_pressure_turret(game_state)
        self.build_deep_lane_counter_turret(game_state)
        core_missing = self._core_is_missing(game_state)
        self.build_damage_reaction_walls(game_state)
        # Turrets are never upgraded (only spawned/rebuilt). Supports only for upgrades.
        self._attempt_core_support_upgrades(game_state)

        if not core_missing:
            planned_turrets = self._planned_turret_locations(game_state)
            for location in self._turret_fill_locations(game_state):
                if self._violates_turret_spacing(location, planned_turrets, r=self.TURRET_SPACING_RADIUS):
                    continue
                placed = game_state.attempt_spawn(TURRET, location)
                if placed:
                    planned_turrets.add((location[0], location[1]))

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

    def _friendly_support_needs_upgrade(self, game_state, location):
        if not game_state.game_map.in_arena_bounds(location):
            return False
        units = game_state.game_map[location[0], location[1]]
        if not units:
            return False
        for unit in units:
            if unit.stationary and unit.player_index == 0 and unit.unit_type == SUPPORT:
                return not unit.upgraded
        return False

    def _attempt_core_support_upgrades(self, game_state):
        """
        Inner supports (13,11)/(14,11) are upgraded whenever SP allows.
        Outer supports upgrade one at a time: (12,11) from turn 1 onward until maxed;
        (15,11) from turn 2 onward once (12,11) no longer needs an upgrade.
        """
        game_state.attempt_upgrade(self.CORE_SUPPORT_INNER)
        tn = game_state.turn_number
        if tn >= 1 and self._friendly_support_needs_upgrade(game_state, [12, 11]):
            game_state.attempt_upgrade([[12, 11]])
        elif tn >= 2 and self._friendly_support_needs_upgrade(game_state, [15, 11]):
            game_state.attempt_upgrade([[15, 11]])

    def _core_is_missing(self, game_state):
        for loc in self.CORE_SUPPORT_LOCATIONS:
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
        reserved = set()
        for loc in self.CORE_SUPPORT_LOCATIONS:
            reserved.add((loc[0], loc[1]))
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

    def build_reactive_defense(self, game_state):
        """
        After a breach, add a reactive turret.
        For y in {12, 13}, try the breach tile first, then two tiles toward map center.
        For other y, try two tiles toward center; if a turret is already there, try the breach tile instead.
        """
        half = game_state.HALF_ARENA
        for location in self.scored_on_locations:
            if not isinstance(location, (list, tuple)) or len(location) < 2:
                continue
            bx, by = int(location[0]), int(location[1])
            dx = 2 if bx < half else -2
            dy = 2 if by < half else -2
            build_location = [bx + dx, by + dy]

            if by in (12, 13):
                on_breach = [bx, by]
                if (
                    game_state.game_map.in_arena_bounds(on_breach)
                    and on_breach[1] < half
                ):
                    if game_state.attempt_spawn(TURRET, on_breach):
                        continue

            if not game_state.game_map.in_arena_bounds(build_location):
                continue

            if by not in (12, 13) and self._is_friendly_turret_at(
                game_state, build_location
            ):
                on_breach = [bx, by]
                if (
                    game_state.game_map.in_arena_bounds(on_breach)
                    and on_breach[1] < half
                ):
                    game_state.attempt_spawn(TURRET, on_breach)
                continue

            game_state.attempt_spawn(TURRET, build_location)

    def _get_valid_scout_path_to_edge(self, game_state, start_location):
        """
        Return the engine path only if the scout can reach the real target edge.
        Otherwise the pathfinder may terminate inside a sealed pocket (self-destruct).
        All stationary structures (ours and theirs) block movement in pathfinding.
        """
        if not game_state.game_map.in_arena_bounds(start_location):
            return None
        if game_state.contains_stationary_unit(start_location):
            return None
        target_edge = game_state.get_target_edge(start_location)
        end_points = game_state.game_map.get_edge_locations(target_edge)
        edge_set = {tuple(p) for p in end_points}
        path = game_state.find_path_to_edge(start_location, target_edge)
        if path is None or len(path) < 2:
            return None
        if tuple(path[-1]) not in edge_set:
            return None
        for tile in path[1:]:
            if game_state.contains_stationary_unit(tile):
                return None
        return path

    def spawn_scouts(self, game_state):
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT)
            + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        deploy_locations = self.filter_trapped_scout_spawns(deploy_locations, game_state)
        if not deploy_locations:
            return

        ranked = self.rank_scout_spawn_locations(game_state, deploy_locations)
        if not ranked:
            return
        best = ranked[0][0]

        available_scouts = int(game_state.number_affordable(SCOUT))
        if available_scouts < 1:
            return

        # Use a small split when we have enough scouts to avoid being predictable.
        second = None
        if available_scouts >= self.SCOUT_SPLIT_MIN_COUNT:
            for loc, _ in ranked[1:]:
                if abs(loc[0] - best[0]) >= self.SCOUT_SPLIT_MIN_X_DISTANCE:
                    second = loc
                    break

        if second is None:
            game_state.attempt_spawn(SCOUT, best, available_scouts)
            return

        secondary_count = max(1, int(available_scouts * self.SCOUT_SPLIT_RATIO))
        primary_count = max(1, available_scouts - secondary_count)
        game_state.attempt_spawn(SCOUT, best, primary_count)
        game_state.attempt_spawn(SCOUT, second, secondary_count)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        Prefer the spawn whose path to the enemy edge estimates lowest turret damage.
        """
        ranked = self.rank_scout_spawn_locations(game_state, location_options)
        if ranked:
            return ranked[0][0]
        for loc in location_options:
            if self._get_valid_scout_path_to_edge(game_state, loc) is not None:
                return loc
        return location_options[0] if location_options else None

    def rank_scout_spawn_locations(self, game_state, location_options):
        """
        Returns list of (location, estimated_damage) sorted by lowest damage.
        Drops spawns that cannot path to the real enemy edge (would self-destruct in a pocket).
        """
        ranked = []
        turret_damage = gamelib.GameUnit(TURRET, game_state.config).damage_i
        for location in location_options:
            path = self._get_valid_scout_path_to_edge(game_state, location)
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
