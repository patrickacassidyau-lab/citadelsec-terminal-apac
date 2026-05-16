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
        [1, 12],
        [26, 12],
    ]
    # Prevent turret clumping during the fill phase.
    # r=1 uses Chebyshev distance (blocks adjacent + diagonal adjacency).
    TURRET_SPACING_RADIUS = 1

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
        global SUPPORT, TURRET, SCOUT, MP, SP
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        MP = 1
        SP = 0
        self.scored_on_locations = []

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

        self.scout_support_turret_turn(game_state)

        game_state.submit_turn()

    def scout_support_turret_turn(self, game_state):
        game_state.attempt_spawn(SUPPORT, self.SUPPORT_LOCATIONS)
        self.build_reactive_defense(game_state)
        for location in self.INITIAL_TURRET_LOCATIONS:
            game_state.attempt_spawn(TURRET, location)
        # Upgrade supports before spending remaining SP on the turret flood fill.
        game_state.attempt_upgrade(self.SUPPORT_LOCATIONS)

        planned_turrets = self._planned_turret_locations(game_state)
        for location in self._turret_fill_locations(game_state):
            if self._violates_turret_spacing(location, planned_turrets, r=self.TURRET_SPACING_RADIUS):
                continue
            placed = game_state.attempt_spawn(TURRET, location)
            if placed:
                planned_turrets.add((location[0], location[1]))

        for location in self.INITIAL_TURRET_LOCATIONS:
            game_state.attempt_upgrade(location)
        for location in self._turret_fill_locations(game_state):
            game_state.attempt_upgrade(location)
        self.spawn_scouts(game_state)

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
        unit = units[0]
        return unit.stationary and unit.player_index == 0 and unit.unit_type == TURRET

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
        After a breach, add a turret one tile toward our side from the breach tile.
        """
        for location in self.scored_on_locations:
            build_location = [location[0], location[1] + 1]
            game_state.attempt_spawn(TURRET, build_location)

    def spawn_scouts(self, game_state):
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT)
            + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        if not deploy_locations:
            return
        center_candidates = [loc for loc in deploy_locations if loc in [[13, 0], [14, 0]]]
        if len(center_candidates) >= 2:
            best = self.least_damage_spawn_location(game_state, center_candidates)
        elif len(center_candidates) == 1:
            best = center_candidates[0]
        elif len(deploy_locations) >= 2:
            best = self.least_damage_spawn_location(game_state, deploy_locations)
        else:
            best = deploy_locations[0]
        game_state.attempt_spawn(SCOUT, best, 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        Prefer the spawn whose path to the enemy edge estimates lowest turret damage.
        """
        damages = []
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            if path is None:
                damages.append(float('inf'))
                continue
            damage = 0
            for path_location in path:
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(
                    TURRET, game_state.config
                ).damage_i
            damages.append(damage)
        return location_options[damages.index(min(damages))]

    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
