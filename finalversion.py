import gamelib
import random
import math
import warnings
from sys import maxsize
import json

"""
Reverse-engineered from 6 Terminal replay files.
Implements the exact strategy observed in every game (identical Turn 0 every time,
then adaptive repair + secondary build-out on subsequent turns).
"""


class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """
        Read in config and perform any initial setup here.
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER, REMOVE, UPGRADE
        FILTER     = config["unitInformation"][0]["shorthand"]  # FF
        ENCRYPTOR  = config["unitInformation"][1]["shorthand"]  # EF
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]  # DF
        PING       = config["unitInformation"][3]["shorthand"]  # PI
        EMP        = config["unitInformation"][4]["shorthand"]  # EI
        SCRAMBLER  = config["unitInformation"][5]["shorthand"]  # SI
        REMOVE     = config["unitInformation"][6]["shorthand"]  # RM
        UPGRADE    = config["unitInformation"][7]["shorthand"]  # UP

        # ---------------------------------------------------------------
        # CORE WALL  (placed AND upgraded on Turn 0)
        # 8 Destructors form the main defence line on rows y=12 and y=13.
        # ---------------------------------------------------------------
        self.core_destructors = [
            [12, 13], [11, 12],  # centre-left pair
            [15, 13], [16, 12],  # centre-right pair
            [4,  13], [3,  12],  # left-flank pair
            [23, 13], [24, 12],  # right-flank pair
        ]

        # 2 Encryptors placed AND UPGRADED on Turn 0.
        # Upgraded Encryptors have doubled shield range (12 tiles) and a
        # per-y-level bonus — critical for buffing Pings through the midfield.
        self.core_encryptors = [
            [12, 12],
            [15, 12],
        ]

        # ---------------------------------------------------------------
        # SECONDARY DESTRUCTORS – filled in priority order on later turns
        # using any SP left over after repairing the core wall.
        # ---------------------------------------------------------------
        self.secondary_destructors = [
            [0,  13], [1,  13],  # far-left edge
            [26, 13], [27, 13],  # far-right edge
            [19, 13], [20, 12],  # mid-right
            [8,  13], [7,  12],  # mid-left
            [10, 11], [6,  11],  # inner back-row (left)
            [17, 11], [21, 11],  # inner back-row (right)
        ]

        # ---------------------------------------------------------------
        # INNER ENCRYPTORS – placed + upgraded after secondary DFs are funded.
        # Slots between the two core EF positions for extra coverage.
        # ---------------------------------------------------------------
        self.inner_encryptors = [
            [13, 12],
            [14, 12],
        ]

        # ---------------------------------------------------------------
        # PING SPAWN LOCATIONS  –  alternate every turn
        # Even turns → [14, 0]  (scouts route through right half)
        # Odd  turns → [13, 0]  (scouts route through left half)
        # Consistent across ALL 6 replays.
        # ---------------------------------------------------------------
        self.ping_spawn_points = [[14, 0], [13, 0]]

    # ---------------------------------------------------------------
    # MAIN TURN HANDLER
    # ---------------------------------------------------------------
    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write(
            'Performing turn {} of your custom algo strategy'.format(
                game_state.turn_number))
        game_state.suppress_warnings(True)

        self.execute_strategy(game_state)
        game_state.submit_turn()

    # ---------------------------------------------------------------
    # STRATEGY
    # ---------------------------------------------------------------
    def execute_strategy(self, game_state):
        """
        Exact strategy derived from 6 replays.

        EVERY TURN
        ----------
        1. Build / repair core Destructors (8 positions on y=12/13).
        2. Build / upgrade core Encryptors at [12,12] and [15,12].
        3. Fill secondary Destructors with remaining SP (priority order).
        4. Build / upgrade inner Encryptors at [13,12] and [14,12].
        5. Spend ALL MP on Pings from the alternating spawn point.
        """
        turn = game_state.turn_number

        self._build_core_destructors(game_state)
        self._build_core_encryptors(game_state)
        self._fill_secondary_destructors(game_state)
        self._build_inner_encryptors(game_state)
        self._deploy_scouts(game_state, turn)

    # ---------------------------------------------------------------
    # STRUCTURE BUILDERS
    # ---------------------------------------------------------------

    def _build_core_destructors(self, game_state):
        """Place or repair the 8 core Destructors."""
        for pos in self.core_destructors:
            game_state.attempt_spawn(DESTRUCTOR, pos)

    def _build_core_encryptors(self, game_state):
        """
        Place core Encryptors at [12,12] and [15,12] and immediately upgrade
        them.  Upgrading doubles shield range to 12 tiles and adds the
        per-y bonus — both Encryptors are upgraded on the very first turn
        they are placed, as seen in every replay.
        """
        self._build_and_upgrade_supports(game_state, self.core_encryptors)

    def _fill_secondary_destructors(self, game_state):
        """
        Fill secondary Destructor positions in priority order using any
        remaining SP after core repairs.
        """
        for pos in self.secondary_destructors:
            if game_state.get_resource(0) < 3:  # DF costs 3 SP
                break
            game_state.attempt_spawn(DESTRUCTOR, pos)

    def _build_inner_encryptors(self, game_state):
        """
        Place and upgrade inner Encryptors at [13,12] and [14,12].
        Placed only after secondary DFs are funded, matching the observed
        priority order in the replays. Upgrades happen on the turn AFTER
        placement (one-turn lag observed across all replays).
        """
        self._build_and_upgrade_supports(game_state, self.inner_encryptors)

    def _build_and_upgrade_supports(self, game_state, positions):
        for pos in positions:
            if not game_state.contains_stationary_unit(pos):
                game_state.attempt_spawn(ENCRYPTOR, pos)
            if self._needs_upgrade(game_state, pos):
                game_state.attempt_upgrade(pos)

    def _needs_upgrade(self, game_state, pos):
        if not game_state.game_map.in_arena_bounds(pos):
            return False
        units = game_state.game_map[pos[0], pos[1]]
        if not units:
            return False
        for u in units:
            if u.stationary and u.player_index == 0 and u.unit_type == ENCRYPTOR:
                return not u.upgraded
        return False

    # ---------------------------------------------------------------
    # MOBILE UNIT DEPLOYMENT
    # ---------------------------------------------------------------

    def _deploy_scouts(self, game_state, turn):
        """
        Spend ALL available MP on Pings (scouts), alternating the spawn
        point every turn so scouts hit both sides of the board.

          Even turns → [14, 0]  (routes right → left through enemy half)
          Odd  turns → [13, 0]  (routes left → right through enemy half)
        """
        spawn = self.ping_spawn_points[turn % 2]
        mp = int(game_state.get_resource(1))
        if mp > 0:
            game_state.attempt_spawn(PING, spawn, mp)

    # ---------------------------------------------------------------
    # ACTION FRAME HANDLER
    # ---------------------------------------------------------------
    def on_action_frame(self, turn_string):
        pass


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
