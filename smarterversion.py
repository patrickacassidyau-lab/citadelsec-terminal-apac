import gamelib
import random
import math
import json
from sys import maxsize
from collections import defaultdict

# ─── game constants (filled in on_game_start) ─────────────────────────────────
WALL = SUPPORT = TURRET = SCOUT = DEMOLISHER = INTERCEPTOR = None
MP = SP = None

# ─── board geometry ───────────────────────────────────────────────────────────
BOARD_SIZE  = 28
MY_EDGES    = []
ENEMY_EDGES = []

# ─── tuning ───────────────────────────────────────────────────────────────────
PROBE_INTERVAL       = 3
MIN_ATTACK_MP        = 5
ALL_IN_THRESHOLD_HP  = 8
ALL_IN_TURN          = 35
SPLIT_ATTACK_MP      = 16   
SELF_DESTRUCT_SCORE  = 5    


# ══════════════════════════════════════════════════════════════════════════════
#  OPENING BOOK
# ══════════════════════════════════════════════════════════════════════════════
OPENING_BOOK = {
    0: [
        ("TURRET", [13, 14]), ("TURRET", [14, 14]),
        ("TURRET", [12, 14]), ("TURRET", [15, 14]),
        ("WALL",   [0,  13]), ("WALL",  [27, 13]),
        ("WALL",   [1,  13]), ("WALL",  [26, 13]),
        ("WALL",   [13, 13]), ("WALL",  [14, 13]),
    ],
    1: [
        ("TURRET", [13, 15]), ("TURRET", [14, 15]),
        ("TURRET", [11, 15]), ("TURRET", [16, 15]),
        ("WALL",   [2,  13]), ("WALL",  [25, 13]),
        ("WALL",   [3,  13]), ("WALL",  [24, 13]),
        ("WALL",   [4,  13]), ("WALL",  [23, 13]),
        ("WALL",   [5,  13]), ("WALL",  [22, 13]),
    ],
    2: [
        ("SUPPORT", [13, 16]), ("SUPPORT", [14, 16]),
        ("WALL",   [11, 14]), ("WALL",  [10, 15]),
        ("WALL",   [9,  16]), ("WALL",  [8,  17]),
        ("WALL",   [6,  13]), ("WALL",  [21, 13]),
        ("WALL",   [7,  13]), ("WALL",  [20, 13]),
        ("WALL",   [8,  13]), ("WALL",  [19, 13]),
    ],
    3: [
        ("TURRET", [10, 16]), ("TURRET", [9,  17]),
        ("TURRET", [17, 16]), ("TURRET", [18, 17]),
        ("WALL",   [9,  13]), ("WALL",  [18, 13]),
        ("WALL",   [10, 13]), ("WALL",  [17, 13]),
        ("WALL",   [11, 13]), ("WALL",  [16, 13]),
        ("WALL",   [12, 13]), ("WALL",  [15, 13]),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  OPPONENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class OpponentModel:
    def __init__(self):
        self.turn_data       = []
        self.breach_heatmap  = defaultdict(int)
        self.attack_heatmap  = defaultdict(int)
        self.sp_history      = []
        self.mp_history      = []
        self.last_known_sp   = 40.0
        self.last_known_mp   = 0.0
        self.evidence        = defaultdict(float)

    def update(self, game_state, new_breaches, enemy_spawn_locs, enemy_demo_spawns=0):
        t = game_state.turn_number
        try:
            cur_sp = game_state.get_resource(SP, 1)
            cur_mp = game_state.get_resource(MP, 1)
        except Exception:
            cur_sp = self.last_known_sp
            cur_mp = self.last_known_mp

        cur_sp = cur_sp if cur_sp is not None else self.last_known_sp
        cur_mp = cur_mp if cur_mp is not None else self.last_known_mp

        sp_delta = self.last_known_sp - cur_sp
        self.sp_history.append(cur_sp)
        self.mp_history.append(cur_mp)

        for loc in enemy_spawn_locs:
            self.attack_heatmap[tuple(loc)] += 1
        for loc in new_breaches:
            self.breach_heatmap[tuple(loc)] += 1

        self.turn_data.append({
            "turn": t, "sp_delta": sp_delta,
            "mp_spent": cur_mp, "spawn_count": len(enemy_spawn_locs),
            "breach_count": len(new_breaches),
        })
        self.last_known_sp = cur_sp
        self.last_known_mp = cur_mp
        self._update_evidence(t, sp_delta, cur_mp, enemy_spawn_locs, new_breaches, enemy_demo_spawns)

    def _update_evidence(self, turn, sp_delta, mp, spawn_locs, breaches, demo_spawns=0):
        e = self.evidence
        for k in list(e.keys()):
            e[k] *= 0.92

        if turn < 8 and mp > 8:
            e["rush"] += 0.3
        if turn < 5 and sp_delta < 5:
            e["rush"] += 0.2
        if mp > 15:
            e["emp_spam"] += 0.25
        if demo_spawns > 0 and turn > 3:
            e["emp_spam"] += 0.15
        if mp > 5 and len(spawn_locs) > 3:
            e["scout_flood"] += 0.2
        if len(breaches) > 0:
            e["scout_flood"] += 0.15
        if sp_delta > 12:
            e["attrition"] += 0.2
        if turn > 10 and mp < 8:
            e["attrition"] += 0.1
        if sp_delta > 8 and len(spawn_locs) == 0 and turn > 4:
            e["wall_lock"] += 0.25
        if sp_delta > 15 and turn > 6:
            e["wall_lock"] += 0.2
        total_b = sum(self.breach_heatmap.values())
        if total_b > 3 and len(self.breach_heatmap) <= 2:
            e["wall_lock"] += 0.15

        for k in list(e.keys()):
            e[k] = max(0.0, min(1.0, e[k]))

    def dominant_threat(self):
        if not self.evidence:
            return "unknown"
        return max(self.evidence, key=self.evidence.get)

    def likely_attack_side(self):
        left  = sum(v for (x, _), v in self.attack_heatmap.items() if x < 13)
        right = sum(v for (x, _), v in self.attack_heatmap.items() if x > 14)
        if left > right * 1.3:
            return "left"
        if right > left * 1.3:
            return "right"
        return "centre"

    def weakest_breach_zone(self):
        if not self.breach_heatmap:
            return None
        return max(self.breach_heatmap, key=self.breach_heatmap.get)

    def predicted_enemy_spawn(self):
        if not self.attack_heatmap:
            return None
        return list(max(self.attack_heatmap, key=self.attack_heatmap.get))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

class AlgoStrategy(gamelib.AlgoCore):

    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('APEX v3.1 — seed: {}'.format(seed))

    def on_game_start(self, config):
        gamelib.debug_write('APEX v3.1 starting...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL        = config["unitInformation"][0]["shorthand"]
        SUPPORT     = config["unitInformation"][1]["shorthand"]
        TURRET      = config["unitInformation"][2]["shorthand"]
        SCOUT       = config["unitInformation"][3]["shorthand"]
        DEMOLISHER  = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0

        self.opponent          = OpponentModel()
        self.scored_on_locs    = []
        self.enemy_spawn_locs  = []
        self.enemy_demo_spawns = 0
        self.funnel_side       = "left"
        self.last_hp           = 30
        self.mp_bank_turns     = 0
        self.mp_bank_target    = 10
        self.probe_counter     = 0
        
        # Action Queues for resolving same-turn limits
        self.pending_upgrades  = []
        self.pending_removals  = []
        self.corridor_supports = []
        self.is_unlocked       = False

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN TURN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        t = game_state.turn_number
        game_state.suppress_warnings(True)

        # Dynamic Map Orientation
        global MY_EDGES, ENEMY_EDGES
        if game_state.player_index == 0:
            MY_EDGES = gamelib.game_map.BOTTOM_LEFT + gamelib.game_map.BOTTOM_RIGHT
            ENEMY_EDGES = gamelib.game_map.TOP_LEFT + gamelib.game_map.TOP_RIGHT
        else:
            MY_EDGES = gamelib.game_map.TOP_LEFT + gamelib.game_map.TOP_RIGHT
            ENEMY_EDGES = gamelib.game_map.BOTTOM_LEFT + gamelib.game_map.BOTTOM_RIGHT

        # Process Pending Actions from Last Turn
        for loc in self.pending_upgrades:
            game_state.attempt_upgrade(loc)
        self.pending_upgrades.clear()

        for loc in self.pending_removals:
            game_state.attempt_remove(loc)
        self.pending_removals.clear()

        raw_sp = game_state.get_resource(SP)
        raw_mp = game_state.get_resource(MP)
        gamelib.debug_write('Turn {} | SP={} MP={} | threat={}'.format(
            t, raw_sp, raw_mp, self.opponent.dominant_threat()))

        self.opponent.update(game_state, self.scored_on_locs,
                             self.enemy_spawn_locs, self.enemy_demo_spawns)
        self._adapt_funnel_side(game_state)

        if t in OPENING_BOOK:
            self._play_opening(game_state, t)
            if t == 2:
                # Queue opening supports for upgrade next turn
                self.pending_upgrades.extend([[13, 16], [14, 16]])
        else:
            self._predict_and_reinforce(game_state)
            self.build_funnel_walls(game_state)
            self.build_core_turrets(game_state)
            self.build_supports(game_state)
            self.fill_secondary_turrets(game_state)
            self.plug_breaches(game_state)
            self.prioritise_upgrades(game_state)

        self.execute_attack(game_state)

        # Queue any placed corridor supports for removal at start of next turn
        self.pending_removals.extend(self.corridor_supports)
        self.corridor_supports.clear()

        self.scored_on_locs    = []
        self.enemy_spawn_locs  = []
        self.enemy_demo_spawns = 0
        self.last_hp           = game_state.my_health
        game_state.submit_turn()

    def _play_opening(self, game_state, turn):
        unit_map = {
            "WALL": WALL, "TURRET": TURRET,
            "SUPPORT": SUPPORT, "INTERCEPTOR": INTERCEPTOR,
        }
        for (type_name, loc) in OPENING_BOOK[turn]:
            utype = unit_map.get(type_name)
            if utype is None:
                continue
            if (game_state.game_map.in_arena_bounds(loc)
                    and not game_state.contains_stationary_unit(loc)):
                game_state.attempt_spawn(utype, [loc])

    def _predict_and_reinforce(self, game_state):
        sp = game_state.get_resource(SP)
        if sp is None or sp < 6:
            return  

        predicted_spawn = self.opponent.predicted_enemy_spawn()
        if predicted_spawn is None:
            return

        try:
            path = game_state.find_path_to_edge(predicted_spawn)
        except Exception:
            return

        if not path:
            return

        placed = 0
       hp = 15  # scout hp (adjust if needed)
damage = 0

for step in path:
    attackers = game_state.get_attackers(step, 0)
    step_damage = sum(a.damage_s for a in attackers)
    damage += step_damage
    if damage >= hp:
        break
    survival_ratio = max(0, (hp - damage) / hp)
score = edge_bonus + path_depth * 2 + survival_ratio * 20

def _adapt_funnel_side(self, game_state):
        if game_state.turn_number < 5:
            return
        attack_side = self.opponent.likely_attack_side()
        if attack_side != "centre":
            self.funnel_side = attack_side

def _occupied_by_turret(self, game_state, loc):
        units = game_state.game_map[loc[0]][loc[1]]
        return bool(units and units[0].unit_type == TURRET and units[0].player_index == 0)

def build_funnel_walls(self, game_state):
        if self.is_unlocked:
            return # Don't rebuild walls we intentionally destroyed

        def safe_walls(coords):
            return [c for c in coords
                    if game_state.game_map.in_arena_bounds(c)
                    and not self._occupied_by_turret(game_state, c)]

        back_walls = safe_walls([[x, 13] for x in range(0, 28)])
        game_state.attempt_spawn(WALL, back_walls)
        if game_state.get_resource(SP) >= 20:
            game_state.attempt_upgrade(back_walls)

        if self.funnel_side == "left":
            funnel_neck = safe_walls([[11,14],[10,15],[9,16],[8,17],[7,18]])
            right_block = safe_walls([[x,14] for x in range(15, 26)])
        else:
            funnel_neck = safe_walls([[16,14],[17,15],[18,16],[19,17],[20,18]])
            right_block = safe_walls([[x,14] for x in range(2, 13)])

        game_state.attempt_spawn(WALL, funnel_neck)
        game_state.attempt_spawn(WALL, right_block)
        if game_state.get_resource(SP) >= 14:
            game_state.attempt_upgrade(funnel_neck)

def _self_destruct_unlock(self, game_state):
        self.is_unlocked = True 
        if self.funnel_side == "left":
            candidates = [[15, 14], [16, 14], [17, 14]]
        else:
            candidates = [[12, 14], [11, 14], [10, 14]]

        for loc in candidates:
            units = game_state.game_map[loc[0]][loc[1]]
            if units and units[0].unit_type == WALL and units[0].player_index == 0:
                self.pending_removals.append(loc) 
                gamelib.debug_write('Self-destruct unlock triggered for {}'.format(loc))
                return  

def build_core_turrets(self, game_state):
        core = [
            [13,14],[14,14],[12,14],[15,14],
            [13,15],[14,15],
            [11,15],[10,16],[9,17],
            [16,15],[17,16],[18,17],
        ]
        game_state.attempt_spawn(TURRET, core)

def build_supports(self, game_state):
        locs = [[13,16],[14,16],[11,17],[16,17]]
        game_state.attempt_spawn(SUPPORT, locs)
        game_state.attempt_upgrade(locs)

def fill_secondary_turrets(self, game_state):
        bank_threshold, _ = self._sp_thresholds(game_state.turn_number)
        if game_state.get_resource(SP) < bank_threshold:
            return
        secondary = [
            [3,13],[24,13],[5,14],[22,14],
            [7,16],[20,16],[9,18],[18,18],
            [11,20],[16,20],[2,13],[25,13],
            [0,13],[27,13],
        ]
        game_state.attempt_spawn(TURRET, secondary)

def plug_breaches(self, game_state):
        hot = self.opponent.weakest_breach_zone()
        if hot:
            x, y = hot
            for dx in [0, -1, 1]:
                loc = [x + dx, y + 1]
                if game_state.game_map.in_arena_bounds(loc):
                    if game_state.attempt_spawn(TURRET, loc):
                        self.pending_upgrades.append(loc)

        for location in self.scored_on_locs:
            x, y = location
            candidates = [[x, y+1],[x-1, y+1],[x+1, y+1]]
            for c in candidates:
                if game_state.game_map.in_arena_bounds(c):
                    if game_state.attempt_spawn(TURRET, c):
                        self.pending_upgrades.append(c)
                    for wx in [x-2, x+2]:
                        wloc = [wx, y+1]
                        if game_state.game_map.in_arena_bounds(wloc):
                            game_state.attempt_spawn(WALL, wloc)
                    break

    def prioritise_upgrades(self, game_state):
        sp = game_state.get_resource(SP)
        _, upgrade_reserve = self._sp_thresholds(game_state.turn_number)
        if sp < upgrade_reserve + 6:
            return

        if self.funnel_side == "left":
            priority = [
                [11,15],[10,16],[9,17],
                [13,14],[14,14],[12,14],
                [13,15],[14,15],
                [16,15],[17,16],
                [3,13],[24,13],
            ]
        else:
            priority = [
                [16,15],[17,16],[18,17],
                [13,14],[14,14],[15,14],
                [13,15],[14,15],
                [11,15],[10,16],
                [3,13],[24,13],
            ]

        for loc in priority:
            _, upgrade_reserve = self._sp_thresholds(game_state.turn_number)
            if game_state.get_resource(SP) < upgrade_reserve:
                break
            units = game_state.game_map[loc[0]][loc[1]]
            if units and not units[0].upgraded:
                game_state.attempt_upgrade(loc)

def _sp_thresholds(self, turn):
        if turn < 10:
            return 8, 4
        elif turn < 25:
            return 12, 5
        else:
            return 18, 8

def _should_bank_mp(self, game_state, mp, threat, all_in):
        if all_in:
            return False
        if threat == "rush":
            return False
        hp_lost = self.last_hp - game_state.my_health
        if hp_lost >= 3:
            return False
        if self.mp_bank_turns >= 3:
            return False
        if mp >= self.mp_bank_target:
            return False
        return True

def execute_attack(self, game_state):
        raw_mp   = game_state.get_resource(MP)
        mp       = float(raw_mp) if raw_mp is not None else 0.0
        turn     = game_state.turn_number
        threat   = self.opponent.dominant_threat()
        enemy_hp = game_state.enemy_health

        all_in = (enemy_hp <= ALL_IN_THRESHOLD_HP) or (turn >= ALL_IN_TURN)

        self.probe_counter += 1
        if self.probe_counter >= PROBE_INTERVAL and mp >= 1 and not all_in:
            self._probe(game_state)
            self.probe_counter = 0
            raw_mp = game_state.get_resource(MP)
            mp = float(raw_mp) if raw_mp is not None else 0.0

        if self._should_bank_mp(game_state, mp, threat, all_in):
            self.mp_bank_turns += 1
            return
        self.mp_bank_turns = 0

        if mp < MIN_ATTACK_MP and not all_in:
            return

        spawn_candidates = self._get_spawn_candidates(game_state)
        scored = self._score_all_spawns(game_state, spawn_candidates)

        if not scored:
            return

        best_spawn, best_score = scored[0]

        # Reset destruct loop if paths clear up
        if best_score > SELF_DESTRUCT_SCORE + 5:
            self.is_unlocked = False

        if best_score < SELF_DESTRUCT_SCORE:
            self.mp_bank_target = 16
            self._self_destruct_unlock(game_state)  
        elif best_score < 10:
            self.mp_bank_target = 14
        elif best_score > 30:
            self.mp_bank_target = 8
        else:
            self.mp_bank_target = 10

        if all_in:
            self._execute_all_in(game_state, best_spawn, scored)
        elif threat in ("emp_spam", "attrition", "wall_lock"):
            self._execute_demo_assault(game_state, best_spawn, mp, threat)
        elif threat == "rush":
            self._execute_interceptor_bait(game_state, best_spawn, mp)
        elif mp >= SPLIT_ATTACK_MP and len(scored) >= 2:
            self._execute_split_attack(game_state, scored, mp)
        else:
            self._execute_scout_flood(game_state, best_spawn, mp)

def _probe(self, game_state):
        for loc in [[3, 10], [24, 10]]:
            if (game_state.game_map.in_arena_bounds(loc)
                    and not game_state.contains_stationary_unit(loc)):
                if game_state.attempt_spawn(SCOUT, loc, 1):
                    return

def _get_spawn_candidates(self, game_state):
        attack_side = self.opponent.likely_attack_side()
        if attack_side == "right":
            preferred = [[3,10],[4,9],[6,7],[13,0],[0,13],[2,11],[5,8]]
            fallback  = [[24,10],[23,9],[21,7],[14,0],[27,13]]
        elif attack_side == "left":
            preferred = [[24,10],[23,9],[21,7],[14,0],[27,13],[25,11],[22,8]]
            fallback  = [[3,10],[4,9],[6,7],[13,0],[0,13]]
        else:
            preferred = [[3,10],[24,10],[13,0],[14,0],[6,7],[21,7],[4,9],[23,9]]
            fallback  = [[0,13],[27,13],[2,11],[25,11]]

        all_candidates = preferred + fallback
        valid = [
            loc for loc in all_candidates
            if game_state.game_map.in_arena_bounds(loc)
            and not game_state.contains_stationary_unit(loc)
        ]
        return valid if valid else [[13, 0]]

def _score_all_spawns(self, game_state, spawn_candidates):
        enemy_edge_set = set(map(tuple, ENEMY_EDGES))
        results = []

        for loc in spawn_candidates:
            try:
                path = game_state.find_path_to_edge(loc)
                if not path:
                    continue

                incoming = 0.0
                # Performance opt: Step by 2
                for step in path[::2]:
                    attackers = game_state.get_attackers(step, 0)
                    # Fixed damage lookup targeting scouts
                    incoming += sum(getattr(a, 'damage_s', 0) for a in attackers)

                shields    = self._count_shield_buffs(game_state, path)
                dmg_weight = max(0.25, 0.5 - shields * 0.15)
                path_depth = sum(1 for step in path if step[1] < 14)
                last_step  = tuple(path[-1])
                edge_bonus = 40 if last_step in enemy_edge_set else 0
                
                # Corrected logic multiplier due to skipping steps
                score = edge_bonus + path_depth * 3 - (incoming * 2) * dmg_weight

                results.append((loc, score))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results

def _count_shield_buffs(self, game_state, path):
        UPGRADED_SUPPORT_RANGE = 7
        path_set = {(s[0], s[1]) for s in path}
        count = 0
        for y in range(14, 28):
            for x in range(28):
                units = game_state.game_map[x][y]
                if not units:
                    continue
                u = units[0]
                if u.unit_type == SUPPORT and u.player_index == 0 and u.upgraded:
                    for (px, py) in path_set:
                        if abs(px - x) + abs(py - y) <= UPGRADED_SUPPORT_RANGE:
                            count += 1
                            break
        return count

def _place_corridor_supports(self, game_state, spawn):
        sx, sy = spawn[0], spawn[1]
        candidates = [
            [sx,     sy + 2],
            [sx + 1, sy + 2],
            [sx - 1, sy + 2],
            [sx,     sy + 3],
        ]
        placed = 0
        sp = game_state.get_resource(SP)
        if sp is None or sp < 8:
            return  

        for loc in candidates:
            if placed >= 2:
                break
            if (game_state.game_map.in_arena_bounds(loc)
                    and not game_state.contains_stationary_unit(loc)
                    and loc[1] >= 14):
                if game_state.attempt_spawn(SUPPORT, [loc]):
                    self.pending_upgrades.append(loc)
                    self.corridor_supports.append(loc)
                    placed += 1

def _best_demo_spawn(self, game_state, spawn_candidates):
        DEMO_RANGE = 4  
        best_loc   = spawn_candidates[0]
        best_hits  = -1

        for loc in spawn_candidates:
            try:
                path = game_state.find_path_to_edge(loc)
                if not path:
                    continue
                hits = 0
                for step in path[::2]:
                    for dy in range(-DEMO_RANGE, DEMO_RANGE + 1):
                        for dx in range(-DEMO_RANGE, DEMO_RANGE + 1):
                            if abs(dx) + abs(dy) > DEMO_RANGE:
                                continue
                            tx, ty = step[0] + dx, step[1] + dy
                            if not (0 <= tx < 28 and 0 <= ty < 28):
                                continue
                            units = game_state.game_map[tx][ty]
                            if (units and units[0].player_index == 1
                                    and units[0].unit_type in [WALL, TURRET]):
                                hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_loc  = loc
            except Exception:
                continue

        return best_loc

def _execute_scout_flood(self, game_state, spawn, mp):
        self._place_corridor_supports(game_state, spawn)
        game_state.attempt_spawn(SCOUT, spawn, 1000)

def _execute_split_attack(self, game_state, scored, mp):
        best_spawn,   _ = scored[0]
        second_spawn, _ = scored[1]

        if abs(best_spawn[0] - second_spawn[0]) < 10:
            best_x = best_spawn[0]
            for loc, score in scored[1:]:
                if abs(loc[0] - best_x) >= 10:
                    second_spawn = loc
                    break
            else:
                self._execute_scout_flood(game_state, best_spawn, mp)
                return

        self._place_corridor_supports(game_state, best_spawn)
        primary_count   = max(1, int(mp * 0.6))
        secondary_count = 1000  

        game_state.attempt_spawn(SCOUT, best_spawn,   primary_count)
        game_state.attempt_spawn(SCOUT, second_spawn, secondary_count)

def _execute_demo_assault(self, game_state, spawn, mp, threat="attrition"):
        all_candidates  = self._get_spawn_candidates(game_state)
        demo_spawn      = self._best_demo_spawn(game_state, all_candidates)

        max_demos = 5 if threat == "wall_lock" else 3
        n_demo    = min(max_demos, max(1, int(mp // 4)))

        game_state.attempt_spawn(DEMOLISHER, demo_spawn, n_demo)

        remaining = game_state.get_resource(MP)
        if remaining is not None and remaining >= 1:
            game_state.attempt_spawn(SCOUT, spawn, 1000)

def _execute_interceptor_bait(self, game_state, spawn, mp):
        intercept_locs = [[13,17],[14,17],[10,17],[17,17]]
        n_intercept    = min(3, int(mp // 3))
        for loc in intercept_locs[:n_intercept]:
            if not game_state.contains_stationary_unit(loc):
                game_state.attempt_spawn(INTERCEPTOR, loc, 1)
        remaining = game_state.get_resource(MP)
        if remaining is not None and remaining >= MIN_ATTACK_MP:
            game_state.attempt_spawn(SCOUT, spawn, 1000)

def _execute_all_in(self, game_state, spawn, scored=None):
        mp = game_state.get_resource(MP)
        mp = float(mp) if mp is not None else 0.0

        all_candidates = self._get_spawn_candidates(game_state)
        demo_spawn     = self._best_demo_spawn(game_state, all_candidates)

        if mp >= 6:
            game_state.attempt_spawn(DEMOLISHER, demo_spawn, 2)

        if scored and len(scored) >= 2 and mp >= SPLIT_ATTACK_MP:
            best_spawn,   _ = scored[0]
            for loc, _ in scored[1:]:
                if abs(loc[0] - best_spawn[0]) >= 10:
                    game_state.attempt_spawn(SCOUT, best_spawn, 1000)
                    game_state.attempt_spawn(SCOUT, loc,        1000)
                    return

        game_state.attempt_spawn(SCOUT, spawn, 1000)

def on_action_frame(self, turn_string):
        state  = json.loads(turn_string)
        events = state.get("events", {})

        for breach in events.get("breach", []):
            location   = breach[0]
            owner_self = (breach[4] == 1)
            if not owner_self:
                self.scored_on_locs.append(location)

        for spawn in events.get("spawn", []):
            location  = spawn[0]
            unit_type = spawn[1]
            owner     = spawn[3]
            if owner == 2:
                if unit_type in [SCOUT, DEMOLISHER, INTERCEPTOR]:
                    self.enemy_spawn_locs.append(location)
                if unit_type == DEMOLISHER:
                    self.enemy_demo_spawns += 1

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
