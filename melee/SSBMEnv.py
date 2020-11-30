import gym, melee, sys, signal, time, os
from ray.rllib.env import MultiAgentEnv
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from melee import enums
import pynput
import json
from melee.utils import Timeout
from itertools import product

"""
Gym compatible env for libmelee (an RL framework for SSBM)

Attr:
    -
"""
SLIPPI_ADDRESS = "127.0.0.1"
SLIPPI_PORT=51441
PLAYER_PORT=1
OP_PORT=2
CONNECT_CODE=""

# TODO: allow increasable bot difficulty
"""
BUTTON_A = "A"
BUTTON_B = "B"
BUTTON_X = "X"
BUTTON_Y = "Y"
BUTTON_Z = "Z"
BUTTON_L = "L"
BUTTON_R = "R"
BUTTON_START = "START"
BUTTON_D_UP = "D_UP"
BUTTON_D_DOWN = "D_DOWN"
BUTTON_D_LEFT = "D_LEFT"
BUTTON_D_RIGHT = "D_RIGHT"
#Control sticks considered "buttons" here
BUTTON_MAIN = "MAIN"
BUTTON_C = "C"
Action space: [BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y, BUTTON_Z, BUTTON_L, BUTTON_R, BUTTON_D_UP, BUTTON_D_DOWN, BUTTON_D_LEFT, BUTTON_D_RIGHT,
                BUTTON_A_R, BUTTON_B_R, BUTTON_X_R, BUTTON_Y_R, BUTTON_Z_R, BUTTON_L_R, BUTTON_R_R, BUTTON_D_UP_R, BUTTON_D_DOWN_R, BUTTON_D_LEFT_R, BUTTON_D_RIGHT_R,
                BUTTON_MAIN (0, 0), BUTTON_MAIN (0.5, 0), BUTTON_MAIN (0, 0.5), BUTTON_MAIN (1, 0), BUTTON_MAIN (0, 1), BUTTON_MAIN (1, 0.5), BUTTON_MAIN (0.5, 1), BUTTON_MAIN (1, 1),
                BUTTON_C (0, 0), BUTTON_C (0.5, 0), BUTTON_C (0, 0.5), BUTTON_C (1, 0), BUTTON_C (0, 1), BUTTON_C (1, 0.5), BUTTON_C (0.5, 1), BUTTON_C (1, 1)]

Observation space: [p1_char, p1_x, p1_y, p1_percent, p1_shield, p1_facing, p1_action_enum_value, p1_action_frame, p1_invulnerable, p1_invulnerable_left, p1_hitlag, p1_hitstun_frames_left, p1_jumps_left, p1_on_ground, p1_speed_air_x_self,
                    p1_speed_y_self, p1_speed_x_attack, p1_speed_y_attack, p1_speed_ground_x_self, distance_btw_players, ...p2 same attr...]
"""

buttons = [enums.Button.BUTTON_A, enums.Button.BUTTON_B, enums.Button.BUTTON_X, enums.Button.BUTTON_Z,
               enums.Button.BUTTON_L, enums.Button.BUTTON_R, enums.Button.BUTTON_D_UP, enums.Button.BUTTON_D_DOWN, enums.Button.BUTTON_D_LEFT,
               enums.Button.BUTTON_D_RIGHT]

tilt_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
intervals = list(product(tilt_bins, repeat=2))

class SSBMEnv(MultiAgentEnv):
    DOLPHIN_SHUTDOWN_TIME = 5 # number of seconds we wait for dolphin process shutdown before restarting
    NUM_ACTIONS = (len(buttons) + len(intervals))*2 # num actions *2 for press/release and both joysticks
    OBSERVATION_DIM = 87



    """
    SSBMEnv Constructor

    Attr:
        - self.logger: log useful data in csv
        - self.console: console to communicate with slippi dolphin
        - self.cpu: True if one agent is bot
        - self.ctrlr: controller for char1
        - self.ctrlr_op: controller for char2 (could be cpu bot)
        -

    Args:
        - dolphin_exe_path: path to dolphin exe
        - ssbm_iso_path: path to ssbm iso
        - char1, char2: melee.Character enum
        - stage: where we playin?
        - cpu: False if we are training policies for both char1 and char2
                     else char2 is cpu bot
        - cpu_level: if cpu=True this is the level of the cpu
        - log: are we logging stuff?
        - reward_func: custom reward function should take two gamestate objects as input and output a tuple
                       containing the reward for the player and opponent
        - dump_states (bool): If True, every reset, save the past history as JSON
          array of dict {'state': state, agent_name: agent_action ... }
          to the file {statedump_dir}/{statedump_prefix}_{n}.txt where n starts at 1 and
          increments every reset. Only dump if self.dump_states is True
        - statedump_dir (path): Directory to store states, if dump_states is set
        - statedump_prefiex (str): File prefix for statedumps
        - kill_reward: (int) Reward an agents gets (loses) for each kill (death) in default reward function
        - aggro_coeff: (float): Relative weight of damage given to damage taken in potential function. >1 encourages more aggressive agents
        - gamma (float): Discount factor
        - shaping_coeff (float): Relative weight of potential reward to sparse reward
        - off_stage_weight (float): Penalty agent receives for being off stage
        - num_dolpin_retries (int): Number of times we re-try to start dolphin before giving up
        - dolphin_timeout (int): Number of seconds after which we consider a dolphin startup failed
    """
    def __init__(self, dolphin_exe_path, ssbm_iso_path, char1=melee.Character.FOX, char2=melee.Character.FALCO,
                stage=melee.Stage.FINAL_DESTINATION, cpu=False, cpu_level=1, log=False, reward_func=None, kill_reward=200, aggro_coeff=1, gamma=0.99, shaping_coeff=1, off_stage_weight=10, num_dolphin_retries=3, dolphin_timeout=20, dump_states=False, statedump_dir=os.path.abspath('.'), statedump_prefix="ssbm_out", **kwargs):
        ### Args checking ###

        # Paths
        if not os.path.exists(dolphin_exe_path) or not os.path.isdir(dolphin_exe_path):
            raise ValueError("dolphin exe path {} is not a valid path to the executable directory!".format(dolphin_exe_path))
        if not os.path.exists(ssbm_iso_path) or not os.path.isfile(ssbm_iso_path):
            raise ValueError("ssbm_iso_path {} is not a valid path to an ISO file!".format(ssbm_iso_path))
        # constants 
        if dolphin_timeout < 0 or dolphin_timeout > 100:
            raise ValueError("Dolphin_timeout must be an in [0, 100]")
        if not type(num_dolphin_retries) == int or num_dolphin_retries < 0 or num_dolphin_retries > 5:
            raise ValueError("num_dolphin_retries must be an int in [0, 5]")
        if not type(cpu_level) == int or cpu_level < 1 or cpu_level > 9:
            raise ValueError("cpu_level must be an int in [1, 9]")
        # Enums
        try:
            is_valid = char1 in melee.Character
        except:
            raise ValueError("{} is not a valid character!".format(char1))
        else:
            if not is_valid:
                raise ValueError("{} is not a valid character!".format(char1))
        try:
            is_valid = char2 in melee.Character
        except:
            raise ValueError("{} is not a valid character!".format(char2))
        else:
            if not is_valid:
                raise ValueError("{} is not a valid character!".format(char2))
        try:
            is_valid = stage in melee.Stage
        except:
            raise ValueError("{} is not a valid character!".format(stage))
        else:
            if not is_valid:
                raise ValueError("{} is not a valid character!".format(stage))
        
        # Assign instance variables
        self.dolphin_exe_path = dolphin_exe_path
        self.ssbm_iso_path = ssbm_iso_path
        self.char1 = char1
        self.char2 = char2
        self.stage = stage
        self.cpu = cpu
        self.cpu_level = cpu_level
        self.reward_func = reward_func
        self.logger = melee.Logger()
        self.log = log
        self.console = None
        self.gamma = gamma
        self.kill_reward = kill_reward
        self.aggro_coeff = aggro_coeff
        self.shaping_coeff = shaping_coeff
        self.off_stage_weight = off_stage_weight
        self.num_dolphin_retries = num_dolphin_retries
        self.dolphin_timeout = dolphin_timeout
        self._is_dolphin_running = False
        self.dump_states = dump_states
        self.statedump_dir = statedump_dir
        self.statedump_prefix = statedump_prefix
        self.statedump_n = 0
        self.state_data = []

        # Space creation
        self.get_reward = self._default_get_reward if not self.reward_func else self.reward_func
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS+1) # plus one for nop

    def _default_get_reward(self, prev_gamestate, gamestate): # define reward function
        sparse_reward = self._get_sparse_reward(prev_gamestate, gamestate)
        potential_reward = self._get_potential_reward(prev_gamestate, gamestate)

        joint_shaped_reward = {}
        for i, agent in enumerate(self.agents):
            joint_shaped_reward[agent] = sparse_reward[agent] + self.shaping_coeff * potential_reward[agent]

        return joint_shaped_reward

    def _get_sparse_reward(self, prev_gamestate, gamestate):
        # TODO: make sure that the correct damage goes to the correct player
        p1DamageDealt = max(gamestate.player[self.ctrlr_op_port].percent - prev_gamestate.player[self.ctrlr_op_port].percent, 0)
        p1DamageTaken = max(gamestate.player[self.ctrlr_port].percent - prev_gamestate.player[self.ctrlr_port].percent, 0)

        isp1Dead = gamestate.player[self.ctrlr_port].action.value <= 0xa
        isp2Dead = gamestate.player[self.ctrlr_op_port].action.value <= 0xa

        wasp1Dead = prev_gamestate.player[self.ctrlr_port].action.value <= 0xa
        wasp2Dead = prev_gamestate.player[self.ctrlr_op_port].action.value <= 0xa


        p1rkill = isp2Dead and not wasp2Dead 
        p1rdeath = isp1Dead and not wasp1Dead

        p1_reward = self.aggro_coeff * p1DamageDealt - p1DamageTaken + p1rkill * self.kill_reward - p1rdeath * self.kill_reward
        p2_reward = self.aggro_coeff * p1DamageTaken - p1DamageDealt + p1rdeath * self.kill_reward - p1rkill * self.kill_reward

        rewards = [p1_reward, p2_reward]

        joint_reward = {}
        for i, agent in enumerate(self.agents):
            joint_reward[agent] = rewards[i]

        return joint_reward



    def _potential(self, gamestate):
        p1_off_stage = -1 * int(gamestate.player[self.ctrlr_port].off_stage) * self.off_stage_weight
        p2_off_stage = -1 * int(gamestate.player[self.ctrlr_op_port].off_stage) * self.off_stage_weight

        potentials = [p1_off_stage, p2_off_stage]

        joint_potential = {}
        for i, agent in enumerate(self.agents):
            joint_potential[agent] = potentials[i]

        return joint_potential

    def _get_potential_reward(self, s, s_prime):
        phi_s = self._potential(s)
        phi_s_prime = self._potential(s_prime)

        joint_potential_reward = {}

        for i, agent in enumerate(self.agents):
            joint_potential_reward[agent] = self.gamma * phi_s_prime[agent] - phi_s[agent]

        return joint_potential_reward

    def _get_state(self):
        """
        [p1_char, p1_x, p1_y, p1_percent, p1_shield, p1_facing, p1_action_enum_value, p1_action_frame, p1_invulnerable,
         p1_invulnerable_left, p1_hitlag, p1_hitstun_frames_left, p1_jumps_left, p1_on_ground, p1_speed_air_x_self,
         p1_speed_y_self, p1_speed_x_attack, p1_speed_y_attack, p1_speed_ground_x_self, distance_btw_players, ...p2 same attr...]
        """
        # TODO: make sure the correct state goes to the correct player
        # I make the assumption that p1 is *always* a non-cpu and p2 is the cpu, if present
        p1 = self.gamestate.player[self.ctrlr_port]
        p2 = self.gamestate.player[self.ctrlr_op_port]

        # What buttons are pressed (or not)
        p1_button_state = [float(p1.controller_state.button[btn]) for btn in buttons]
        p2_button_state = [float(p2.controller_state.button[btn]) for btn in buttons]

        # Position of joysticks
        p1_stick_state = [p1.controller_state.main_stick[0], p1.controller_state.main_stick[1], 
                          p1.controller_state.c_stick[0], p1.controller_state.c_stick[1]]
        p2_stick_state = [p2.controller_state.main_stick[0], p2.controller_state.main_stick[1], 
                          p2.controller_state.c_stick[0], p2.controller_state.c_stick[1]]

        # All controller info
        p1_controller_state = [*p1_button_state, *p1_stick_state]
        p2_controller_state = [*p2_button_state, *p2_stick_state]

        # All character-centric info
        p1_character_state = [p1.character.value, p1.x, p1.y, p1.percent, p1.shield_strength, p1.facing, p1.action.value, p1.action_frame,
                              float(p1.invulnerable), p1.invulnerability_left, float(p1.hitlag), p1.hitstun_frames_left, p1.jumps_left,
                              float(p1.on_ground), p1.speed_air_x_self, p1.speed_y_self, p1.speed_x_attack, p1.speed_y_attack, p1.speed_ground_x_self,
                              float(p1.off_stage), p1.moonwalkwarning, *p1.ecb_right, *p1.ecb_left, *p1.ecb_top, *p1.ecb_bottom]
        p2_character_state = [p2.character.value, p2.x, p2.y, p2.percent, p2.shield_strength, p2.facing, p2.action.value, p2.action_frame,
                              float(p2.invulnerable), p2.invulnerability_left, float(p2.hitlag), p2.hitstun_frames_left, p2.jumps_left,
                              float(p2.on_ground), p2.speed_air_x_self, p2.speed_y_self, p2.speed_x_attack, p2.speed_y_attack, p2.speed_ground_x_self,
                              float(p2.off_stage), p2.moonwalkwarning, *p2.ecb_right, *p2.ecb_left, *p2.ecb_top, *p2.ecb_bottom]

        # All agent-centric info
        p1_state = [*p1_controller_state, *p1_character_state]
        p2_state = [*p2_controller_state, *p2_character_state]

        # Final observations (note they're symmetrically encoded so that 'my' state always comes first)
        p1_obs = np.array([*p1_state, self.gamestate.distance, *p2_state])
        p2_obs = np.array([*p2_state, self.gamestate.distance, *p1_state])

        observations = [p1_obs, p2_obs]
        obs_dict = { agent_name : observations[i] for i, agent_name in enumerate(self.agents) }

        return obs_dict

    def _get_done(self):
        done =  self.gamestate.player[self.ctrlr_port].action.value <= 0xa or self.gamestate.player[self.ctrlr_op_port].action.value <= 0xa
        return {'__all__' : done }

    def _get_info(self):
        # TODO write frames skipped to info  (I think if we miss more than 6 frames between steps we might be in trouble)
        info = {}
        for agent in self.agents:
            info[agent] = {}
        return info

    def _perform_action(self, player, action_idx):
        if action_idx == 0:
            return
        ctrlr = self.ctrlr if player == 0 else self.ctrlr_op
        action_idx -= 1
        len_b = len(buttons)
        len_i = len(intervals)
        if action_idx < len_b: # button press
            ctrlr.press_button(buttons[action_idx])
        elif action_idx < len_b*2: # button release
            ctrlr.release_button(buttons[action_idx-len_b])
        elif action_idx < len_b*2 + len_i: # main joystick tilt
            tlt = intervals[action_idx-len_b*2]
            ctrlr.tilt_analog(enums.Button.BUTTON_MAIN, tlt[0], tlt[1])
        else: # c joystick tilt
            tlt = intervals[action_idx-(len_b*2+len_i)]
            ctrlr.tilt_analog(enums.Button.BUTTON_C, tlt[0], tlt[1])

    def _start_dolphin(self):
        self.console = melee.Console(path=self.dolphin_exe_path,
                                    slippi_address=SLIPPI_ADDRESS,
                                    slippi_port=SLIPPI_PORT,
                                    blocking_input=False,
                                    polling_mode=False,
                                    logger=self.logger)
        self.ctrlr = melee.Controller(console=self.console,
                                    port=PLAYER_PORT,
                                    type=melee.ControllerType.STANDARD)
        self.ctrlr_op = melee.Controller(console=self.console,
                                    port=OP_PORT,
                                    type=melee.ControllerType.STANDARD)
        self.console.run(iso_path=self.ssbm_iso_path)
        self._is_dolphin_running = True
        print("Connecting to console...")
        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            raise RuntimeError("Failed to connect to console")

        # Plug our controller in
        print("Connecting controller to console...")
        if not self.ctrlr.connect():
            print("ERROR: Failed to connect the controller.")
            raise RuntimeError("Failed to connect to controller")
        if not self.ctrlr_op.connect():
            print("ERROR: Failed to connect the controller.")
            raise RuntimeError("Failed to connect to controller")
        print("Controllers connected")

    def _step_through_menu(self):
        # Step through main menu, player select, stage select scenes # TODO: include frame processing warning stuff
        self.gamestate = self.console.step()
        menu_helper = melee.MenuHelper(controller_1=self.ctrlr,
                                        controller_2=self.ctrlr_op,
                                        character_1_selected=self.char1,
                                        character_2_selected=self.char2,
                                        stage_selected=self.stage,
                                        connect_code=CONNECT_CODE,
                                        autostart=True,
                                        swag=False,
                                        make_cpu=self.cpu,
                                        level=self.cpu_level)

        while self.gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            self.gamestate = self.console.step()
            menu_helper.step(self.gamestate)
            if self.log:
                self.logger.logframe(self.gamestate)
                self.logger.writeframe()

    def _start_game(self):
        self._start_dolphin()
        self._step_through_menu()

    def start_game(self):
        if self._is_dolphin_running:
            self._stop_dolphin()
        
        start_game_timeout = Timeout(self._start_game, self.dolphin_timeout)

        for _ in range(self.num_dolphin_retries):
            success = start_game_timeout()
            if success:
                return True
            print("Failed to start dolphin. Retrying...")
            self._stop_dolphin()
        raise RuntimeError("Failed to properly start game!")



    def _stop_dolphin(self):
        print("STOPPING DOLPHIN")
        if self.console:
            self.console.stop()
            time.sleep(self.DOLPHIN_SHUTDOWN_TIME)
        self._is_dolphin_running = False

    def step(self, joint_action): # step should advance our state (in the form of the obs space)
        if set(joint_action.keys()).intersection(self.agents) != set(joint_action.keys()).union(self.agents):
            raise ValueError("Invalid agent in action dictionary!")

        # why do we need to do this?
        self.ctrlr_port = melee.gamestate.port_detector(self.gamestate, self.char1)
        self.ctrlr_op_port = melee.gamestate.port_detector(self.gamestate, self.char2)

        if self.ctrlr_port != self.ctrlr.port or self.ctrlr_op_port != self.ctrlr_op.port:
            raise RuntimeError("Controller port inconsistency!")

        prev_gamestate = self.gamestate

        # perform actions
        for agent_idx, agent in enumerate(self.agents):
            action = joint_action[agent]
            self._perform_action(agent_idx, action)      

        # step env
        self.gamestate = self.console.step()
        
        # Collect all transition data
        reward = self.get_reward(prev_gamestate, self.gamestate)
        state = self._get_state()
        done = self._get_done()
        info = self._get_info()

        all_done = done['__all__']

        # Log (s_t, a_t, s'_t) data if necessary
        if self.dump_states:
            # Log a_t
            for agent in self.agents:
                self.state_data[-1][agent] = joint_action[agent]
            # Log s'_t
            self.state_data[-1]['next_state'] = state
            # Begin next (s, a, s') tuple if not done
            if not all_done:
                # Log s_{t+1}
                self.state_data.append({ "state" : state })

        if self.gamestate.menu_state != melee.enums.Menu.IN_GAME:
            for key, _  in done.items():
                done[key] = True
        else:
            for key, _ in done.items():
                done[key] = False

        if all_done:
            self._stop_dolphin()
        
        return state, reward, done, info
    
    def state_np_to_list(self, dictionary):
        for entry in dictionary:
            if isinstance(dictionary[entry], np.ndarray):
                dictionary[entry] = dictionary[entry].tolist()
            elif isinstance(dictionary[entry], dict):
                self.state_np_to_list(dictionary[entry])
    
    def dump_state(self):
        if self.dump_states and self.state_data:
            with open(os.path.join(self.statedump_dir, f"{self.statedump_prefix}_{self.statedump_n}.txt"), "w") as f:
                for entry in self.state_data:
                    self.state_np_to_list(entry)
                f.write(json.dumps(self.state_data))
            self.statedump_n += 1
        self.state_data = []

    def reset(self):    # TODO: should reset state to initial state, how to do this?
        self.dump_state()
        # hashtag JustDolphinThings
        self.start_game()

        if not self.cpu:
            self.agents = ['ai_1', 'ai_2']
        else:
            self.agents = ['ai_1']

        self.ctrlr_port = melee.gamestate.port_detector(self.gamestate, self.char1)
        self.ctrlr_op_port = melee.gamestate.port_detector(self.gamestate, self.char2)

        if self.ctrlr_port != self.ctrlr.port or self.ctrlr_op_port != self.ctrlr_op.port:
            raise RuntimeError("Controller port inconsistency!")

        # Return initial observation
        joint_obs = self._get_state()

        if self.dump_states:
            self.state_data.append({ "state" : joint_obs })
        return joint_obs

    
    def render(self, mode='human', close=False):    # FIXME: changing this parameter does nothing rn??
        self.console.render = True
    
actions_list = ["ZERO", "BUTTON_A","BUTTON_B","BUTTON_X","BUTTON_Y","BUTTON_Z","BUTTON_L","BUTTON_R","BUTTON_D_UP","BUTTON_D_DOWN","BUTTON_D_LEFT","BUTTON_D_RIGHT","BUTTON_A_R","BUTTON_B_R","BUTTON_X_R","BUTTON_Y_R","BUTTON_Z_R","BUTTON_L_R","BUTTON_R_R","BUTTON_D_UP_R","BUTTON_D_DOWN_R","BUTTON_D_LEFT_R","BUTTON_D_RIGHT_R","BUTTON_MAIN00", "BUTTON_MAIN50", "BUTTON_MAIN05", "BUTTON_MAIN10", "BUTTON_MAIN55", "BUTTON_MAIN01", "BUTTON_MAIN15", "BUTTON_MAIN51", "BUTTON_MAIN11", "BUTTON_C00", "BUTTON_C50", "BUTTON_C05", "BUTTON_C10", "BUTTON_C55", "BUTTON_C01", "BUTTON_C15", "BUTTON_C51", "BUTTON_C11"]

def get_action(name):
    return actions_list.index(name)

keymap = {
    'w': "BUTTON_MAIN51", # up
    'a': "BUTTON_MAIN05", # left
    's': "BUTTON_MAIN50", # down
    'd': "BUTTON_MAIN15", # right
    'j': "BUTTON_A",      # attack
    'k': "BUTTON_B",      # special
    'l': "BUTTON_L",      # shield
    'i': "BUTTON_X",      # jump
    ';': "BUTTON_Z",      # grab
    ',': "BUTTON_D_UP",   # taunt
    'r': "BUTTON_C51",    # up smash
    'f': "BUTTON_C05",    # left smash
    'g': "BUTTON_C50",    # down smash
    'h': "BUTTON_C15",    # right smash
}

def action_from_keys(keys):
    keystring = "".join(sorted([str(k).strip("'\"") for k in keys]))
    for mapkey in keymap:
        if "".join(sorted(mapkey)) == keystring:
            return get_action(keymap[mapkey])
    return 0

# https://stackoverflow.com/questions/27750536/python-input-single-character-without-enter
def getchar():
    tty.setraw(sys.stdin.fileno())
    return sys.stdin.read(1)

keys_pressed = []
keys_released = []
actions_taken = []

def on_press(key):
    global keys_pressed
    if key not in keys_pressed:
        keys_pressed.append(key)

def on_release(key):
    global keys_released
    if key in keys_pressed:
        keys_pressed.remove(key)
    if key not in keys_released:
        keys_released.append(key)

def process_pressed():
    global keys_pressed
    action = action_from_keys([keys_pressed.pop()])
    if action not in actions_taken:
        actions_taken.append(action)
    return action

def process_released():
    action = action_from_keys([keys_released.pop()])
    if action in actions_taken:
        actions_taken.remove(action)
        return complement(action)
    return get_action("ZERO")

def complement(action):
    if "MAIN" in actions_list[action]:
        return get_action("BUTTON_MAIN55") # idk
    elif "BUTTON_C" in actions_list[action]:
        return get_action("BUTTON_C55") # idk
    elif actions_list[action] == "ZERO":
        return get_action("ZERO")
    else:
        return get_action(actions_list[action]+"_R")

if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser(description='Example of Gym Wrapper in action')
    parser.add_argument('--dolphin_exe_path', '-e', default=None,
                        help='The directory where dolphin is')
    parser.add_argument('--ssbm_iso_path', '-i', default="~/SSMB.iso",
                        help='Full path to Melee ISO file')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help='Whether to set oponent as CPU')
    parser.add_argument('--cpu_level', '-l', type=int, default=3,
                        help='Level of CPU. Only valid if cpu is true')
    parser.add_argument('--human', '-m', action='store_true', help='P1 Human')
    parser.add_argument('--dump_states', '-s', action='store_true', help='Should we log state-action pairs?')
    parser.add_argument('--statedump_dir', '-sd', type=str, default=os.path.abspath('.'), help='where to dump states to')
    parser.add_argument('--statedump_prefix', '-sp', type=str, default='out', help='file prefix for state dumps')
    
    args = parser.parse_args()
    
    pynput.keyboard.Listener(on_press=on_press, on_release=on_release).start()

    ssbm_env = SSBMEnv(**vars(args))
    obs = ssbm_env.reset()

    done = False
    
    while not done:
        # Perform first part of upsmash
        joint_action = {}
        if args.human:
                if keys_released:
                    action = process_released()
                elif keys_pressed:
                    action = process_pressed()
                else:
                    action = 0
                joint_action['ai_1'] = action
        else:
            joint_action['ai_1'] = 68
        if not args.cpu:
            joint_action['ai_2'] = 0
        obs, reward, done, info = ssbm_env.step(joint_action)
        done = done['__all__']

        if not args.human:
            # Perform second part of upsmash
            joint_action = {'ai_1': 65}
            if not done:
                if not args.cpu:
                    joint_action['ai_2'] = 0
                obs, reward, done, info = ssbm_env.step(joint_action)
                done = done['__all__']
        
    ssbm_env.dump_state() # dump the log. normally done w/env.reset()
