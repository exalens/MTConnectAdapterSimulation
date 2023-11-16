import socket
import traceback
from threading import Thread
import logging
import random
from datetime import datetime
import time
import inquirer
import numpy as np
import signal
from typing import Dict, Any
from collections import defaultdict
from enum import Enum

AnomalyType = [
    'Abnormally high value',
    'Abnormally low value',
    'Value increased slower than expected',
    'Value increased faster than expected',
    'Value decreased slower than expected',
    'Value decreased faster than expected',
    'New/Unexpected decrease in value',
    'New/Unexpected increase in value',
    'New/Unexpected change in value',
    'State change was unusually slow',
    'State change was unusually fast',
    'New/Unexpected transition between states',
    'Correlation: Abnormally high value during transition',
    'Correlation: Abnormally low value during transition',
    'Correlation: Value increased slower than expected during transition',
    'Correlation: Value increased faster than expected during transition',
    'Correlation: Value decreased slower than expected during transition',
    'Correlation: Value decreased faster than expected during transition'
]


class CNCServer:
    def __init__(self):
        self.running = True
        self.clients = []
        self.updated_data = {}
        self.send_all_data = False
        self.simulation_interval = 2
        self.simulation_thread = None
        self.shdr_data = self.initialize_shdr_data()
        self.server_logger = self.setup_logging('server', 'server.log')
        self.client_loggers = {}
        self.update_interval = 1
        self.scenarios = self.initialize_scenarios()
        self.execution_map = {'READY': 'ACTIVE', 'ACTIVE': 'FEED_HOLD', 'FEED_HOLD': 'READY'}
        self.non_transition_map = {'READY': 'FEED_HOLD', 'ACTIVE': 'READY', 'FEED_HOLD': 'ACTIVE'}
        self.execution_index_map = self.get_execution_state_mapping_index(self.scenarios)
        print(self.execution_index_map)
        self.min_max_values = self.calculate_min_max_values()
        self.current_scenario_index = 0
        # Initialize a defaultdict for client ping monitoring with a default value of 10
        self.client_ping_monitor = defaultdict(lambda: 10)
        signal.signal(signal.SIGINT, self.signal_handler)
        self._anomaly_flag = False

    def initialize_scenarios(self):
        return [
            {
                "name": "Startup",
                "duration": 15,
                "execution": "READY",
                "controller_mode": "AUTOMATIC",
                "variables": {
                    "temperature": {"normal": (0, 20), "steps": [1.0, 2.0, 0.5]},
                    "speed": {"normal": (100, 500), "steps": [10.0, 20.0, 5.0]},
                    "load": {"normal": (10, 50), "steps": [1.0, 2.0, 0.5]},
                    "current": {"normal": (15, 70), "steps": [1.0, 2.0, 0.5]},
                    "pos": {"normal": (-0.18834, 0.944543), "steps": [0.1, 0.2, 0.05]},
                    "abs": {"normal": (20, 50), "steps": [10, 20]},
                    "frt": {"normal": (-100, 100), "steps": [1.0, 2.0, 0.5]},
                }
            },
            {
                "name": "Normal Operation",
                "duration": 60,
                "execution": "ACTIVE",
                "controller_mode": "AUTOMATIC",
                "variables": {
                    "temperature": {"normal": (20, 80), "steps": [1.0, 0.5, 0.2]},
                    "speed": {"normal": (500, 2800), "steps": [10.0, 5.0, 2.0]},
                    "load": {"normal": (30, 90), "steps": [1.0, 0.5, 0.2]},
                    "current": {"normal": (10, 40), "steps": [1.0, 0.5, 0.2]},
                    "pos": {"normal": (-0.1, 0.9), "steps": [0.1, 0.09, 0.3]},
                    "abs": {"normal": (80, 120), "steps": [2, 3]},
                    "frt": {"normal": (-90, 0), "steps": [1.0, 0.5, 0.2]}
                }
            },
            {
                "name": "Maintenance Operation",
                "duration": 8,
                "execution": "FEED_HOLD",
                "controller_mode": "MANUAL",
                "variables": {
                    "temperature": {"normal": (0.1, 80), "steps": [1.0, 40, 0.2]},
                    "speed": {"normal": (500, 2800), "steps": [100, 300, 500]},
                    "load": {"normal": (30, 90), "steps": [5, 10, 15]},
                    "current": {"normal": (10, 40), "steps": [20, 10, 8]},
                    "pos": {"normal": (0.1, 0.9), "steps": [0.1, 0.9, 0.3]},
                    "abs": {"normal": (150, 170), "steps": [4, 6]},
                    "frt": {"normal": (1000, 1050), "steps": [1.0, 2, 4]}
                }
            }

        ]

    def create_execution_map(self, scenarios):
        execution_map = {}
        for i in range(len(scenarios) - 1):  # -1 to avoid the last scenario
            current_execution = scenarios[i]["execution"]
            next_execution = scenarios[i + 1]["execution"]
            execution_map[current_execution] = next_execution
        print(execution_map)
        return execution_map

    def compile_non_transition_map(self, scenarios):
        # Extract all scenario names
        scenario_names = [scenario["name"] for scenario in scenarios]

        # Create a map where each scenario cannot transition to any other scenario
        non_transition_map = {name: [other_name for other_name in scenario_names if other_name != name] for name in
                              scenario_names}

        return non_transition_map

    def get_execution_state_mapping_index(self, scenarios):
        return {scenario["execution"]: index for index, scenario in enumerate(scenarios)}

    def initialize_shdr_data(self):
        X_axis_data = {
            "Xabs": 0.0,
            "Xpos": 0.0,
            "Xtravel": "NORMAL",
            "Xfrt": 0.0,
            "Xload": 0.0,
            "Xspeed": 0.0,
            "Xcurrent": 0.0,
            "Xtemperature": 0.0,
        }

        # Y_axis_data = {
        #     "Yabs": 0.0,
        #     "Ypos": 0.0,
        #     "Ytravel": "NORMAL",
        #     "Yfrt": 0.0,
        #     "Yload": 0.0,
        #     "Yspeed": 0.0,
        #     "Ycurrent": 0.0,
        #     "Ytemperature": 0.0,
        # }
        #
        # Z_axis_data = {
        #     "Zabs": 0.0,
        #     "Zpos": 0.0,
        #     "Ztravel": "NORMAL",
        #     "Zfrt": 0.0,
        #     "Zload": 0.0,
        #     "Zspeed": 0.0,
        #     "Zcurrent": 0.0,
        #     "Ztemperature": 0.0,
        # }
        #
        C_axis_data = {
            "Cload": 0.0,
            "Ctravel": "NORMAL",
            "Cfrt": 0.0,
            "Cabs": 0.0,
            "Cpos": 0.0,
            "Caxis_state": "ACTIVE",
            "Crotary_mode": "SPINDLE",
            "Cspeed": 0.0,
            "Ccurrent": 0.0,
            "Ctemperature": 0.0,
        }
        #
        # spindle_data = {
        #     "Sload": 0.0,
        #     "Srpm": 0.0,
        #     "Stemp": 0.0,
        #     "Sload_cond": "NORMAL",
        #     "Stemp_cond": "NORMAL",
        #     "Scurrent": 0.0,
        # }

        general_data = {
            "line_number": 0,
            "deceleration": 10000,
            ""
            "rapid_override": 0,
            "feedrate_override": 1000,
            "rotary_velocity_override": None,
            "program": None,
            "active_program": None,
            "part_count_actual": 0,
            "part_count_target": 0,
            "actual_path_feedrate": 0.0,
            "tool_number": 99,
            "tool_group": None,
            "execution": None,
            "wait_state": None,
            "controller_mode": None,
            "program_comment": None,
            "active_program_comment": None,
            "motion_condition": "NORMAL",
            "system_condition": "NORMAL",
            "machine_axis_lock_override": None,
            "single_block_override": None,
            "dry_run_override": None,
            "line_label": None,
            "variables": None,
            "cutting_speed_actual": 0.0,
            "work_offset_translation": None,
            "work_offset_rotation": None,
            "work_offset": None,
            "active_axes": None,
            "path_position": None,
            "orientation": None,
            "process_time_start": None,
            "process_timer": 0.0,
            "p1_block": None,
            "p1_line": None,
            "MAG1d_temperature": 0.0,
            "A1d_speed": 0.0,
            "U1d_speed": 0.0,
            "U1d_temperature": 0.0,
            "Y2d_speed": 0.0,
            "Y2d_temperature": 0.0,
        }

        shdr_data = {
            "general": general_data,
            "avail": "AVAILABLE",
            # "X_axis": X_axis_data,
            # "Y_axis": Y_axis_data,
            # "Z_axis": Z_axis_data,
            "C_axis": C_axis_data,
            # "spindle": spindle_data,

        }

        return shdr_data

    def get_scenario_data(self, current_scenario_index):
        try:
            return self.scenarios[current_scenario_index]
        except IndexError:
            raise ValueError(f"No scenario found with index: {current_scenario_index}")

    def setup_logging(self, name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def signal_handler(self, sig, frame):
        self.running = False
        self.server_logger.debug("Pressed Ctrl+C")
        print("Pressed Ctrl+C")

    def calculate_min_max_values(self):
        variables = [
            "temperature", "speed", "load",
            "current", "pos", "abs", "frt"
        ]

        min_max_values = {}

        for variable in variables:
            min_max_values[variable] = {"min": float('inf'), "max": float('-inf')}

        for scenario in self.scenarios:
            for variable in variables:
                current_range = scenario["variables"][variable]["normal"]
                min_max_values[variable]["min"] = min(min_max_values[variable]["min"], current_range[0])
                min_max_values[variable]["max"] = max(min_max_values[variable]["max"], current_range[1])

        return min_max_values

    def simulate_cnc_running(self, simulation_interval: float) -> None:
        self.send_all_data = True

        start_time = time.time()
        self.server_logger.debug(f"simulation started")
        print("Simulation started")
        if not self._anomaly_flag:
            self.current_scenario_index = 0

        while self.running and self.send_all_data:
            current_scenario = self.scenarios[self.current_scenario_index]
            scenario_start_time = time.time()

            # If the current scenario is not "Active", set the rotary_mode to INDEX and continue to the next iteration
            if current_scenario["name"] != "Active":
                self.shdr_data["C_axis"]["Crotary_mode"] = "INDEX"

            current_values = {var: current_scenario["variables"][var]["normal"][0] for var in
                              current_scenario["variables"]}
            direction = {var: "increase" for var in current_scenario["variables"]}
            step_index = {var: 0 for var in current_scenario["variables"]}
            toggle = True  # Variable to toggle between lower and upper values

            while time.time() - scenario_start_time < current_scenario["duration"]:
                self.shdr_data["general"]["execution"] = current_scenario["execution"]
                self.shdr_data["general"]["controller_mode"] = current_scenario["controller_mode"]

                self.shdr_data["general"]["line_number"] += 2
                self.shdr_data["general"]["deceleration"] -= 2

                # Simulate changes in the CNC machine's rotary_mode for C_axis based on the current scenario and
                # elapsed time
                if current_scenario["name"] == "Normal Operation":
                    elapsed_time = time.time() - scenario_start_time
                    if elapsed_time < 10:
                        self.shdr_data["C_axis"]["Crotary_mode"] = "SPINDLE"
                    elif 10 <= elapsed_time < 15:
                        self.shdr_data["C_axis"]["Crotary_mode"] = "CONTOUR"
                    elif 15 <= elapsed_time < 25:
                        self.shdr_data["C_axis"]["Crotary_mode"] = "SPINDLE"
                    elif 25 <= elapsed_time < 30:
                        self.shdr_data["C_axis"]["Crotary_mode"] = "CONTOUR"
                    elif 30 <= elapsed_time < 50:
                        self.shdr_data["C_axis"]["Crotary_mode"] = "SPINDLE"
                    elif 50 <= elapsed_time < 60:
                        self.shdr_data["C_axis"]["Crotary_mode"] = "INDEX"
                for axis_data, axis_prefix in zip([self.shdr_data["C_axis"]], ['C']):
                    for var in current_scenario["variables"]:
                        if current_scenario["name"] != "FEED_HOLD":
                            delta = current_scenario["variables"][var]["steps"][step_index[var]]
                            min_val, max_val = current_scenario["variables"][var]["normal"]
                            if direction[var] == "increase":
                                current_values[var] += delta
                                if current_values[var] >= max_val:
                                    current_values[var] = max_val
                                    direction[var] = "decrease"
                                    step_index[var] = (step_index[var] + 1) % len(
                                        current_scenario["variables"][var]["steps"])
                            else:
                                current_values[var] -= delta
                                if current_values[var] <= min_val:
                                    current_values[var] = min_val
                                    direction[var] = "increase"
                                    step_index[var] = (step_index[var] + 1) % len(
                                        current_scenario["variables"][var]["steps"])
                            axis_data[f"{axis_prefix}{var}"] = current_values[var]
                        else:
                            index = 0 if toggle else 1
                            axis_data[f"{axis_prefix}{var}"] = current_scenario["variables"][var]["normal"][index]

                self.shdr_data["general"]["process_timer"] += 1.0

                time.sleep(simulation_interval - 0.2)
                if not self.send_all_data:
                    break

                if current_scenario["name"] == "FEED_HOLD":
                    toggle = not toggle

            if self.send_all_data:
                self.current_scenario_index = (self.current_scenario_index + 1) % len(self.scenarios)

        self.send_all_data = False
        self.server_logger.debug(f"simulation ended")

    def update_axes_data(self) -> None:
        axes = ['X', 'Y', 'Z', 'C', 'Exit to main menu']
        axis_choices = [inquirer.List('axis', message="Which axis would you like to update?", choices=axes)]
        axis = inquirer.prompt(axis_choices)['axis']

        if axis == 'Exit to main menu':
            return

        axis_data = self.shdr_data[f"{axis}_axis"]
        keys = list(axis_data.keys())
        variable_choices = [inquirer.List('variable', message="Which variable would you like to update?",
                                          choices=keys + ['Raise Anomaly', 'Exit to main menu'])]
        variable = inquirer.prompt(variable_choices)['variable']

        if variable == 'Exit to main menu':
            return
        if variable.endswith('travel'):
            travel_conditions = ["NORMAL", "WARNING", "FAULT"]
            travel_choices = [inquirer.List('travel', message="Choose a travel condition",
                                            choices=travel_conditions + ['Exit to main menu'])]
            value = inquirer.prompt(travel_choices)['travel']
            if value == 'Exit to main menu':
                return
        elif variable.endswith('rotary_mode'):
            rotary_modes = ["SPINDLE", "INDEX"]
            rotary_choices = [inquirer.List('rotary_mode', message="Choose a rotary mode",
                                            choices=rotary_modes + ['Exit to main menu'])]
            value = inquirer.prompt(rotary_choices)['rotary_mode']
            if value == 'Exit to main menu':
                return
        elif variable.endswith('axis_state'):
            axis_states = ["ACTIVE", "INACTIVE"]
            axis_state_choices = [inquirer.List('axis_state', message="Choose an axis state",
                                                choices=axis_states + ['Exit to main menu'])]
            value = inquirer.prompt(axis_state_choices)['axis_state']
            if value == 'Exit to main menu':
                return
        else:
            update_type_choices = [
                inquirer.List('update_type', message="Do you want to update a single value or a range?",
                              choices=['Single', 'Range', 'Exit to main menu'])
            ]
            update_type = inquirer.prompt(update_type_choices)['update_type']

            if update_type == 'Exit to main menu':
                return

            if update_type == 'Single':
                value = self.get_float_input(f"Enter the new value for {variable}: ")
            else:  # Range
                start = self.get_float_input(f"Enter the start value for {variable}: ")
                end = self.get_float_input(f"Enter the end value for {variable}: ")
                step = self.get_float_input(f"Enter the step value: ")
                interval = self.get_float_input(f"Enter the interval in seconds: ")

                while True:
                    random_order_input = input(
                        "Do you want to update the values in random order? (yes/no): ").strip().lower()
                    if random_order_input in ("yes", "no"):
                        random_order = random_order_input == 'yes'
                        break
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")

                values = list(np.arange(start, end, step))
                if random_order:
                    random.shuffle(values)

                self.update_interval = interval
                time.sleep(1)
                for value in values:
                    if not self.running:
                        break
                    axis_data[variable] = value
                    self.updated_data[variable] = value
                    print(f"Updated {variable} to {value}")
                    time.sleep(interval)
                self.update_interval = self.simulation_interval
                return

        axis_data[variable] = value
        self.updated_data[variable] = value

    def raise_anomaly(self) -> None:
        self.stop_simulation()

        # Directly use the AnomalyType list for choices
        anomaly_choices = [inquirer.List('anomaly_type', message="Which type of anomaly would you like to raise?",
                                         choices=AnomalyType)]
        selected_anomaly_type_str = inquirer.prompt(anomaly_choices)['anomaly_type']

        # Get the index of the selected anomaly type from the AnomalyType list
        selected_anomaly_type_index = AnomalyType.index(selected_anomaly_type_str)

        selected_variable = "Cabs"

        anomaly_handlers = {
            0: lambda: self.handle_abnormal_high_value(selected_variable),
            1: lambda: self.handle_abnormal_low_value(selected_variable),
            2: lambda: self.handle_slow_rate_of_change_in_acc(selected_variable),
            3: lambda: self.handle_fast_rate_of_change_in_acc(selected_variable),
            4: lambda: self.handle_slow_rate_of_change_in_dec(selected_variable),
            5: lambda: self.handle_fast_rate_of_change_in_dec(selected_variable),
            6: self.handle_unseen_deceleration,
            7: self.handle_unseen_acceleration,
            8: self.handle_unseen_data_change,
            9: lambda: self.handle_slow_chg_trans_duration(),
            10: lambda: self.handle_fast_chg_trans_duration(),
            11: lambda: self.handle_unseen_transition(),
            12: lambda: self.handle_correlated_abnormal_high_value(selected_variable),
            13: lambda: self.handle_correlated_abnormal_low_value(selected_variable),
            14: lambda: self.handle_correlated_value_increased_slower_than_expected_during_transition(
                selected_variable),
            15: lambda: self.handle_correlated_value_increased_faster_than_expected_during_transition(
                selected_variable),
            16: lambda: self.handle_correlated_value_decreased_slower_than_expected_during_transition(
                selected_variable),
            17: lambda: self.handle_correlated_value_decreased_faster_than_expected_during_transition(selected_variable)
        }

        handler = anomaly_handlers.get(selected_anomaly_type_index)
        if handler:
            handler()  # This will execute the lambda function or method

    def _sleep(self, duration: float) -> None:
        whole_seconds = int(duration)
        fractional_seconds = duration - whole_seconds

        for _ in range(whole_seconds):
            if not self.running:
                break
            time.sleep(1)

        if fractional_seconds > 0:
            time.sleep(fractional_seconds)

    def handle_abnormal_high_value(self, selected_variable):
        self._anomaly_flag = True
        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        scenario_variable_name = selected_variable[1:]
        deviation = abs(self.min_max_values[scenario_variable_name]['max'] -
                        self.min_max_values[scenario_variable_name]['min']) * 0.15

        print(f"************** creating anomaly  **********************")
        self.updated_data[selected_variable] = self.min_max_values[scenario_variable_name]['max'] + deviation
        print(f'{selected_variable}:{self.updated_data[selected_variable]}')
        self._anomaly_flag = False

    def handle_abnormal_low_value(self, selected_variable):
        self._anomaly_flag = True
        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        scenario_variable_name = selected_variable[1:]
        deviation = abs(self.min_max_values[scenario_variable_name]['max'] -
                        self.min_max_values[scenario_variable_name]['min'])

        print(f"************** creating anomaly  **********************")
        self.updated_data[selected_variable] = self.min_max_values[scenario_variable_name]['min'] - deviation
        print(f'{selected_variable}:{self.updated_data[selected_variable]}')
        self._anomaly_flag = False

    def handle_slow_rate_of_change_in_acc(self, selected_variable):
        self._anomaly_flag = True
        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        scenario_variable_name = selected_variable[1:]
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        steps = variable_data["steps"]
        min_val, max_val = variable_data["normal"]
        act_val = min_val
        self.updated_data[selected_variable] = act_val
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        time.sleep(2)
        steps = sorted(steps)
        print("****************** creating anomaly ***************************")
        sleep_dur = 20

        for step in steps:
            print(f"{sleep_dur + step} seconds sleep....")
            self._sleep(sleep_dur + step)
            act_val = act_val + step
            self.updated_data[selected_variable] = act_val
            print(f"{self.get_human_readable_timestamp()} "
                  f"{selected_variable}:{self.updated_data[selected_variable]}")
        self._anomaly_flag = False

    def handle_fast_rate_of_change_in_acc(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        scenario_variable_name = selected_variable[1:]
        print("****************** creating anomaly ***************************")
        self.updated_data[selected_variable] = self.min_max_values[scenario_variable_name]['min']
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        self._sleep(2)
        anomaly_value = self.min_max_values[scenario_variable_name]['max']
        if self.min_max_values[scenario_variable_name]['max'] > 1:
            anomaly_value = self.min_max_values[scenario_variable_name]['max'] - 2
        self.updated_data[selected_variable] = anomaly_value
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        self._anomaly_flag = False

    def handle_slow_rate_of_change_in_dec(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        scenario_variable_name = selected_variable[1:]
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        steps = variable_data["steps"]
        min_val, max_val = variable_data["normal"]
        act_val = min_val
        self.updated_data[selected_variable] = act_val
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        time.sleep(2)
        steps = sorted(steps)
        print("****************** creating anomaly ***************************")
        sleep_dur = 20

        for step in steps:
            print(f"{sleep_dur + step} seconds sleep....")
            self._sleep(sleep_dur + step)
            act_val = act_val - step
            self.updated_data[selected_variable] = act_val
            print(f"{self.get_human_readable_timestamp()} "
                  f"{selected_variable}:{self.updated_data[selected_variable]}")
        self._anomaly_flag = False

    def handle_fast_rate_of_change_in_dec(self, selected_variable):
        self._anomaly_flag = True
        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 60 seconds")
        self._sleep(20)
        self.stop_simulation()

        scenario_variable_name = selected_variable[1:]
        print("****************** creating anomaly ***************************")
        self.updated_data[selected_variable] = self.min_max_values[scenario_variable_name]['max']
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        self._sleep(2)
        anomaly_value = self.min_max_values[scenario_variable_name]['min']
        if self.min_max_values[scenario_variable_name]['min'] > 1:
            anomaly_value = self.min_max_values[scenario_variable_name]['min'] - 2
        self.updated_data[selected_variable] = anomaly_value
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        self._anomaly_flag = False

    def handle_correlated_abnormal_high_value(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        print(f"{current_scenario_data=}")
        scenario_variable_name = selected_variable[1:]
        print(f'execution:{current_scenario_data["execution"]}')
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        if not variable_data:
            raise ValueError(f"No variable found with name: {scenario_variable_name}")

        min_val, max_val = variable_data["normal"]
        print(f"{max_val=}")
        deviation = max_val * 0.15
        print(f'{selected_variable}:{max_val + deviation}')
        self.updated_data[selected_variable] = max_val + deviation
        self._sleep(5)
        next_state = self.execution_map[current_scenario_data["execution"]]
        self.updated_data["execution"] = next_state
        print(f'execution:{self.updated_data["execution"]}')
        self._anomaly_flag = False

    def handle_correlated_abnormal_low_value(self, selected_variable):
        self._anomaly_flag = True
        self.current_scenario_index = self.execution_index_map["MAINTENANCE"]
        self.start_simulation()
        self._sleep(30)
        self.stop_simulation()
        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        print(f"{current_scenario_data=}")
        scenario_variable_name = selected_variable[1:]
        print(f'execution:{current_scenario_data["execution"]}')
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        if not variable_data:
            raise ValueError(f"No variable found with name: {scenario_variable_name}")

        min_val, max_val = variable_data["normal"]
        deviation = min_val * 0.20
        print(f'{selected_variable}:{min_val - deviation}')
        self.updated_data[selected_variable] = min_val - deviation
        self._sleep(5)
        next_state = self.execution_map[current_scenario_data["execution"]]
        self.updated_data["execution"] = next_state
        print(f'execution:{self.updated_data["execution"]}')
        self._anomaly_flag = False

    def handle_correlated_value_increased_slower_than_expected_during_transition(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["MAINTENANCE"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        scenario_variable_name = selected_variable[1:]
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        steps = variable_data["steps"]
        min_val, max_val = variable_data["normal"]
        act_val = min_val
        self.updated_data[selected_variable] = act_val
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        time.sleep(2)
        steps = sorted(steps)
        print("****************** creating anomaly ***************************")
        sleep_dur = 6

        for step in steps:
            print(f"{sleep_dur + step} seconds sleep....")
            self._sleep(sleep_dur + step)
            act_val = act_val + (step + 1)
            self.updated_data[selected_variable] = act_val
            print(f"{self.get_human_readable_timestamp()} "
                  f"{selected_variable}:{self.updated_data[selected_variable]}")
        self._sleep(5)
        next_state = self.execution_map[current_scenario_data["execution"]]
        self.updated_data["execution"] = next_state
        print(f'execution:{self.updated_data["execution"]}')
        self._anomaly_flag = False

    def handle_correlated_value_increased_faster_than_expected_during_transition(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["MAINTENANCE"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        scenario_variable_name = selected_variable[1:]
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        steps = variable_data["steps"]
        min_val, max_val = variable_data["normal"]
        act_val = min_val
        self.updated_data[selected_variable] = act_val
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        time.sleep(2)
        steps = sorted(steps)
        print("****************** creating anomaly ***************************")
        sleep_dur = 5

        for step in steps:
            print(f"{sleep_dur + step} seconds sleep....")
            self._sleep(sleep_dur + step)
            act_val = act_val + (step + 1)
            self.updated_data[selected_variable] = act_val
            print(f"{self.get_human_readable_timestamp()} "
                  f"{selected_variable}:{self.updated_data[selected_variable]}")
        self._sleep(5)
        next_state = self.execution_map[current_scenario_data["execution"]]
        self.updated_data["execution"] = next_state
        print(f'execution:{self.updated_data["execution"]}')
        self._anomaly_flag = False

    def handle_correlated_value_decreased_faster_than_expected_during_transition(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["MAINTENANCE"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        scenario_variable_name = selected_variable[1:]
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        steps = variable_data["steps"]
        min_val, max_val = variable_data["normal"]
        act_val = min_val
        act_val += 20
        self.updated_data[selected_variable] = act_val
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        steps = sorted(steps)
        print("****************** creating anomaly ***************************")
        for step in steps:
            self._sleep(2)
            act_val = act_val - (step * 2)
            self.updated_data[selected_variable] = act_val
            print(f"{self.get_human_readable_timestamp()} "
                  f"{selected_variable}:{self.updated_data[selected_variable]}")
        self._sleep(5)
        next_state = self.execution_map[current_scenario_data["execution"]]
        self.updated_data["execution"] = next_state
        print(f'execution:{self.updated_data["execution"]}')
        self._anomaly_flag = False

    def handle_correlated_value_decreased_slower_than_expected_during_transition(self, selected_variable):
        self._anomaly_flag = True

        self.current_scenario_index = self.execution_index_map["MAINTENANCE"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()

        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        scenario_variable_name = selected_variable[1:]
        variable_data = current_scenario_data["variables"].get(scenario_variable_name)
        steps = variable_data["steps"]
        min_val, max_val = variable_data["normal"]
        act_val = min_val
        self.updated_data[selected_variable] = act_val
        print(f"{self.get_human_readable_timestamp()} {selected_variable}:{self.updated_data[selected_variable]}")
        time.sleep(2)
        steps = sorted(steps)
        print("****************** creating anomaly ***************************")
        sleep_dur = 6

        for step in steps:
            print(f"{sleep_dur + step} seconds sleep....")
            self._sleep(10)
            act_val = act_val - 3
            self.updated_data[selected_variable] = act_val
            print(f"{self.get_human_readable_timestamp()} "
                  f"{selected_variable}:{self.updated_data[selected_variable]}")
        self._sleep(5)
        next_state = self.execution_map[current_scenario_data["execution"]]
        self.updated_data["execution"] = next_state
        print(f'execution:{self.updated_data["execution"]}')
        self._anomaly_flag = False

    def handle_unseen_deceleration(self):
        self.updated_data["line_number"] = 50
        print(f"{self.get_human_readable_timestamp()} line_number:50")
        self._sleep(2)
        print(f"{self.get_human_readable_timestamp()} line_number:46")
        self.updated_data["line_number"] = 46

    def handle_unseen_acceleration(self):
        self.updated_data["deceleration"] = 50
        print(f"{self.get_human_readable_timestamp()} deceleration:50")
        self._sleep(2)
        print(f"{self.get_human_readable_timestamp()} deceleration:55")
        self.updated_data["deceleration"] = 55

    def handle_unseen_data_change(self):
        print(f"{self.get_human_readable_timestamp()} tool_number:50")
        self.updated_data["tool_number"] = 50

    def handle_slow_chg_trans_duration(self):
        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()
        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        next_state = self.execution_map[current_scenario_data["execution"]]
        print(f"{current_scenario_data['execution']=} {next_state=} ")
        print(f"{self.get_human_readable_timestamp()} "
              f"execution:{current_scenario_data['execution']}")
        print(f"{current_scenario_data['duration']} seconds sleep....")
        self._sleep(current_scenario_data['duration'])
        print(f"{self.get_human_readable_timestamp()} "
              f"execution:{next_state}")
        self.updated_data["execution"] = next_state

    def handle_fast_chg_trans_duration(self):
        self.current_scenario_index = self.execution_index_map["READY"]
        self.start_simulation()
        print(f"running the simulation for 30 seconds")
        self._sleep(30)
        self.stop_simulation()
        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        next_state = self.execution_map[current_scenario_data["execution"]]
        print(f"{current_scenario_data['execution']=} {next_state=} ")
        print(f"{self.get_human_readable_timestamp()} "
              f"execution:{current_scenario_data['execution']}")
        print(f"10 seconds sleep....")
        self._sleep(10)
        print(f"{self.get_human_readable_timestamp()} "
              f"execution:{next_state}")
        self.updated_data["execution"] = next_state

    def handle_unseen_transition(self):
        self.start_simulation()
        self._sleep(10)
        self.stop_simulation()
        print("****************** creating anomaly ***************************")
        current_scenario_data = self.get_scenario_data(self.current_scenario_index)
        next_state = self.non_transition_map[current_scenario_data["execution"]]
        print(f"{current_scenario_data['execution']=} {next_state=} ")
        self._sleep(2)
        self.updated_data["execution"] = next_state

    def get_human_readable_timestamp(self) -> str:
        """Return the current timestamp in a human-readable format with milliseconds."""
        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # includes milliseconds
        return formatted_timestamp

    def get_float_input(self, prompt: str) -> float:
        while True:
            try:
                value = float(input(prompt))
                break
            except ValueError:
                print("Invalid input. Please enter a valid float value.")
        return value

    def get_int_input(self, prompt: str) -> int:
        while True:
            try:
                value = int(input(prompt))
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer value.")
        return value

    def format_shdr_data(self, data: Dict[str, Any]) -> str:
        """Formats the SHDR data dictionary into a string."""
        parts = []
        for key, value in data.items():
            if isinstance(value, dict):
                nested_parts = [f"{k}|{v}" for k, v in value.items()]
                parts.extend(nested_parts)
            else:
                parts.append(f"{key}|{value}")
        return '|'.join(parts)

    def heartbeat_protocol(self) -> None:

        while self.running:
            for conn, addr in list(self.clients):
                try:
                    conn.settimeout(2)
                    data = conn.recv(1024)
                    if data:
                        msg = data.decode('utf-8').strip()
                        if msg == '* PING':
                            response = '* PONG 10000\n'
                            conn.sendall(response.encode('utf-8'))
                            self.server_logger.info(f"Received PING from {addr}, responded with PONG")
                            # Reset the client's ping counter to the default
                            # value if PING received.
                            self.client_ping_monitor[addr] = 10
                        else:
                            self.server_logger.info(f"Received {msg} from {addr}.")
                    else:
                        self.server_logger.info(f"No data received from {addr}")
                except socket.timeout:
                    # Decrement the client's ping counter on timeout
                    self.client_ping_monitor[addr] -= 1

                    # Remove the client if more than 10 consecutive timeouts happen.
                    if self.client_ping_monitor[addr] <= 0:
                        self.server_logger.debug(f"Received no PING from {addr} for a long time. Removing {addr}.")
                        self.clients.remove((conn, addr))
                # self.server_logger.info(f"Socket timeout: {addr}")
                except socket.error:
                    self.server_logger.error(f"Connection with {addr} lost.")
                    if (conn, addr) in self.clients:
                        self.clients.remove((conn, addr))
                    conn.close()
            time.sleep(1)  # Adjust the sleep time as necessary

    def send_data_to_clients(self, shdr_string):
        clients_to_remove = []

        for conn, addr in self.clients:
            try:
                conn.sendall(shdr_string.encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError, socket.error) as e:
                self.server_logger.error(f"Tried to send data to client {addr}: \
{str(e)}. But connection already closed.")
                clients_to_remove.append((conn, addr))
            except Exception as e:
                self.server_logger.error(f"Something unusual happened: {str(e)}")
                clients_to_remove.append((conn, addr))

        for conn, addr in clients_to_remove:
            self.server_logger.info(f"Client {addr} removed from the list")
            self.clients.remove((conn, addr))
            conn.close()

    def handle_client(self) -> None:
        try:
            while self.running:
                s_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                if self.send_all_data:
                    shdr_string = s_str + '|' + self.format_shdr_data(self.shdr_data) + '\n'
                    self.server_logger.info(f"{shdr_string}")
                    self.send_data_to_clients(shdr_string)
                    time.sleep(self.simulation_interval)

                if self.updated_data:
                    shdr_string = s_str + '|' + self.format_shdr_data(self.updated_data) + '\n'
                    self.updated_data = {}
                    self.server_logger.info(f"{shdr_string}")
                    self.send_data_to_clients(shdr_string)
                else:
                    time.sleep(self.update_interval)
                    continue


        except Exception as e:
            self.server_logger.error(traceback.format_exc())

    def start_simulation(self):
        if self.simulation_thread is None or not self.simulation_thread.is_alive():
            self.simulation_thread = Thread(target=self.simulate_cnc_running, args=(self.simulation_interval,))
            self.simulation_thread.start()
        else:
            print("Simulation is already running.")

    def stop_simulation(self):
        if self.simulation_thread is not None and self.simulation_thread.is_alive():
            self.send_all_data = False
            self.simulation_thread.join()
            self.simulation_thread = None
            print("Simulation stopped.")
        else:
            print("No simulation is running.")

    def server_thread(self) -> None:
        self.server_logger.info(f"Running status: {self.running}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                server_socket.bind(('0.0.0.0', 7878))
                server_socket.listen()
                server_socket.settimeout(5)

                # Start the heartbeat protocol thread
                heartbeat_thread = Thread(target=self.heartbeat_protocol)
                heartbeat_thread.start()

                while self.running:
                    try:
                        conn, addr = server_socket.accept()
                        self.server_logger.info(f"Connected by {addr}")
                        self.clients.append((conn, addr))
                    except socket.timeout:
                        if not self.running:
                            break
                        continue
                    except Exception as e:
                        self.server_logger.error(f"An error occurred while accepting a connection: {e}")
                        continue

            except Exception as e:
                self.server_logger.error(traceback.format_exc())
            finally:
                for conn, addr in self.clients:
                    self.server_logger.debug(f"closing the connection {addr}")
                    conn.close()

                # Ensure the heartbeat thread is stopped when the server stops
                if heartbeat_thread.is_alive():
                    heartbeat_thread.join()

    def main_menu(self) -> None:
        try:
            choices = [
                "Update individual variable",
                "Raise Anomaly",
                "Change simulation interval",
                "Run the simulation",
                "Stop the simulation",
                "Exit"
            ]
            questions = [
                inquirer.List('main_menu', message="What would you like to do?", choices=choices),
            ]
            answer = inquirer.prompt(questions)
            if answer is None:
                print("No option selected, exiting.")
                self.running = False
                exit()
            else:
                answer = answer['main_menu']

            if answer == "Run the simulation":
                if self.simulation_thread is None or not self.simulation_thread.is_alive():
                    self.simulation_thread = Thread(target=self.simulate_cnc_running, args=(self.simulation_interval,))
                    self.simulation_thread.start()
                else:
                    print("Simulation is already running.")
            elif answer == "Stop the simulation":
                if self.simulation_thread is not None and self.simulation_thread.is_alive():
                    self.send_all_data = False
                    self.simulation_thread.join()
                    self.simulation_thread = None
                    print("Simulation stopped.")
                else:
                    print("No simulation is running.")
            elif answer == "Update individual variable":
                self.update_axes_data()
            elif answer == 'Raise Anomaly':
                self.raise_anomaly()
            elif answer == "Change simulation interval":
                self.simulation_interval = self.get_float_input("Enter the new simulation interval (in seconds): ")
            elif answer == "Exit":
                self.running = False
                exit()
        except KeyboardInterrupt:
            print("\nReceived Ctrl+C. Exiting...")
            self.running = False
            if self.simulation_thread is not None and self.simulation_thread.is_alive():
                self.send_all_data = False
                self.simulation_thread.join()
            exit()

    def is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except socket.error:
                return False
        return True

    def main(self) -> None:
        port = 7878
        if not self.is_port_available(port):
            print(f"Could not start the server because port {port} is already in use.")
            return
        server_thread_instance = Thread(target=self.server_thread)
        server_thread_instance.start()
        client_thread = Thread(target=self.handle_client)
        client_thread.start()

        while self.running:
            self.main_menu()

        client_thread.join()
        server_thread_instance.join()


if __name__ == "__main__":
    CNCServer().main()
