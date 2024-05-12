# examples/python/reconstruction_system/run_system.py

import json
import argparse
import time
import datetime
import os, sys
from os.path import isfile

import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import check_folder_structure

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from initialize_config import initialize_config, dataset_loader

import make_fragments
import register_fragments
import refine_registration
import integrate_scene
import slac
import slac_integrate

import matplotlib
matplotlib.use('MacOSX')


def run_reconstruction_system():

    config_json_file = None
    # config_json_file = 'config/realsense_l515.json'
    # default_dataset = 'lounge'
    # default_dataset = 'bedroom'
    default_dataset = 'jack_jack'
    # default_dataset = 'custom_realsense'
    debug_mode = False
    device = 'cpu:0'
    make = False  # make fragments
    register = False  # register fragments
    refine = True  # refine registration
    integrate = True  # integrate scene
    slac = False  # [Optional] Use --slac and --slac_integrate flags to perform SLAC optimisation.
    slac_integrate = False
    python_multi_threading = False

    if config_json_file is not None:
        with open(config_json_file) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            check_folder_structure(config['path_dataset'])
    else:
        # load default dataset.
        config = dataset_loader(default_dataset)

    assert config is not None

    config['debug_mode'] = debug_mode
    config['device'] = device
    config['python_multi_threading'] = python_multi_threading

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))


    times = [0, 0, 0, 0, 0, 0]
    if make:  # make fragments
        start_time = time.time()
        make_fragments.run(config)
        times[0] = time.time() - start_time
    if register:  # register fragments
        start_time = time.time()
        register_fragments.run(config)
        times[1] = time.time() - start_time
    if refine:  # refine registration
        start_time = time.time()
        refine_registration.run(config)
        times[2] = time.time() - start_time
    if integrate:  # integrate scene
        start_time = time.time()
        integrate_scene.run(config)
        times[3] = time.time() - start_time
    if slac:
        start_time = time.time()
        slac.run(config)
        times[4] = time.time() - start_time
    if slac_integrate:
        start_time = time.time()
        slac_integrate.run(config)
        times[5] = time.time() - start_time

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- SLAC                %s" % datetime.timedelta(seconds=times[4]))
    print("- SLAC Integrate      %s" % datetime.timedelta(seconds=times[5]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()


    pass


if __name__ == '__main__':

    run_reconstruction_system()

    pass