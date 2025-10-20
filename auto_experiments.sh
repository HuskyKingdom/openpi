#!/bin/bash


xhost +local:docker

# without energy
echo "Running Evaluations Automatically ------------------------------"



echo "Evaluatiing Spatial ------------------------------"

sudo MUJOCO_GL=osmesa SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_energy --policy.dir checkpoints/pi05_libero_energy/my_experiment/29999/" CLIENT_ARGS="--args.task-suite-name libero_spatial" docker compose -f examples/libero/compose.yml up --build