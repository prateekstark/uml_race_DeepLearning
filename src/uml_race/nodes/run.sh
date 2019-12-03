gnome-terminal -e 'roslaunch uml_race racetrack.launch'
sleep 5
gnome-terminal -e 'python ~/race_ws/src/solver/uml_race_solver_rl.py'
