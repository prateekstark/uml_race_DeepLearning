gnome-terminal -e 'roslaunch uml_race racetrack.launch'
sleep 4
python uml_race_solver_DQN.py
sleep 10
