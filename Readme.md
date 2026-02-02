In the folder hydrophone_placement_scripts there are some notebooks and to handle them with git, you have to transform them in classic python file, this is done automaticly after each commit and they are also transform back after each merge, but after cloning the repository you have to run :
```
./setup_hooks.sh
```

Then to ensure all modules are installed, please run

```
pip install -r requirements.txt
```

The folder beluga_watch is for the geolocalisation of the belugas made by Paul Ducroq and Paul Laurent and the folder hydrophone_placement_scripts is for the localisation of the hydrophones and some work for the prediction of the trajectories (Valentin Allard). In each folder you will find another readme which explains the folder 