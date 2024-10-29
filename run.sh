# submit
sbatch submit.sh

# check submission history
squeue -u gyang16

# real-time console output
tail -f gyang16.log

# real-time GPU usage
tail -f -n 30 gpu_usage.log