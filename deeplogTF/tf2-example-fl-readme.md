````
unset http_proxy && unset https_proxy
taskset -c 0-11 python3 tf2-example-fl-distribute.py --job_name ps --task_index 0
taskset -c 12-23 python3 tf2-example-fl-distribute.py --job_name worker --task_index 0
taskset -c 24-35 python3 tf2-example-fl-distribute.py --job_name worker --task_index 1
````