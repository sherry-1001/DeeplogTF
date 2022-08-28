````
unset http_proxy && unset https_proxy
taskset -c 0-1 python3 tf2-example-fl-distribute.py --job_name ps --task_index 0 | tee -a ps0-python.log
taskset -c 8-9 python3 tf2-example-fl-distribute.py --job_name worker --task_index 0 | tee -a worker0-python.log
taskset -c 12-13 python3 tf2-example-fl-distribute.py --job_name worker --task_index 1 | tee -a worker1-python.log
````
