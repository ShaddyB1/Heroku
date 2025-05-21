import os
import multiprocessing

# Worker settings
workers = 1
threads = 2
worker_class = 'gthread'

# Optimize for memory efficiency
worker_tmp_dir = '/dev/shm'
worker_connections = 100
timeout = 180

# Restart workers to free up memory
max_requests = 5
max_requests_jitter = 2

# Preload app to share memory
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

bind = "0.0.0.0:10000"
