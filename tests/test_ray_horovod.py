import socket
import ray

import horovod.tensorflow.keras as hvd
# import horovod.tensorflow as hvd
from horovod.ray import RayExecutor

# Start the Ray cluster or attach to an existing Ray cluster
ray.init(address="auto")

# Ray executor settings
setting = RayExecutor.create_settings(timeout_s=100)
num_hosts = 1  # number of machine to use
num_slots = 6  # number of workers to use on each machine
cpus_per_slot = 1  # number of cores to allocate to each worker
gpus_per_slot = 1 # number of GPUs to allocate to each worker

# Start num_hosts * num_slots actors on the cluster
# https://horovod.readthedocs.io/en/stable/api.html#horovod-ray-api
executor = RayExecutor(
    setting,
    num_hosts=num_hosts,
    num_slots=num_slots,
    cpus_per_slot=cpus_per_slot,
    gpus_per_slot=gpus_per_slot,
    use_gpu=True
)

# Launch the Ray actors on each machine
# This will launch `num_slots` actors on each machine
print("Start executor...", end="", flush=True)
executor.start()
print("OK", flush=True)


#! Note that there is an implicit assumption on the cluster being
#! homogenous in shape (i.e., all machines have the same number of
#! slots available). This is simply an implementation detail and is
#! not a fundamental limitation.

# Using the stateless `run` method, a function can take in any args or kwargs
def simple_fn():


    hvd.init()

    rank = hvd.rank()


    ## getting the hostname by socket.gethostname() method
    hostname = socket.gethostname()
    ## getting the IP address using socket.gethostbyname() method
    ip_address = socket.gethostbyname(hostname)

    print(f"hvd rank[{ip_address}]", rank)

    return rank


# Execute the function on all workers at once
result = executor.run(simple_fn)
print(result)

# Check that the rank of all workers is unique
assert len(set(result)) == num_hosts * num_slots

executor.shutdown()