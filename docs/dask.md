# Dask Cluster

## How to connect and start the cluster

### Prerequistes:
- set up your static ip address to be `192.168.2.1/24` - this will work as default gateway for the clusters
- enable NAT from ethernet port to wireless interface, this means that the jetsons can connect to the internet through the control machine
- connect physically to the same switch that the jetsons are connected to (do not connect to the first port in the switch)

### Connect 
- SSH into the machines
- password for all machines is: `glacing-diffused-sports-overhand`

```shell
# SCHEDULER
ssh jetson@192.168.2.2 

# WORKER 1
ssh jetson@192.168.2.4 

# OTHER WORKERS (set static ip for them to be in the same subnet)
ssh jetson@192.168.2.X 
```

### Start nodes

#### start the scheduler 
- it should automatically  create a virtual environment, download the dependencies and set up the scheduler on `192.168.2.2:8786`

```shell
### SCHEDULER

# enter the directory 
cd /semester-project-group-13/cluster-node

# start the scheduler 
./dask_manager.sh scheduler
```

#### start the workers
- it should automatically do the same things as previously but connect to the scheduler on the same address


```shell
### WORKER

# enter the directory 
cd /semester-project-group-13/cluster-node

# start the worker 
./dask_manager.sh worker
```

## Docs regarding Dask - distributed cluster

- https://distributed.dask.org/en/stable/install.html
- https://docs.dask.org/en/latest/deploying.html
- https://docs.dask.org/en/latest/deploying-python.html
- https://docs.dask.org/en/latest/deploying-cli.html