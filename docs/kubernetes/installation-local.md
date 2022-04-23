Installation
---

#### What is it ?
TODO: Explain about it

Setup Environment
---

We need the following VMs:
- 1 Controller (Control-Plane)
- 2 Workers (or more)

In this research we've used VirtualBox for setup the environment, so I'll describe how to configure the VMs.

### Controller Setup
_we've used Ubuntu 18.04 LTS for VMs_

##### Basic configurations
You need to define 2 network interfaces for each VM:
- Host-Only (enp0s3)
- NAT Network (enp0s8)

**NOTE:** It's important that VMs can ping each other over the NAT network. (Internal network)
_The name of interfaces may be a difference from mentioned_

following IPs should be set on `enp0s3` of each VM:
- Controller: `192.168.99.110`
- Worker 1: `192.168.99.11`
- Worker 2: `192.168.99.12`

**HELP**: You can use [`netplan`](https://linuxize.com/post/how-to-configure-static-ip-address-on-ubuntu-18-04/) in order to set IP addresses.

Install the following packages:
```bash
# update the repositories
sudo apt Update
```

###### Docker Installation

Kubernetes is prefered to use `systemd` as `cgroup Driver`, so we will configure it for docker ([Reference](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)):

```bash
sudo apt install -y \
  apt-transport-https ca-certificates curl software-properties-common gnupg2

# Add Docker's official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add the Docker apt repository:
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"

# Install Docker CE
sudo apt update && sudo apt install -y \
containerd.io=1.2.13-2 \
docker-ce=5:19.03.11~3-0~ubuntu-$(lsb_release -cs) \
docker-ce-cli=5:19.03.11~3-0~ubuntu-$(lsb_release -cs)
```

After install the Docker, we create a file to tell docker to use `systemd` as `cgroup driver`:
```bash
# Set up the Docker daemon
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

sudo mkdir -p /etc/systemd/system/docker.service.d

# Restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
```

Finally, you should see something like this after run the command `docker info`:
```text
ubuntu@kubernetes-master:~$ docker info
...snippet...
Server Version: 19.03.11
Storage Driver: overlay2
 Backing Filesystem: extfs
 Supports d_type: true
 Native Overlay Diff: true
Logging Driver: json-file
Cgroup Driver: systemd
 ...snippet...
```

###### Kubernetes Installation

Install Kubernetes packages:
```bash
# get repository Key
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# add repository
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF

# download packages
sudo apt update && sudo apt install -y kubelet kubeadm kubectl
```

we have to turn off the swap:
```bash
sudo swapoff -a && sudo sed -i 's/\/swap/#\/swap/g' /etc/fstab
```

###### **NOTE**: *_Because of common steps between a controller and workers, take a clone from VM for workers in this step._*


We need to set a hostname for controller node:
```bash
sudo hostnamectl set-hostname "kubernetes-master"
```

###### **IMPORTANT NOTE**: It's important to framework that using `master` for master node name.

Initialize Kubernetes (this process may take few minutes, it depends on your internet speed):
```bash
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.99.110 --v=10 | tee ~/kubeadm.log
```

Finally you should see something like this:
```text
...snippet...
Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:

  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.168.99.110:6443 --token <GENERATED_TOKEN> \
    --discovery-token-ca-cert-hash sha256:<GENERATED_HASH>
```
_you should use above link for joining any nodes to your cluster_

**NOTE:** You can change value of `--pod-network-cidr` and `--apiserver-advertise-address`.

**NOTE:** If you face any issues , you can run `sudo kubeadm reset` to remove previous configurations and start it again.

For finalize the installation process run the following commands:
```bash
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

You need to setup a network policy for your cluster:
```bash
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
```

### Workers Setup

**NOTE: After configure the network of each worker, follow the below steps.**

We need to set a hostname for each worker node:
```bash
sudo hostnamectl set-hostname "kubernetes-slave-X"
```
_replace `X` with number of slave_

It's time to join each worker to the cluster.
In order to do that, we run the following (it's generated at the initialization step of master node) command in each worker node:
```bash
sudo kubeadm join 192.168.99.110:6443 --token <GENERATED_TOKEN> --discovery-token-ca-cert-hash sha256:<GENERATED_HASH>
```

**NOTE:** If you face any issues , you can run `sudo kubeadm reset` to remove previous configurations and start it again.

##### Show nodes
Show final status of cluster:
```bash
ubuntu@kubernetes-master:~$ sudo kubectl get nodes
NAME                 STATUS   ROLES    AGE    VERSION
kubernetes-master    Ready    master   140m   v1.19.2
kubernetes-slave-1   Ready    <none>   132m   v1.19.2
kubernetes-slave-2   Ready    <none>   44m    v1.19.2
```

##### Kubernetes Cluster Config
You can find the config file of your cluster in the path `/etc/kubernetes/admin.conf` in the controller node.

**NOTE:** In order to use [Library](../README.md), you have to download it in your system with `scp` or every tools which you want.
```bash
# for example
scp [USER]@192.168.99.110:/etc/kubernetes/admin.conf .
```

Install Metics-Server
---

Go to [metrics-server](https://github.com/kubernetes-sigs/metrics-server/releases) github repository and download the last version of `components.yaml`.
Till now, you can download the last version with clicking this [link](https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.3.7/components.yaml).

**NOTE** DO NOT apply it on the cluster by the following link:
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```
so, download it in your machine and change some sections like as you see in below.

You have to change some specifications in Deployment section. You need to add an argument `- --kubelet-insecure-tls`
to `Deployment.spec.containers.[name=metrics-server].args` and property `hostNetwork: true` to container section:
```yaml
...snippet...
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    k8s-app: metrics-server
  name: metrics-server
  namespace: kube-system
spec:
  selector:
    matchLabels:
      k8s-app: metrics-server
  strategy:
    rollingUpdate:
      maxUnavailable: 0
  template:
    metadata:
      labels:
        k8s-app: metrics-server
    spec:
      containers:
      - args:
        - --cert-dir=/tmp
        - --secure-port=4443
        - --kubelet-preferred-address-types=InternalIP
        - --kubelet-use-node-status-port
        - --kubelet-insecure-tls
        hostNetwork: true
        image: k8s.gcr.io/metrics-server/metrics-server:v0.4.1
        imagePullPolicy: IfNotPresent
        livenessProbe:
          failureThreshold: 3
          httpGet:
            path: /livez
            port: https
            scheme: HTTPS
          periodSeconds: 10
        name: metrics-server
        ports:
        - containerPort: 4443
          name: https
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /readyz
            port: https
            scheme: HTTPS
          periodSeconds: 10
        securityContext:
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
        volumeMounts:
        - mountPath: /tmp
          name: tmp-dir
      nodeSelector:
        kubernetes.io/os: linux
      priorityClassName: system-cluster-critical
      serviceAccountName: metrics-server
      volumes:
      - emptyDir: {}
        name: tmp-dir
...snippet...
```

After change the file you need to apply it into the cluster:
```bash
kubectl apply -f components.yaml
```

now you can see the resource usage of each pod or nodes:
```bash
ubuntu@kubernetes-master:~$ kubectl top nodes
NAME                 CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%   
kubernetes-master    1381m        69%    1300Mi          63%       
kubernetes-slave-1   55m          2%     719Mi           35%       
kubernetes-slave-2   71m          3%     729Mi           35%
```

##### Node Labeling
In order to assign a pod to a specific worker node, you need to set label for each node what you want to do with:
```bash
ubuntu@kubernetes-master:~$ kubectl label node kubernetes-slave-1 node=slave-1
ubuntu@kubernetes-master:~$ kubectl label node kubernetes-slave-2 node=slave-2
```
