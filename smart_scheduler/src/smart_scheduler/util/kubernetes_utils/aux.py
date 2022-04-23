from .utils import (
    get_node_capacity,
    get_node_name,
    get_pod_name
)
from kubernetes.client import V1Node, V1Pod, V1Service
from smart_scheduler.util import logger


class ResourceUsage:
    """Resource Usage of Node/Pod"""
    def __init__(self, usage: dict):
        """ResourceUsage
            Used resources by a Pod / Node

        :param usage: dict
            required keys: cpu, memory
        """
        try:
            # store usage dictionary
            self.usage = usage

            # remove n from last of cpu
            cpu = usage.get('cpu')
            self.cpu = int(cpu[:-1] if 'n' in cpu else cpu)

            # remove Ki from last of memory
            memory = usage.get('memory')
            if 'Ki' in memory or 'Mi' in memory:
                memory = memory[:-2]
            self.memory = int(memory)
        except Exception as e:
            logger.error(e)
            exit(-1)

    def __str__(self):
        return "Usage(cpu='{}n', memory='{}Ki')".format(
            self.cpu, self.memory
        )


class Node:
    """Node Descriptor"""

    def __init__(self, id: str, node: V1Node, is_auxiliary: bool = False):
        """Constructor of Node Descriptor

        :param node: V1Node
            node object
        """
        self.id = id

        # is auxiliary node
        self.is_auxiliary = is_auxiliary

        # get name of node
        self.name = get_node_name(node)

        # get capacity of node
        capacity = get_node_capacity(node)

        # extract memory of node
        # remove Ki from last of memory
        memory = capacity.get('memory')
        self.memory = int(memory[:-2] if 'Ki' in memory else memory)

        # extract number of cpu of node
        self.cpu = int(capacity.get('cpu'))

    def __str__(self):
        """Describe a Node by its details (name, capacity)"""
        return "Node(id='{}', name='{}', memory='{}Ki', cpu='{}', is_auxiliary={})".format(
            self.id, self.name, self.memory, self.cpu, self.is_auxiliary
        )


class Service:
    """Service Descriptor"""

    def __init__(self, id: str, pod: V1Pod, svc: V1Service):
        """Constructor of Service Descriptor

        **NOTE** each Service refers to one Pod, so it should not be confiused
            with concept of Pod and Service in Kubernetes.

        :param id
            ID of service

        :param pod: V1Pod
            pod object

        :param svc: V1Service
            service object
        """

        self.id = id

        # Pod
        self.pod = pod

        # svc
        self.svc = svc

        # container name
        self.container_name = get_pod_name(pod, source='container')

        # metadata name
        self.metadata_name = get_pod_name(pod, source='metadata')

        # Node name
        self.node_name = self.pod.spec.node_name

    def __str__(self):
        return "Service(id='{}', container_name='{}', metadata_name='{}', node_name='{}')".format(
            self.id, self.container_name, self.metadata_name, self.node_name
        )
