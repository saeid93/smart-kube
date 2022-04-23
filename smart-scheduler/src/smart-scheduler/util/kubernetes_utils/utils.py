from string import ascii_lowercase
from kubernetes.client import (
    V1Pod,
    V1Node,
    V1EnvVar,
    V1PodSpec,
    V1Container,
    V1ObjectMeta,
    V1ResourceRequirements,
    V1Service, V1ServiceSpec, V1ServicePort,
)
import random


def get_node_capacity(node: V1Node) -> dict:
    """Get Capacity of a Node

    :param node: V1Node
        node object

    :return dict
    """
    return node.status.capacity


def get_node_name(node: V1Node) -> str:
    """Get name of a Node

    :param node: V1Node
        node object

    :return str
    """
    return node.metadata.name


def get_pod_name(pod: V1Pod, source='container') -> str:
    """Get name of a Pod

    :param pod: V1Pod
        pod object

    :param source: str (default: container)
        valid sources: container, metadata

    :return str
    """
    if source == 'container':
        containers = pod.spec.containers
        if len(containers) > 0:
            return containers[0].name

    return pod.metadata.name


def get_service_name(service: V1Service) -> str:
    """Get name of a Service

    :param service: V1Service
        service object

    :return str
    """
    return service.metadata.name


def generate_random_service_name(
    service_id: int = 0, node_id = 0, size: int = 10) -> str:
    """Generate a random string

    :param size: int (default: 10)
        size of generated string
    
    format: s+service_id+n+node_id+(some random string)
    :return str
    """
    return 's' + str(service_id) + 'n' + str(node_id) +\
        '-'+\
            ''.join(random.choices(ascii_lowercase, k=size))


def construct_pod(
        name: str,
        image: str,
        node_name: str = None,
        labels: dict = None,
        namespace: str = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
        env: dict = None
) -> V1Pod:
    """Construct a Pod with a specific name in a determined node

    :param name: str
        name of pod

    :param image: str
        name of image (e.g. nginx)

    :param node_name: str (default: None)
        name of worker which you want to start the pod in

    :param labels: dict (default: {'env': 'park'})
        label of Pod

    :param namespace: str (default: 'consolidation')
        namespace of Pod

    :param request_mem: str (default: None)
        requested memory

    :param request_cpu: str (default: None)
        requested cpu

    :param limit_mem: str (default: None)
        limited memory

    :param limit_cpu: str (default: None)
        limited cpu

    :param env: dict (default: {'RAM': "80M"})
        using environmet for pods
    :return V1Pod
    """

    if labels is None:
        # set default value for labels
        labels = dict(
            env='park',
            svc=name
        )

    if namespace is None:
        # set default value for namespace
        namespace = 'consolidation'

    # set default value for env
    if env is None:
        # It works for `r0ot/stress:memory`
        env = dict(
            RAM="80M"
        )

    # prepare environment variables for Pod
    env = [
        V1EnvVar(name=key, value=value)
        for key, value in env.items()
    ]

    # limits, requests and environment variables
    limits, requests = dict(), dict()

    # assign limit memory
    if limit_mem is not None:
        limits.update(memory=limit_mem)

    # assign limit cpu
    if limit_cpu is not None:
        limits.update(cpu=limit_cpu)

    # assign requeste memory
    if request_mem is not None:
        requests.update(memory=request_mem)

    # assign request cpu
    if request_cpu is not None:
        requests.update(cpu=request_cpu)

    pod = V1Pod(
        api_version='v1',
        kind='Pod',
        metadata=V1ObjectMeta(
            name=name,
            labels=labels,
            namespace=namespace
        ),
        spec=V1PodSpec(
            hostname=name,
            containers=[
                V1Container(
                    name=name,
                    image=image,
                    env=env,
                    image_pull_policy='IfNotPresent',
                    resources=V1ResourceRequirements( # TODO check here
                        limits=limits,
                        requests=requests
                    )
                )
            ],
            node_name=node_name
        )
    )

    return pod


def construct_svc(
        name: str,
        namespace: str = None,
        labels: dict = None,
        portName: str = None,
        port: int = None,
        targetPort: int = None,
        portProtocol: str = None
):

    if labels is None:
        # set default value for labels
        labels = dict(
            env='park',
            svc=name
        )

    if namespace is None:
        # set default value for namespace
        namespace = 'consolidation'

    if portName is None:
        portName = 'web'

    if port is None:
        port = 80

    if targetPort is None:
        targetPort = 80

    if portProtocol is None:
        portProtocol = 'TCP'

    # create a service
    service = V1Service(
        api_version="v1",
        kind="Service",
        metadata=V1ObjectMeta(
            name=name,
            labels=labels,
            namespace=namespace
        ),
        spec=V1ServiceSpec(
            ports=[
                V1ServicePort(
                    name=portName, protocol=portProtocol, port=port, target_port=targetPort
                )
            ],
            selector=dict(
                svc=name
            )
        )
    )

    return service


def mapper(function, data: list, conv=None):
    """Mapping

    :param function: func
        mapper function

    :param data: list
        apply mapper function on each item of data

    :param conv: type (default: list)
        convert map function into conv
    """

    if conv is None:
        # set default value for conv
        conv = list

    return conv(map(function, data))
