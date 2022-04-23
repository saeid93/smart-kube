import numpy as np

class ServiceResource:
    def __init__(self, identifier, capacity, usage, length=50):
        self.identifier = identifier
        self.capacity = capacity
        self.usage = usage
        self.length = length

    def __str__(self):
        normalizer = self.length/self.capacity
        return '[' + '░' * int(self.usage*normalizer) +\
               '-' * int((self.capacity-self.usage)*normalizer) + ']'


class Server:
    def __init__(self, identifier, plot_longth, capacity):
        self.identifier = identifier
        self.plot_length = plot_longth
        self.capacity = capacity
        self.resources = dict({
            'RAM': [ServiceResource(0, 15, 60),
                    ServiceResource(1, 30, 25)],
            'CPU': [ServiceResource(0, 15, 60),
                    ServiceResource(1, 30, 25)],
        })

    def resource_string(self, resource_name):
        ram_sum = 0
        resource = self.resources[resource_name]
        for r in resource:
            ram_sum += r.capacity
        ram_normalizer = self.plot_length/self.capacity
        start_str = ': <'
        ram_plot = resource_name + start_str
        r_sum = 0
        r_count = 0
        for r in resource:
            ram_plot = ram_plot + '[' + ('{:░^%d}' % (int(ram_normalizer*r.usage))).format(str(r.identifier)) + ']'
            r_sum += int(ram_normalizer*r.usage)
            r_count += 1
        if r_sum < self.plot_length - len(resource_name+start_str):
            ram_plot = ram_plot +\
                       '-'*(self.plot_length - len(ram_plot) - 1) + '>'
        else:
            ram_plot = (ram_plot[:self.plot_length
                        + len(resource_name+start_str)] + '>' +
                        ram_plot[self.plot_length +
                        len(resource_name+start_str):])
        return ram_plot

    def __str__(self):
        return ('Server ' +
                str(self.identifier) + ':\n\t' +
                'Resource usages:\n\t' +
                self.resource_string('RAM') + '\n\t' +
                self.resource_string('CPU') + '\n\t')


def plot_resource_allocation(service_node, node_resource_cap,
                             service_resource_request, service_resource_usage,
                             plot_length=80):
    servers = []
    for i in range(len(node_resource_cap)):
        servers.append(Server(identifier=i,
                              capacity=np.max(node_resource_cap[i]),
                              plot_longth=plot_length))

        servers[i].resources['RAM'] = [
            ServiceResource(identifier=j,
                              capacity=service_resource_request[j][0],
                              usage=service_resource_usage[j][0]
            ) for j in range(len(service_resource_request)) if service_node[j] == i
        ]

        servers[i].resources['CPU'] = [
            ServiceResource(identifier=j,
                              capacity=service_resource_request[j][1],
                              usage=service_resource_usage[j][1]
            ) for j in range(len(service_resource_request)) if service_node[j] == i
        ]
        print(str(servers[i]))

        for c in range(len(service_resource_request)):
            if service_node[c] == i:
                print('\tservice %d' % c)
                print('\t\tRAM:', ServiceResource(identifier=c,
                                                    capacity=service_resource_request[c][0],
                                                    usage=service_resource_usage[c][0]))
                print('\t\tCPU:', ServiceResource(identifier=c,
                                                     capacity=service_resource_request[c][1],
                                                     usage=service_resource_usage[c][1]))
