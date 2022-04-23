from models import WorkLoads, Dataset
from flask import jsonify
from flask import Flask
import schedule
import requests
import threading
import logging
import pickle
import time

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

# Flask configurations
PORT = 80
app = Flask(__name__)

# Time Interval (seconds)
INTERVAL = 7

# Enable Scheduler
IS_ENABLED_SCHEDULER = False

# Current TimeStep
CURRENT_TIME_STEP = 0

# WorkLoads
WORKLOAD_PATH = '/workload.pickle'
DATASET_PATH = '/dataset.pickle'
WORKLOADS: WorkLoads
DATASET: Dataset

# Services
SERVICES = dict()
"""
Service: {
    "type": 0,
    "specs": {
        "ram": blah_0,
        "cpu": blah_1
    } 
}
"""

# Moved Services
MOVED_SERVICES = dict()


@app.route('/hostname/update/<string:previous>/<string:new>/', methods=['GET'])
def update(previous: str, new: str):
    global MOVED_SERVICES

    if SERVICES.get(previous, None) is None:
        return "Service '{}' does not exist.".format(
            previous
        ), 404
    found = False
    for source, dest in MOVED_SERVICES.items():
        if dest == previous:
            MOVED_SERVICES[source] = new
            found = True
            break
    if not found:
        MOVED_SERVICES[previous] = new
    SERVICES[new] = SERVICES[previous]
    del SERVICES[previous]
    return 'Updated.', 200


@app.route('/metrics/<string:hostname>/', methods=['GET'])
def metrics(hostname: str):
    global WORKLOADS, DATASET, CURRENT_TIME_STEP, SERVICES, MOVED_SERVICES, IS_ENABLED_SCHEDULER, INTERVAL
    """Register a hostname and return him current metrics

        Firstly, it's searching for hostname, if it's not present, so it will be created
    :param hostname: str
        hostname of service
    """

    # service exists, moved service does not exist
    if (SERVICES.get(hostname, None) is not None) and (MOVED_SERVICES.get(hostname, None) is None):
        return "service '{}' exists, and there is no mapping, so you need to hold on...".format(
            hostname
        ), 404

    # moved service exsits
    if MOVED_SERVICES.get(hostname, None) is not None:
        hostname = MOVED_SERVICES[hostname]

    # service does not exist
    if SERVICES.get(hostname, None) is None:
        logging.info('service "{}" does not exist, so its resources will be allocated'.format(hostname))
        # logging.info('services: {}'.format(SERVICES))
        # logging.info('moved services: {}'.format(MOVED_SERVICES))
        # logging.info('services_types: {}'.format(DATASET.data.get('services_types')))
        service_type = DATASET.data.get('services_types')[len(SERVICES)]
        INIT_RAM, INIT_CPU = WORKLOADS.data[service_type, CURRENT_TIME_STEP, :]

        service = {
            hostname: {
                "type": service_type,
                "specs": {
                    "ram": INIT_RAM,
                    "cpu": INIT_CPU
                }
            }
        }

        SERVICES.update(service)

        if len(SERVICES.keys()) == WORKLOADS.nWorkloads and not IS_ENABLED_SCHEDULER:
            schedule.every(INTERVAL).seconds.do(updateTimesteps)
            threading.Thread(target=runScheduler).start()
            IS_ENABLED_SCHEDULER = True

        return jsonify(SERVICES.get(hostname).get('specs')), 200

    # service exists, moved service exists
    return jsonify(SERVICES.get(hostname).get('specs')), 200


def updateTimesteps():
    global CURRENT_TIME_STEP, SERVICES

    logging.info('increasing the current time step from {} to {}'.format(
        CURRENT_TIME_STEP, CURRENT_TIME_STEP + 1
    ))
    CURRENT_TIME_STEP = CURRENT_TIME_STEP % WORKLOADS.nTimesteps + 1

    try:
        for hostname, detail in SERVICES.items():

            URL = "http://{}.consolidation.svc/".format(hostname)

            container_type = detail.get('type')
            RAM, CPU = WORKLOADS.data[:, CURRENT_TIME_STEP, container_type]

            logging.info("Updating specs of service '{}' from '{}' to '{}'".format(
                hostname,
                SERVICES[hostname]['specs'],
                {
                    'ram': RAM,
                    'cpu': CPU
                }
            ))

            SERVICES[hostname]['specs'] = {
                'ram': RAM,
                'cpu': CPU
            }

            requests.get(url=URL, params=SERVICES[hostname]['specs'], timeout=3)
    except Exception as e:
        logging.error(e)


def runScheduler():

    time.sleep(2)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':

    while True:
        try:

            with open(WORKLOAD_PATH, 'rb') as file:
                WORKLOADS = WorkLoads(pickle.load(file))

            with open(DATASET_PATH, 'rb') as file:
                DATASET = Dataset(pickle.load(file))

            break
        except Exception as e:
            logging.info(
                'looking for file "{}" and "{}", '
                'in order to run web server, you need to upload them: {}'.format(
                    WORKLOAD_PATH,
                    DATASET_PATH,
                    e
                ))
        time.sleep(1)

    logging.info("serving 'app' on port {}".format(PORT))
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
