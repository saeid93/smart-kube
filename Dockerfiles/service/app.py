import time

from flask import Flask, request
import requests
import logging
import socket
import os
import time
import signal
import threading

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

PORT = 80
COMMAND_TEMPLATE = "stress-ng --vm 1 --vm-bytes '{}M' --cpu 1 --cpu-load '{}' &"
SUCCESS_MESSAGE = "Resources allocated successfully"
app = Flask(__name__)


DURATION = int(os.environ['DURATION'])
logging.info(f'DURATION set to: {DURATION}')



def stop_server():
    print('Stopping Flask app...')
    # Send a SIGTERM signal to the current process
    os.kill(os.getpid(), signal.SIGTERM)

@app.route('/', methods=['GET'])
def index():

    try:
        logging.info('killing previous stress-ng procceses...')
        os.system("killall -9 stress-ng")
    except Exception as e:
        logging.warning(e)
        pass

    try:

        logging.info("received new configuration from sdghafoLIFETIMEuri/utilization-server-smart-scheduler: {}".format(
            request.form
        ))

        ram = request.args.get('ram')
        cpu = request.args.get('cpu')

        command = COMMAND_TEMPLATE.format(ram, cpu)
        logging.info('running stress-ng: "{}"'.format(command))
        os.system(command)

        logging.info(SUCCESS_MESSAGE)
        return SUCCESS_MESSAGE
    except Exception as e:
        logging.error(e)
        return 'An issue faces, please check out the logs.'


if __name__ == '__main__':
    # get hostname of current machine
    hostname = socket.gethostname()
    command = None
    logging.info('starting service ...')
    # logging.info('trying to connect to the sdghafouri/utilization-server-smart-scheduler')
    # while True:
    #     try:
    #         # register into controller and setup the stress
    #         controller = requests.get('http://sdghafouri/utilization-server-smart-scheduler.consolidation.svc/metrics/{}/'.format(hostname))

    #         if controller.status_code == 404:
    #             logging.info(controller.content)
    #             time.sleep(1)
    #             continue

    #         content = controller.json()
    #         logging.info("got resources: {}".format(content))

    #         # running stress
    #         command = COMMAND_TEMPLATE.format(
    #             content.get('ram'),
    #             content.get('cpu'),
    #         )
    #         break

    #     except Exception as e:
    #         logging.error(e)
    #         exit(-1)

    # logging.info('running stress-ng: "{}"'.format(command))
    # os.system(command)

    timer = threading.Timer(DURATION, stop_server)
    timer.start()

    # logging.info("serving 'app' on port {}".format(PORT))
    # app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
