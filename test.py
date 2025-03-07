import logging
import requests

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)

fh = logging.FileHandler("log.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s'))
ch.setLevel(logging.INFO)
module_logger.addHandler(ch)

module_logger.info("This is an info!")
module_logger.debug("This is a debug msg!")
module_logger.debug("This is a warning!")
module_logger.debug("This is an error!")

# this request produces DEBUG-level logging output which I want to see in the
# logfile but NOT in the console output.
x = requests.get("https://google.com").status_code