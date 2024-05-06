import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logging():
    # # Create a logger
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)  # Set the logging level
    #
    # # Create a file handler that logs even debug messages
    # if not os.path.exists('logs'):
    #     os.makedirs('logs')
    # file_handler = RotatingFileHandler('logs/application.log', maxBytes=1024*1024*5, backupCount=5)
    # file_handler.setLevel(logging.DEBUG)
    #
    # # Create a console handler with a higher log level
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.ERROR)
    #
    # # Create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s;%(name)s;%(levelname)s;%(message)s')
    # file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    #
    # # Add the handlers to the logger
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    logging.basicConfig(filename=r"C:\Users\albert\PycharmProjects\PacketSWMM\helpers\application.log",
                        filemode='w',  # Overwrites the file
                        level=logging.NOTSET,
                        format='%(asctime)s;%(name)s;%(levelname)s;%(message)s')


def main():
    pass


if __name__ == "__main__":
    main()
    pass
