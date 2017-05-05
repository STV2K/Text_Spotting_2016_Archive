# !/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6.1 -*-

import os
import logging  
import logging.handlers  

class stLogger(object):
    """
    Logging class.
    !! This class must be instantiated before use.
    @pythonVersion: 3.5.2
    @methods:
            __init__        Initiate logger instance. Change log level with 
                            'loggername, loglevel = logging.LEVEL'.
                            LEVEL can be INFO/DEBUG/WARNING(default)/ERROR/CRITICAL.
            clean           Cleanup logfile for current logger.
    @author: X.Huang
    @creation: 2016-12-17
    @modified: 2016-12-17
    @version: 0.1
    """
    log = None
    logfile = None
    def __init__(self, loggername, loglevel = logging.WARNING):
        self.logfile = loggername + ".stlog"
        # Init logging handler
        handler = logging.handlers.RotatingFileHandler(self.logfile, maxBytes = 1024*1024, backupCount = 5, encoding = "utf-8")
        # Set up formatter for handler
        fmt = '[%(levelname)s] %(asctime)s\n%(filename)s:%(lineno)s\t%(message)s\t | %(funcName)s'  
        formatter = logging.Formatter(fmt) 
        handler.setFormatter(formatter) 
          
        # Set up logger
        self.log = logging.getLogger(loggername) 
        self.log.addHandler(handler) 
        self.log.setLevel(loglevel)

    def clean(self):
        try:
            os.remove(self.logfile)
        except Exception as _Eall:
            self.log.error("Cannot clean up logfile: %s"%_Eall)
