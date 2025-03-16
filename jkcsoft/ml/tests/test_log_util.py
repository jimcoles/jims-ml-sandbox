#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.

import unittest

from jkcsoft.ml.log_utils import logger

class MyTestCase(unittest.TestCase):

    def test_logging(self):
        #
        log = logger(__name__)
        #
        log.fatal("fatal logged")
        log.exception("exception logged")
        log.critical("critical logged")
        log.error("error logged")
        log.warning("warning logged")
        log.info("info logged")
        log.debug("debug logged")
        #
        self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()
