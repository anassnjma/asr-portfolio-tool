#!/usr/bin/env python3
"""Entry point – delegates to controllers.main."""

import sys
sys.dont_write_bytecode = True

from controllers.main import main

if __name__ == "__main__":
    main()
