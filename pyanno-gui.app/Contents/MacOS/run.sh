#!/bin/bash

# Launch pyanno GUI in a terminal

CURRPATH=$(cd "$(dirname "$0")"; pwd)
open $CURRPATH/../Resources/*egg -a Terminal
