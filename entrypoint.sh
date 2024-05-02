#!/bin/bash

set -euo pipefail

accelerate launch -m lm_eval "$@"
