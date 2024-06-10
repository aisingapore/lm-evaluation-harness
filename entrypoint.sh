#!/bin/bash

accelerate launch -m lm_eval "$@"
