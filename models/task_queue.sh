#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

nohup bash task_queue2.sh > weights/exp14/trial.out; 
nohup bash task_queue3.sh > weights/exp15/trial.out;