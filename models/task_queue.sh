#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

task_queue2.sh > weights/exp14/trial.out; 
task_queue3.sh > weights/exp15/trial.out;