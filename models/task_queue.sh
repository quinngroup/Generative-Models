#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

nohup bash task_queue2.sh > weights/exp14/trial.out; 
cp task_queue2.sh runs/exp14/task_queue.sh;
nohup bash task_queue3.sh > weights/exp15/trial.out;
cp task_queue3.sh runs/exp15/task_queue.sh;
