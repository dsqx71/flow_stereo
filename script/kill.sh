#!/usr/bin/env bash
ps -aux|grep dispnet|grep -v grep|cut -c 9-15|xargs kill -s 9
ps -aux|grep -v grep|grep EXPERIMENT_NAME |xargs kill -s 9