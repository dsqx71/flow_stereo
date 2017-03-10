#!/usr/bin/env bash
ps -aux|grep exp_name|grep -v grep|cut -c 9-15|xargs kill -s 9