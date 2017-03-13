#!/usr/bin/env bash
ps -aux|grep EXP_NAME|grep -v grep|cut -c 9-15|xargs kill -s 9