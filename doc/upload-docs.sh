#! /bin/sh

rsync --progress --verbose --archive --delete build/html/* buster:doc/pycuda
