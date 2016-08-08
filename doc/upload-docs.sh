#! /bin/sh

rsync --verbose --archive --delete build/html/* doc-upload:doc/pycuda
