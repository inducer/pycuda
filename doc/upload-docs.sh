#! /bin/sh

rsync --progress --verbose --archive --delete build/html/* doc-upload:doc/pycuda
