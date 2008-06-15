#! /bin/sh

rsync --progress --verbose --archive --delete build/html/* tikernet@tiker.net:public_html/doc/pycuda
