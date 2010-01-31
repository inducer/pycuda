#! /usr/bin/env python

import xmlrpclib
destwiki = xmlrpclib.ServerProxy("http://wiki.tiker.net?action=xmlrpc2")

import os
try:
    os.mkdir("wiki-examples")
except OSError:
    pass

print "fetching page list..."
all_pages = destwiki.getAllPages()

for page in all_pages:
    if not page.startswith("PyCuda/Examples/"):
        continue

    print page
    try:
        content = destwiki.getPage(page)

        import re
        match = re.search(r"\{\{\{\#\!python(.*)\}\}\}", content, re.DOTALL)
        code = match.group(1)

        match = re.search("([^/]+)$", page)
        fname = match.group(1)

        outf = open(os.path.join("wiki-examples", fname+".py"), "w")
        outf.write(code)
        outf.close()

        for att_name in destwiki.listAttachments(page):
            content = destwiki.getAttachment(page, att_name)

            outf = open(os.path.join("wiki-examples", att_name), "w")
            outf.write(str(content))
            outf.close()

    except Exception, e:
        print "Error when processing %s: %s" % (page, e)
