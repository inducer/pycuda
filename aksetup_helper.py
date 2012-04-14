# dealings with ez_setup ------------------------------------------------------
import distribute_setup
distribute_setup.use_setuptools()

import setuptools
from setuptools import Extension

def count_down_delay(delay):
    from time import sleep
    import sys
    while delay:
        sys.stdout.write("Continuing in %d seconds...   \r" % delay)
        sys.stdout.flush()
        delay -= 1
        sleep(1)
    print("")




if not hasattr(setuptools, "_distribute"):
    print("-------------------------------------------------------------------------")
    print("Setuptools conflict detected.")
    print("-------------------------------------------------------------------------")
    print("When I imported setuptools, I did not get the distribute version of")
    print("setuptools, which is troubling--this package really wants to be used")
    print("with distribute rather than the old setuptools package. More than likely,")
    print("you have both distribute and setuptools installed, which is bad.")
    print("")
    print("See this page for more information:")
    print("http://wiki.tiker.net/DistributeVsSetuptools")
    print("-------------------------------------------------------------------------")
    print("I will continue after a short while, fingers crossed.")
    print("Hit Ctrl-C now if you'd like to think about the situation.")
    print("-------------------------------------------------------------------------")

    count_down_delay(delay=10)


def setup(*args, **kwargs):
    from setuptools import setup
    import traceback
    try:
        setup(*args, **kwargs)
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except:
        print ("----------------------------------------------------------------------------")
        print ("Sorry, your build failed. Try rerunning configure.py with different options.")
        print ("----------------------------------------------------------------------------")
        raise




class NumpyExtension(Extension):
    # nicked from
    # http://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
    # solution by Michael Hoffmann
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self._include_dirs = self.include_dirs
        del self.include_dirs # restore overwritten property

    def get_numpy_incpath(self):
        from imp import find_module
        # avoid actually importing numpy, it screws up distutils
        file, pathname, descr = find_module("numpy")
        from os.path import join
        return join(pathname, "core", "include")

    def get_include_dirs(self):
        return self._include_dirs + [self.get_numpy_incpath()]
    def set_include_dirs(self, value):
        self._include_dirs = value
    def del_include_dirs(self):
        pass

    include_dirs = property(get_include_dirs, set_include_dirs, del_include_dirs)




class PyUblasExtension(NumpyExtension):
    def get_module_include_path(self, name):
        from imp import find_module
        file, pathname, descr = find_module(name)
        from os.path import join, exists
        installed_path = join(pathname, "..", "include")
        development_path = join(pathname, "..", "src", "cpp")
        if exists(installed_path):
            return installed_path
        elif exists(development_path):
            return development_path
        else:
            raise RuntimeError("could not find C include path for module '%s'" % name)

    @property
    def include_dirs(self):
        return self._include_dirs + [
                self.get_numpy_incpath(),
                self.get_module_include_path("pyublas"),
                ]




class HedgeExtension(PyUblasExtension):
    @property
    def include_dirs(self):
        return self._include_dirs + [
                self.get_numpy_incpath(),
                self.get_module_include_path("pyublas"),
                self.get_module_include_path("hedge"),
                ]




# tools -----------------------------------------------------------------------
def flatten(list):
    """For an iterable of sub-iterables, generate each member of each
    sub-iterable in turn, i.e. a flattened version of that super-iterable.

    Example: Turn [[a,b,c],[d,e,f]] into [a,b,c,d,e,f].
    """
    for sublist in list:
        for j in sublist:
            yield j




def humanize(sym_str):
    words = sym_str.lower().replace("_", " ").split(" ")
    return " ".join([word.capitalize() for word in words])




# siteconf handling -----------------------------------------------------------
def get_config(schema=None, warn_about_no_config=True):
    if schema is None:
        from setup import get_config_schema
        schema = get_config_schema()

    if (not schema.have_config() and not schema.have_global_config()
            and warn_about_no_config):
        print("*************************************************************")
        print("*** I have detected that you have not run configure.py.")
        print("*************************************************************")
        print("*** Additionally, no global config files were found.")
        print("*** I will go ahead with the default configuration.")
        print("*** In all likelihood, this will not work out.")
        print("*** ")
        print("*** See README_SETUP.txt for more information.")
        print("*** ")
        print("*** If the build does fail, just re-run configure.py with the")
        print("*** correct arguments, and then retry. Good luck!")
        print("*************************************************************")
        print("*** HIT Ctrl-C NOW IF THIS IS NOT WHAT YOU WANT")
        print("*************************************************************")

        count_down_delay(delay=10)

    return expand_options(schema.read_config())




def hack_distutils(debug=False, fast_link=True, what_opt=3):
    # hack distutils.sysconfig to eliminate debug flags
    # stolen from mpi4py

    def remove_prefixes(optlist, bad_prefixes):
        for bad_prefix in bad_prefixes:
            for i, flag in enumerate(optlist):
                if flag.startswith(bad_prefix):
                    optlist.pop(i)
                    break
        return optlist

    import sys
    if not sys.platform.lower().startswith("win"):
        from distutils import sysconfig

        cvars = sysconfig.get_config_vars()
        cflags = cvars.get('OPT')
        if cflags:
            cflags = remove_prefixes(cflags.split(),
                    ['-g', '-O', '-Wstrict-prototypes', '-DNDEBUG'])
            if debug:
                cflags.append("-g")
            else:
                if what_opt is None:
                    pass
                else:
                    cflags.append("-O%s" % what_opt)
                    cflags.append("-DNDEBUG")

            cvars['OPT'] = str.join(' ', cflags)
            cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]

        if fast_link:
            for varname in ["LDSHARED", "BLDSHARED"]:
                ldsharedflags = cvars.get(varname)
                if ldsharedflags:
                    ldsharedflags = remove_prefixes(ldsharedflags.split(),
                            ['-Wl,-O'])
                    cvars[varname] = str.join(' ', ldsharedflags)



# configure guts --------------------------------------------------------------
def default_or(a, b):
    if a is None:
        return b
    else:
        return a



def expand_str(s, options):
    import re

    def my_repl(match):
        sym = match.group(1)
        try:
            repl = options[sym]
        except KeyError:
            from os import environ
            repl = environ[sym]

        return expand_str(repl, options)

    return re.subn(r"\$\{([a-zA-Z0-9_]+)\}", my_repl, s)[0]

def expand_value(v, options):
    if isinstance(v, str):
        return expand_str(v, options)
    elif isinstance(v, list):
        result = []
        for i in v:
            try:
                exp_i = expand_value(i, options)
            except:
                pass
            else:
                result.append(exp_i)

        return result
    else:
        return v


def expand_options(options):
    return dict(
            (k, expand_value(v, options)) for k, v in options.items())






class ConfigSchema:
    def __init__(self, options, conf_file="siteconf.py", conf_dir="."):
        self.optdict = dict((opt.name, opt) for opt in options)
        self.options = options
        self.conf_dir = conf_dir
        self.conf_file = conf_file

        from os.path import expanduser
        self.user_conf_file = expanduser("~/.aksetup-defaults.py")

        import sys
        if not sys.platform.lower().startswith("win"):
            self.global_conf_file = "/etc/aksetup-defaults.py"
        else:
            self.global_conf_file = None

    def get_conf_file(self):
        import os
        return os.path.join(self.conf_dir, self.conf_file)

    def set_conf_dir(self, conf_dir):
        self.conf_dir = conf_dir

    def get_default_config(self):
        return dict((opt.name, opt.default) for opt in self.options)

    def read_config_from_pyfile(self, filename):
        result = {}
        filevars = {}
        infile = open(filename, "r")
        try:
            contents = infile.read()
        finally:
            infile.close()

        exec(compile(contents, filename, "exec"), filevars)

        for key, value in filevars.items():
            if key in self.optdict:
                result[key] = value

        return result

    def update_conf_file(self, filename, config):
        result = {}
        filevars = {}

        try:
            exec(compile(open(filename, "r").read(), filename, "exec"), filevars)
        except IOError:
            pass

        if "__builtins__" in filevars:
            del filevars["__builtins__"]

        for key, value in config.items():
            if value is not None:
                filevars[key] = value

        keys = filevars.keys()
        keys.sort()

        outf = open(filename, "w")
        for key in keys:
            outf.write("%s = %s\n" % (key, repr(filevars[key])))
        outf.close()

        return result

    def update_user_config(self, config):
        self.update_conf_file(self.user_conf_file, config)

    def update_global_config(self, config):
        if self.global_conf_file is not None:
            self.update_conf_file(self.global_conf_file, config)

    def get_default_config_with_files(self):
        result = self.get_default_config()

        import os

        confignames = []
        if self.global_conf_file is not None:
            confignames.append(self.global_conf_file)
        confignames.append(self.user_conf_file)

        for fn in confignames:
            if os.access(fn, os.R_OK):
                result.update(self.read_config_from_pyfile(fn))

        return result

    def have_global_config(self):
        import os
        result = os.access(self.user_conf_file, os.R_OK)

        if self.global_conf_file is not None:
            result = result or os.access(self.global_conf_file, os.R_OK)

        return result

    def have_config(self):
        import os
        return os.access(self.get_conf_file(), os.R_OK)

    def read_config(self, warn_if_none=True):
        import os
        cfile = self.get_conf_file()

        result = self.get_default_config_with_files()
        if os.access(cfile, os.R_OK):
            filevars = {}
            exec(compile(open(cfile, "r").read(), cfile, "exec"), filevars)

            for key, value in filevars.items():
                if key in self.optdict:
                    result[key] = value
                elif key == "__builtins__":
                    pass
                else:
                    raise KeyError("invalid config key in %s: %s" % (
                            cfile, key))

        return result

    def add_to_configparser(self, parser, def_config=None):
        if def_config is None:
            def_config = self.get_default_config_with_files()

        for opt in self.options:
            default = default_or(def_config.get(opt.name), opt.default)
            opt.add_to_configparser(parser, default)

    def get_from_configparser(self, options):
        result = {}
        for opt in self.options:
            result[opt.name] = opt.take_from_configparser(options)
        return result

    def write_config(self, config):
        outf = open(self.get_conf_file(), "w")
        for opt in self.options:
            value = config[opt.name]
            if value is not None:
                outf.write("%s = %s\n" % (opt.name, repr(config[opt.name])))
        outf.close()

    def make_substitutions(self, config):
        return dict((opt.name, opt.value_to_str(config[opt.name]))
                for opt in self.options)








class Option(object):
    def __init__(self, name, default=None, help=None):
        self.name = name
        self.default = default
        self.help = help

    def as_option(self):
        return self.name.lower().replace("_", "-")

    def metavar(self):
        last_underscore = self.name.rfind("_")
        return self.name[last_underscore+1:]

    def get_help(self, default):
        result = self.help
        if self.default:
            result += " (default: %s)" % self.value_to_str(
                    default_or(default, self.default))
        return result

    def value_to_str(self, default):
        return default

    def add_to_configparser(self, parser, default=None):
        default = default_or(default, self.default)
        default_str = self.value_to_str(default)
        parser.add_option(
            "--" + self.as_option(), dest=self.name,
            default=default_str,
            metavar=self.metavar(), help=self.get_help(default))

    def take_from_configparser(self, options):
        return getattr(options, self.name)

class Switch(Option):
    def add_to_configparser(self, parser, default=None):
        option = self.as_option()

        if not isinstance(self.default, bool):
            raise ValueError("Switch options must have a default")

        if default is None:
            default = self.default

        option_name = self.as_option()

        if default:
            option_name = "no-" + option_name
            action = "store_false"
        else:
            action = "store_true"

        parser.add_option(
            "--" + option_name,
            dest=self.name,
            help=self.get_help(default),
            default=default,
            action=action)

class StringListOption(Option):
    def value_to_str(self, default):
        if default is None:
            return None

        return ",".join([str(el) for el in default])

    def get_help(self, default):
        return Option.get_help(self, default) + " (several ok)"

    def take_from_configparser(self, options):
        opt = getattr(options, self.name)
        if opt is None:
            return None
        else:
            if opt:
                return opt.split(",")
            else:
                return []


class IncludeDir(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(self, "%s_INC_DIR" % lib_name, default,
                help=help or ("Include directories for %s"
                % (human_name or humanize(lib_name))))

class LibraryDir(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(self, "%s_LIB_DIR" % lib_name, default,
                help=help or ("Library directories for %s"
                % (human_name or humanize(lib_name))))

class Libraries(StringListOption):
    def __init__(self, lib_name, default=None, human_name=None, help=None):
        StringListOption.__init__(self, "%s_LIBNAME" % lib_name, default,
                help=help or ("Library names for %s (without lib or .so)"
                % (human_name or humanize(lib_name))))

class BoostLibraries(Libraries):
    def __init__(self, lib_base_name):
        Libraries.__init__(self, "BOOST_%s" % lib_base_name.upper(),
                ["boost_%s" % lib_base_name],
                help="Library names for Boost C++ %s library (without lib or .so)"
                    % humanize(lib_base_name))

def set_up_shipped_boost_if_requested(project_name, conf):
    """Set up the package to use a shipped version of Boost.

    Return a tuple of a list of extra C files to build and extra
    defines to be used.
    """
    from os.path import exists
    import sys

    if conf["USE_SHIPPED_BOOST"]:
        if not exists("bpl-subset/bpl_subset/boost/version.hpp"):
            print("------------------------------------------------------------------------")
            print("The shipped Boost library was not found, but USE_SHIPPED_BOOST is True.")
            print("(The files should be under bpl-subset/.)")
            print("------------------------------------------------------------------------")
            print("If you got this package from git, you probably want to do")
            print("")
            print(" $ git submodule init")
            print(" $ git submodule update")
            print("")
            print("to fetch what you are presently missing. If you got this from")
            print("a distributed package on the net, that package is broken and")
            print("should be fixed. For now, I will turn off 'USE_SHIPPED_BOOST'")
            print("to try and see if the build succeeds that way, but in the long")
            print("run you might want to either get the missing bits or turn")
            print("'USE_SHIPPED_BOOST' off.")
            print("------------------------------------------------------------------------")
            conf["USE_SHIPPED_BOOST"] = False

            count_down_delay(delay=10)

    if conf["USE_SHIPPED_BOOST"]:
        conf["BOOST_INC_DIR"] = ["bpl-subset/bpl_subset"]
        conf["BOOST_LIB_DIR"] = []
        conf["BOOST_PYTHON_LIBNAME"] = []
        conf["BOOST_THREAD_LIBNAME"] = []

        from glob import glob
        source_files = (glob("bpl-subset/bpl_subset/libs/*/*/*/*.cpp")
                + glob("bpl-subset/bpl_subset/libs/*/*/*.cpp")
                + glob("bpl-subset/bpl_subset/libs/*/*.cpp"))

        # make sure next line succeeds even on Windows
        source_files = [f.replace("\\","/") for f in source_files]

        source_files = [f for f in source_files
                if not f.startswith("bpl-subset/bpl_subset/libs/thread/src")]

        if sys.platform == "win32":
            source_files += glob(
                    "bpl-subset/bpl_subset/libs/thread/src/win32/*.cpp")
            source_files += glob(
                    "bpl-subset/bpl_subset/libs/thread/src/*.cpp")
        else:
            source_files += glob(
                    "bpl-subset/bpl_subset/libs/thread/src/pthread/*.cpp")

        from os.path import isdir
        main_boost_inc = "bpl-subset/bpl_subset/boost"
        bpl_project_boost_inc = "bpl-subset/bpl_subset/%sboost" % project_name

        if not isdir(bpl_project_boost_inc):
            try:
                from os import symlink
            except ImportError:
                from shutil import copytree
                print("Copying files, hang on... (do not interrupt)")
                copytree(main_boost_inc, bpl_project_boost_inc)
            else:
                symlink("boost", bpl_project_boost_inc)

        return (source_files,
                {
                    # do not pick up libboost link dependency on windows
                    "BOOST_ALL_NO_LIB": 1,
                    "BOOST_THREAD_BUILD_DLL": 1,

                    "BOOST_MULTI_INDEX_DISABLE_SERIALIZATION": 1,
                    "BOOST_PYTHON_SOURCE": 1,
                    "boost": '%sboost' % project_name
                    }
                )
    else:
        return [], {}


def make_boost_base_options():
    return [
        IncludeDir("BOOST", []),
        LibraryDir("BOOST", []),
        Option("BOOST_COMPILER", default="gcc43",
            help="The compiler with which Boost C++ was compiled, e.g. gcc43"),
        ]







def configure_frontend():
    from optparse import OptionParser

    from setup import get_config_schema
    schema = get_config_schema()
    if schema.have_config():
        print("************************************************************")
        print("*** I have detected that you have already run configure.")
        print("*** I'm taking the configured values as defaults for this")
        print("*** configure run. If you don't want this, delete the file")
        print("*** %s." % schema.get_conf_file())
        print("************************************************************")

    import sys

    description = "generate a configuration file for this software package"
    parser = OptionParser(description=description)
    parser.add_option(
            "--python-exe", dest="python_exe", default=sys.executable,
            help="Which Python interpreter to use", metavar="PATH")

    parser.add_option("--prefix", default=None,
            help="Ignored")
    parser.add_option("--enable-shared", help="Ignored", action="store_false")
    parser.add_option("--disable-static", help="Ignored", action="store_false")
    parser.add_option("--update-user", help="Update user config file (%s)" % schema.user_conf_file,
            action="store_true")
    parser.add_option("--update-global",
            help="Update global config file (%s)" % schema.global_conf_file,
            action="store_true")

    schema.add_to_configparser(parser, schema.read_config())

    options, args = parser.parse_args()

    config = schema.get_from_configparser(options)
    schema.write_config(config)

    if options.update_user:
        schema.update_user_config(config)

    if options.update_global:
        schema.update_global_config(config)

    import os
    if os.access("Makefile.in", os.F_OK):
        substs = schema.make_substitutions(config)
        substs["PYTHON_EXE"] = options.python_exe

        substitute(substs, "Makefile")




def substitute(substitutions, fname):
    import re
    var_re = re.compile(r"\$\{([A-Za-z_0-9]+)\}")
    string_var_re = re.compile(r"\$str\{([A-Za-z_0-9]+)\}")

    fname_in = fname+".in"
    lines = open(fname_in, "r").readlines()
    new_lines = []
    for l in lines:
        made_change = True
        while made_change:
            made_change = False
            match = var_re.search(l)
            if match:
                varname = match.group(1)
                l = l[:match.start()] + str(substitutions[varname]) + l[match.end():]
                made_change = True

            match = string_var_re.search(l)
            if match:
                varname = match.group(1)
                subst = substitutions[varname]
                if subst is None:
                    subst = ""
                else:
                    subst = '"%s"' % subst

                l = l[:match.start()] + subst  + l[match.end():]
                made_change = True
        new_lines.append(l)
    new_lines.insert(1, "# DO NOT EDIT THIS FILE -- it was generated by configure.py\n")
    import sys
    new_lines.insert(2, "# %s\n" % (" ".join(sys.argv)))
    open(fname, "w").write("".join(new_lines))

    from os import stat, chmod
    infile_stat_res = stat(fname_in)
    chmod(fname, infile_stat_res.st_mode)




def check_git_submodules():
    from os.path import isdir
    if not isdir(".git"):
        # not a git repository
        return

    git_error = None
    from subprocess import Popen, PIPE
    try:
        popen = Popen(["git", "--version"], stdout=PIPE)
        stdout_data, _ = popen.communicate()
        if popen.returncode != 0:
            git_error = "git returned error code %d" % popen.returncode
    except OSError:
        git_error = "(os error, likely git not found)"

    if git_error is not None:
        print("-------------------------------------------------------------------------")
        print("Trouble invoking git")
        print("-------------------------------------------------------------------------")
        print("The package directory appears to be a git repository, but I could")
        print("not invoke git to check whether my submodules are up to date.")
        print("")
        print("The error was:")
        print(git_error)
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print("-------------------------------------------------------------------------")
        count_down_delay(delay=5)
        return

    popen = Popen(["git", "submodule", "status"], stdout=PIPE)
    stdout_data, _ = popen.communicate()
    stdout_data = stdout_data.decode("ascii")
    if popen.returncode != 0:
        git_error = "git returned error code %d" % popen.returncode

    pkg_warnings = []

    lines = stdout_data.split("\n")
    for l in lines:
        if not l.strip():
            continue

        status = l[0]
        sha, package = l[1:].split(" ", 1)

        if package == "bpl-subset":
            # treated separately
            continue

        if status == "+":
            pkg_warnings.append("version of '%s' is not what this outer package wants"
                    % package)
        elif status == "-":
            pkg_warnings.append("subpackage '%s' is not initialized" 
                    % package)
        elif status == " ":
            pass
        else:
            pkg_warnings.append("subpackage '%s' has unrecognized status '%s'"
                    % package)

    if pkg_warnings:
            print("-------------------------------------------------------------------------")
            print("git submodules are not up-to-date or in odd state")
            print("-------------------------------------------------------------------------")
            print("If this makes no sense, you probably want to say")
            print("")
            print(" $ git submodule init")
            print(" $ git submodule update")
            print("")
            print("to fetch what you are presently missing and move on with your life.")
            print("If you got this from a distributed package on the net, that package is")
            print("broken and should be fixed. Please inform whoever gave you this package.")
            print("")
            print("These issues were found:")
            for w in pkg_warnings:
                print("  %s" % w)
            print("")
            print("I will try to continue after a short wait, fingers crossed.")
            print("-------------------------------------------------------------------------")
            print("Hit Ctrl-C now if you'd like to think about the situation.")
            print("-------------------------------------------------------------------------")

            from os.path import exists
            if not exists(".dirty-git-ok"):
                count_down_delay(delay=10)









