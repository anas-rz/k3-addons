import namex

# Unique source of truth for the version number.
__version__ = "0.0.2"


@namex.export(package="k3_addons", path="k3_addons.version")
def version():
    return __version__
