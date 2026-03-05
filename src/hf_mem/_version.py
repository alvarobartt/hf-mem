from importlib.metadata import version

# NOTE: Single source of truth for the package version, imported by all sub-modules that need it
# to avoid circular imports through hf_mem.__init__
__version__: str = version("hf-mem")
