"""Domain errors raised at pipeline boundaries."""


class PipelineError(Exception):
    """A failure that can be reported without a traceback to a pipeline user."""


class ContractError(PipelineError, ValueError):
    """An artifact or table violates its data contract."""


class ConfigurationError(PipelineError, ValueError):
    """Run settings are invalid or internally inconsistent."""


class ArtifactError(PipelineError):
    """An artifact could not be read or written."""


class SplitError(PipelineError, ValueError):
    """The shipment set cannot support the requested split."""
