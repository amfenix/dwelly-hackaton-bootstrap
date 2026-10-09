"""The errors the desk raises on purpose. Anything else is a bug."""


class DeskError(Exception):
    """Base class."""


class PolicyError(DeskError):
    """The policy file, or a function it names, is wrong. Raised at load, before any case runs."""


class CaseError(DeskError):
    """A case could not be finished by the rules: a loop, a function that broke the contract."""


class ModelError(DeskError):
    """A model call failed. `cause` is `timeout`, `transport`, `malformed` or `replay`."""

    def __init__(self, cause: str, detail: str) -> None:
        super().__init__(f"{cause}: {detail}")
        self.cause = cause
        self.detail = detail
