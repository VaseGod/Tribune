"""Parse, Don't Validate Domain Boundary Enforcement.

Ensures untrusted raw incoming payloads (dictionaries, JSON structures, parameters) are
parsed directly into strongly typed domain objects instead of being sanitized as raw primitives.
Fails early with structured ValidationErrorList if type parsing or constraints fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

T = TypeVar("T")


@dataclass
class FieldError:
    field: str
    message: str
    invalid_value: Any = None


@dataclass
class ValidationErrorList(Exception):
    """Aggregate error container representing all boundary parsing failures."""

    errors: list[FieldError] = field(default_factory=list)

    def __str__(self) -> str:
        err_str = "; ".join([f"'{e.field}': {e.message}" for e in self.errors])
        return f"Domain parsing failed with {len(self.errors)} error(s): [{err_str}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "parse_error",
            "error_count": len(self.errors),
            "errors": [
                {"field": e.field, "message": e.message, "value": str(e.invalid_value)}
                for e in self.errors
            ],
        }


class DomainParser(Generic[T]):
    """Boundary parser enforcing 'Parse, Don't Validate' design patterns."""

    def __init__(self, target_type: Type[T]) -> None:
        self.target_type = target_type

    def parse(self, raw_payload: Any) -> T:
        """Parse raw untrusted payload directly into a strongly typed domain object T.

        Fails early with ValidationErrorList if parsing or constraint checks fail.
        """
        if raw_payload is None:
            raise ValidationErrorList(
                errors=[FieldError(field="root", message="Payload cannot be None", invalid_value=None)]
            )

        if not isinstance(raw_payload, dict):
            raise ValidationErrorList(
                errors=[
                    FieldError(
                        field="root",
                        message=f"Expected dictionary payload, got {type(raw_payload).__name__}",
                        invalid_value=raw_payload,
                    )
                ]
            )

        # Handle Pydantic BaseModel types
        if issubclass(self.target_type, BaseModel):
            try:
                return self.target_type.model_validate(raw_payload)
            except PydanticValidationError as exc:
                errors: list[FieldError] = []
                for p_err in exc.errors():
                    field_name = ".".join([str(loc) for loc in p_err["loc"]])
                    errors.append(
                        FieldError(
                            field=field_name,
                            message=p_err["msg"],
                            invalid_value=p_err.get("input"),
                        )
                    )
                raise ValidationErrorList(errors=errors) from exc

        # Handle custom dataclasses / callables
        try:
            return self.target_type(**raw_payload)
        except TypeError as exc:
            raise ValidationErrorList(
                errors=[FieldError(field="type_constructor", message=f"Constructor error: {exc}", invalid_value=raw_payload)]
            ) from exc
        except Exception as exc:
            raise ValidationErrorList(
                errors=[FieldError(field="unknown", message=str(exc), invalid_value=raw_payload)]
            ) from exc


def parse_domain_object(target_type: Type[T], raw_payload: Any) -> T:
    """Convenience function for parsing untrusted raw inputs into strongly typed domain objects."""
    parser = DomainParser(target_type)
    return parser.parse(raw_payload)
