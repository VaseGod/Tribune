"""Unit tests for 'Parse, Don't Validate' domain boundary enforcement."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from pydantic import BaseModel, Field

from tribune.domain.boundary import (
    ValidationErrorList,
    parse_domain_object,
)


class UserProfile(BaseModel):
    user_id: str = Field(min_length=3)
    age: int = Field(ge=18)
    email: str


@dataclass
class SimpleConfig:
    host: str
    port: int


class TestDomainBoundary(unittest.TestCase):
    def test_successful_pydantic_parsing(self) -> None:
        raw = {"user_id": "usr_100", "age": 25, "email": "user@example.com"}
        profile = parse_domain_object(UserProfile, raw)
        self.assertIsInstance(profile, UserProfile)
        self.assertEqual(profile.user_id, "usr_100")
        self.assertEqual(profile.age, 25)

    def test_pydantic_validation_error_list_failures(self) -> None:
        raw_invalid = {"user_id": "a", "age": 15, "email": "invalid"}
        with self.assertRaises(ValidationErrorList) as ctx:
            parse_domain_object(UserProfile, raw_invalid)

        err_list = ctx.exception
        self.assertGreaterEqual(len(err_list.errors), 2)

        err_dict = err_list.to_dict()
        self.assertEqual(err_dict["status"], "parse_error")
        self.assertGreaterEqual(err_dict["error_count"], 2)

    def test_non_dict_payload_rejection(self) -> None:
        with self.assertRaises(ValidationErrorList) as ctx:
            parse_domain_object(UserProfile, "raw_string_input")

        self.assertIn("Expected dictionary payload", str(ctx.exception))

    def test_none_payload_rejection(self) -> None:
        with self.assertRaises(ValidationErrorList) as ctx:
            parse_domain_object(UserProfile, None)

        self.assertIn("cannot be None", str(ctx.exception))

    def test_dataclass_parsing(self) -> None:
        raw = {"host": "localhost", "port": 8080}
        cfg = parse_domain_object(SimpleConfig, raw)
        self.assertIsInstance(cfg, SimpleConfig)
        self.assertEqual(cfg.host, "localhost")
        self.assertEqual(cfg.port, 8080)


if __name__ == "__main__":
    unittest.main()
