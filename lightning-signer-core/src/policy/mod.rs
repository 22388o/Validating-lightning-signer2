/// Policy errors
#[macro_use]
pub mod error;
/// Null policy enforcement
#[cfg(feature = "test_utils")]
pub mod null_validator;
/// Basic policy enforcement plus on-chain validation
pub mod onchain_validator;
/// Basic policy enforcement
pub mod simple_validator;
/// Policy enforcement interface
pub mod validator;

use crate::prelude::*;

/// An enforcement policy
pub trait Policy {
    /// A policy error has occured.
    /// Policy errors can be converted to warnings by returning `Ok(())`
    fn policy_error(&self, _tag: String, msg: String) -> Result<(), error::ValidationError>;
}
