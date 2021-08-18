#[cfg(feature = "backtrace")]
use backtrace::Backtrace;

use ValidationErrorKind::{Mismatch, Policy, ScriptFormat, TransactionFormat};

use crate::prelude::*;

/// Kind of validation error
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationErrorKind {
    /// The transaction could not be parsed or had non-standard elements
    TransactionFormat(String),
    /// A scriptPubkey could not be parsed or was non-standard for Lightning
    ScriptFormat(String),
    /// A script element didn't match the channel setup
    Mismatch(String),
    /// A policy was violated
    Policy(String),
}

// Explicit PartialEq which ignores backtrace.
impl PartialEq for ValidationError {
    fn eq(&self, other: &ValidationError) -> bool {
        self.kind == other.kind
    }
}

/// Validation error
#[derive(Clone)]
pub struct ValidationError {
    /// The kind of error
    pub kind: ValidationErrorKind,
    /// A non-resolved backtrace
    #[cfg(feature = "backtrace")]
    pub bt: Backtrace,
}

impl ValidationError {
    /// Resolve the backtrace for display to the user
    #[cfg(feature = "backtrace")]
    pub fn resolved_backtrace(&self) -> Backtrace {
        let mut mve = self.clone();
        mve.bt.resolve();
        mve.bt
    }
}

impl core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self.kind)
    }
}

impl core::fmt::Debug for ValidationError {
    #[cfg(not(feature = "backtrace"))]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ValidationError")
            .field("kind", &self.kind)
            .finish()
    }
    #[cfg(feature = "backtrace")]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ValidationError")
            .field("kind", &self.kind)
            .field("bt", &self.resolved_backtrace())
            .finish()
    }
}

impl Into<String> for ValidationError {
    fn into(self) -> String {
        match self.kind {
            TransactionFormat(s) => "transaction format: ".to_string() + &s,
            ScriptFormat(s) => "script format: ".to_string() + &s,
            Mismatch(s) => "script template mismatch: ".to_string() + &s,
            Policy(s) => "policy failure: ".to_string() + &s,
        }
    }
}

pub(crate) fn transaction_format_error(msg: impl Into<String>) -> ValidationError {
    ValidationError {
        kind: TransactionFormat(msg.into()),
        #[cfg(feature = "backtrace")]
        bt: Backtrace::new_unresolved(),
    }
}

pub(crate) fn script_format_error(msg: impl Into<String>) -> ValidationError {
    ValidationError {
        kind: ScriptFormat(msg.into()),
        #[cfg(feature = "backtrace")]
        bt: Backtrace::new_unresolved(),
    }
}

pub(crate) fn mismatch_error(msg: impl Into<String>) -> ValidationError {
    ValidationError {
        kind: Mismatch(msg.into()),
        #[cfg(feature = "backtrace")]
        bt: Backtrace::new_unresolved(),
    }
}

pub(crate) fn policy_error(msg: impl Into<String>) -> ValidationError {
    ValidationError {
        kind: Policy(msg.into()),
        #[cfg(feature = "backtrace")]
        bt: Backtrace::new_unresolved(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_error_test() {
        assert_eq!(
            format!("{}", transaction_format_error("testing".to_string())),
            "TransactionFormat(\"testing\")"
        );
        assert_eq!(
            Into::<String>::into(transaction_format_error("testing".to_string())),
            "transaction format: testing"
        );
        assert_eq!(
            format!("{}", script_format_error("testing".to_string())),
            "ScriptFormat(\"testing\")"
        );
        assert_eq!(
            Into::<String>::into(script_format_error("testing".to_string())),
            "script format: testing"
        );
        assert_eq!(
            format!("{}", mismatch_error("testing".to_string())),
            "Mismatch(\"testing\")"
        );
        assert_eq!(
            Into::<String>::into(mismatch_error("testing".to_string())),
            "script template mismatch: testing"
        );
        assert_eq!(
            format!("{}", policy_error("testing".to_string())),
            "Policy(\"testing\")"
        );
        assert_eq!(
            Into::<String>::into(policy_error("testing".to_string())),
            "policy failure: testing"
        );
    }
}
