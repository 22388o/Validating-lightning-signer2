use thiserror::Error;

use lightning_invoice::SignedRawInvoice;

/// Invoice Utility Error
#[derive(Error, Debug, PartialEq)]
pub enum Error {
    /// Lightning Invoice Parse Error
    #[error("parse invoice failed: {0}")]
    ParseError(#[from] lightning_invoice::ParseError),

    /// Unknown Error
    #[error("unknown invoice util error")]
    Unknown,
}

/// Parse a signed invoice
pub fn parse_signed_invoice(rawinvstr: String) -> Result<SignedRawInvoice, Error> {
    // Should SignedRawInvoice handle the lightning: prefix instead?
    let lwrinvstr = rawinvstr.to_lowercase();
    let invstr = lwrinvstr.strip_prefix("lightning:").unwrap_or(&lwrinvstr);

    Ok(invstr.parse::<SignedRawInvoice>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ok() {
        let rv = parse_signed_invoice("lnbc2500u1pvjluezsp5zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zyg3zygspp5qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqypqdq5xysxxatsyp3k7enxv4jsxqzpu9qrsgquk0rl77nj30yxdy8j9vdx85fkpmdla2087ne0xh8nhedh8w27kyke0lp53ut353s06fv3qfegext0eh0ymjpf39tuven09sam30g4vgpfna3rh".to_string());
        assert!(rv.is_ok());
    }

    #[test]
    fn test_parse_error() {
        let rv = parse_signed_invoice("this is not a valid invoice".to_string());
        assert!(rv.is_err());
        let err = rv.unwrap_err();
        assert_eq!(
            err.to_string(),
            "parse invoice failed: Invalid bech32: missing human-readable separator, \"1\""
        );
    }
}
