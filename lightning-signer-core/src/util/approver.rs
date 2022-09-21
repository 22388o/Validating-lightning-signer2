use lightning::ln::PaymentHash;

use crate::node::InvoiceState;

/// Approve payments
pub trait Approver: Sync + Send {
    ///  Approve an invoice for payment
    fn approve_invoice(&self, hash: &PaymentHash, invoice_state: &InvoiceState) -> bool;
}

/// An approver that always approves
pub struct PositiveApprover();

impl Approver for PositiveApprover {
    fn approve_invoice(&self, _hash: &PaymentHash, _invoice_state: &InvoiceState) -> bool {
        true
    }
}

/// An approver that always declines
pub struct NegativeApprover();

impl Approver for NegativeApprover {
    fn approve_invoice(&self, _hash: &PaymentHash, _invoice_state: &InvoiceState) -> bool {
        false
    }
}
