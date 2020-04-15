use bitcoin::Network;

use crate::tx::script::ValidationError;
use crate::tx::script::ValidationError::{Policy, TransactionFormat};
use crate::tx::tx::{CommitmentInfo, CommitmentInfo2};

pub trait Validator {
    /// Phase 1 remote tx validation
    fn validate_remote_tx_phase1(&self, state: &ValidatorState, info: &CommitmentInfo, info2: &CommitmentInfo2) -> Result<(), ValidationError>;
    /// Phase 2 remote tx validation
    fn validate_remote_tx(&self, state: &ValidatorState, info2: &CommitmentInfo2) -> Result<(), ValidationError>;
    /// Validate channel open
    fn validate_channel_open(&self) -> Result<(), ValidationError>;
}

pub struct ValidatorState {
    pub current_height: u32,
}

#[derive(Clone)]
pub struct SimplePolicy {
    /// Minimum delay in blocks
    pub min_delay: u16,
    /// Maximum delay in blocks
    pub max_delay: u16,
    /// Maximum channel value in satoshi
    pub max_channel_size: u64,
    /// amounts below this number of satoshi are not considered important
    pub epsilon: u64,
    /// Maximum number of in-flight HTLCs
    pub max_htlcs: usize,
    /// Maximum value of in-flight HTLCs
    pub max_htlc_value_satoshi: u64,
}

pub struct SimpleValidator {
    pub policy: SimplePolicy,
    pub channel_value_satoshi: u64,
}

/// A validator
///
/// Some of the rules will be implicitly enforced in phase 2, where the signer constructs the
/// transaction rather than receiving it from the caller.  Such rules are marked as
/// "by construction".
///
/// Rules:
/// - Input - the single input must spend the funding output
/// -- by construction
/// - Value - if we are the funder, the value to us of the initial commitment transaction
/// should be equal to our funding value
/// - Format - version, locktime and nsequence must be as specified in BOLT 3
/// -- by construction
/// - Output - the outputs must be at most one to-local, at most one to-remote and HTLCs
/// -- by construction
/// - Funded - if this is not the first commitment, the funding UTXO must be active on chain
/// with enough depth
/// - HTLC in-flight value - the inflight value should not be too large
/// -- done via max_htlc_value_satoshi
/// - Fee - must be in range
/// -- done via epsilon
/// - Number of HTLC outputs - must not be too large
/// -- done via max_htlcs
/// - HTLC routing - each offered HTLC must be balanced via a received HTLC
/// - HTLC receive channel validity - the funding UTXO of the receive channel must be active on chain
/// with enough depth
/// - Our revocation pubkey - must be correct
/// -- by construction
/// - To self delay and HTLC delay - must be within range
/// -- done via min_delay, max_delay
/// - Our payment pubkey - must be correct
/// -- by construction
/// - Our delayed payment pubkey - must be correct
/// -- by construction
/// - Our HTLC pubkey - must be correct
/// -- by construction
/// - Offered payment hash - must be related to received HTLC payment hash
/// - Trimming - outputs are trimmed iff under the dust limit
/// -- done via epsilon
/// - Revocation - the previous commitment transaction was properly revoked by peer disclosing secret.
/// - Note that this requires unbounded storage.
/// - No breach - if signing a local commitment transaction, we must not have revoked it

impl SimpleValidator {
    fn validate_delay(&self, name: &str, delay: u32) -> Result<(), ValidationError> {
        let policy = &self.policy;

        if delay < policy.min_delay as u32 {
            return Err(Policy(format!("{} delay too small", name)));
        }
        if delay > policy.max_delay as u32 {
            return Err(Policy(format!("{} delay too large", name)));
        }

        Ok(())
    }
}

impl Validator for SimpleValidator {
    fn validate_remote_tx_phase1(&self, state: &ValidatorState, info: &CommitmentInfo, info2: &CommitmentInfo2) -> Result<(), ValidationError> {
        if info.to_local_value != info2.to_local_value {
            return Err(TransactionFormat("to_local value mismatch".to_string()));
        }
        if info.to_remote_value != info2.to_remote_value {
            return Err(TransactionFormat("to_remote value mismatch".to_string()));
        }
        if info.to_remote_address.as_ref().unwrap_or(&info2.to_remote_address) != &info2.to_remote_address {
            return Err(TransactionFormat("to_remote address mismatch".to_string()));
        }
        if info.to_local_delayed_key.unwrap_or(info2.to_local_delayed_key) != info2.to_local_delayed_key {
            return Err(TransactionFormat("to_remote address mismatch".to_string()));
        }
        if info.to_local_delay != info2.to_local_delay {
            return Err(TransactionFormat("to_local delay mismatch".to_string()));
        }

        self.validate_remote_tx(state, info2)
    }

    fn validate_remote_tx(&self, state: &ValidatorState, info: &CommitmentInfo2) -> Result<(), ValidationError> {
        let policy = &self.policy;

        self.validate_delay("to_local", info.to_local_delay as u32)?;

        let shortage = self.channel_value_satoshi -
            (info.to_local_value + info.to_remote_value);
        if shortage > policy.epsilon {
            return Err(Policy(format!("channel value short by {} > {}", shortage, policy.epsilon)));
        }

        if info.offered_htlcs.len() + info.received_htlcs.len() > policy.max_htlcs {
            return Err(Policy("too many HTLCs".to_string()));
        }

        let mut htlc_value = 0;

        for htlc in &info.offered_htlcs {
            self.validate_delay("offered HTLC", htlc.cltv_expiry - state.current_height)?;
            htlc_value += htlc.value;
        }

        for htlc in &info.received_htlcs {
            self.validate_delay("received HTLC", htlc.cltv_expiry - state.current_height)?;
            htlc_value += htlc.value;
        }

        if htlc_value > policy.max_htlc_value_satoshi {
            return Err(Policy(format!("sum of HTLC values {} too large", htlc_value)))
        }

        Ok(())
    }

    fn validate_channel_open(&self) -> Result<(), ValidationError> {
        if self.channel_value_satoshi > self.policy.max_channel_size {
            return Err(Policy("channel value too large".to_string()));
        }
        Ok(())
    }
}

pub fn make_simple_policy(network: Network) -> SimplePolicy {
    if network == Network::Bitcoin {
        SimplePolicy {
            min_delay: 60,
            max_delay: 1440,
            max_channel_size: 100_000_000,
            epsilon: 10_000,
            max_htlcs: 1000,
            max_htlc_value_satoshi: 10_000_000,
        }
    } else {
        SimplePolicy {
            min_delay: 6,
            max_delay: 1440,
            max_channel_size: 100_000_000,
            epsilon: 10_000,
            max_htlcs: 1000,
            max_htlc_value_satoshi: 10_000_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use lightning::ln::channelmanager::PaymentHash;

    use crate::tx::tx::HTLCInfo;
    use crate::util::crypto_utils::payload_for_p2wpkh;
    use crate::util::test_utils::make_test_pubkey;

    use super::*;

    #[test]
    fn validate_channel_open_test() {
        let policy = make_simple_policy(Network::Testnet);
        let validator = SimpleValidator { policy: policy.clone(), channel_value_satoshi: 100_000_000 };
        assert!(validator.validate_channel_open().is_ok());
        let validator_large = SimpleValidator { policy: policy.clone(), channel_value_satoshi: 100_000_001 };
        assert!(validator_large.validate_channel_open().is_err());
    }

    fn make_info(to_local_value: u64, to_remote_value: u64,
                 to_local_delay: u16,
                 offered_htlcs: Vec<HTLCInfo>,
                 received_htlcs: Vec<HTLCInfo>) -> CommitmentInfo2 {
        let to_remote_pubkey = make_test_pubkey(1);
        let revocation_key = make_test_pubkey(2);
        let to_local_delayed_key = make_test_pubkey(3);
        let to_remote_address = payload_for_p2wpkh(&to_remote_pubkey);
        CommitmentInfo2 {
            to_remote_address,
            to_remote_value,
            revocation_key,
            to_local_delayed_key,
            to_local_value,
            to_local_delay,
            offered_htlcs,
            received_htlcs,
        }
    }

    fn make_validator() -> SimpleValidator {
        let policy = make_simple_policy(Network::Testnet);
        SimpleValidator { policy: policy.clone(), channel_value_satoshi: 100_000_000 }
    }

    fn make_htlc_info(expiry: u32) -> HTLCInfo {
        HTLCInfo {
            value: 10,
            payment_hash: PaymentHash([0;32]),
            cltv_expiry: expiry
        }
    }

    fn assert_policy_error(res: Result<(), ValidationError>, expected: &str) {
        assert_eq!(res.unwrap_err(), Policy(expected.to_string()));
    }

    #[test]
    fn validate_remote_tx_test() {
        let validator = make_validator();
        let state = ValidatorState { current_height: 1000 };
        let info = make_info(99_000_000, 990_000, 6, vec![], vec![]);
        assert!(validator.validate_remote_tx(&state, &info).is_ok());
    }

    #[test]
    fn validate_remote_tx_shortage_test() {
        let validator = make_validator();
        let state = ValidatorState { current_height: 1000 };
        let info_bad = make_info(99_000_000, 990_000 - 1, 6, vec![], vec![]);
        assert_policy_error(validator.validate_remote_tx(&state, &info_bad),
                            "channel value short by 10001 > 10000");
    }

    #[test]
    fn validate_remote_tx_htlc_count_test() {
        let validator = make_validator();
        let state = ValidatorState { current_height: 1000 };
        let htlcs = (0..1001).map(|_| make_htlc_info(1100)).collect();
        let info_bad = make_info(99_000_000, 990_000, 6, vec![], htlcs);
        assert_policy_error(validator.validate_remote_tx(&state, &info_bad),
                            "too many HTLCs");
    }

    #[test]
    fn validate_remote_tx_htlc_value_test() {
        let validator = make_validator();
        let state = ValidatorState { current_height: 1000 };
        let htlcs = (0..1000).map(|_| HTLCInfo {
            value: 10001,
            payment_hash: PaymentHash([0;32]),
            cltv_expiry: 1100
        }).collect();
        let info_bad = make_info(99_000_000, 990_000, 6, vec![], htlcs);
        assert_policy_error(validator.validate_remote_tx(&state, &info_bad),
                            "sum of HTLC values 10001000 too large");
    }

    #[test]
    fn validate_remote_tx_htlc_delay_test() {
        let validator = make_validator();
        let state = ValidatorState { current_height: 1000 };
        let info_good =
            make_info(99_000_000, 990_000, 6, vec![], vec! [make_htlc_info(1006)]);
        assert!(validator.validate_remote_tx(&state, &info_good).is_ok());
        let info_good =
            make_info(99_000_000, 990_000, 6, vec![], vec! [make_htlc_info(2440)]);
        assert!(validator.validate_remote_tx(&state, &info_good).is_ok());
        let info_bad =
            make_info(99_000_000, 990_000, 6, vec![], vec! [make_htlc_info(1005)]);
        assert_policy_error(validator.validate_remote_tx(&state, &info_bad),
                            "received HTLC delay too small");
        let info_bad =
            make_info(99_000_000, 990_000, 6, vec![], vec! [make_htlc_info(2441)]);
        assert_policy_error(validator.validate_remote_tx(&state, &info_bad),
                            "received HTLC delay too large");
    }
}
