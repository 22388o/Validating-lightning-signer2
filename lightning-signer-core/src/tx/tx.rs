use core::cmp;
use core::convert::TryInto;
use core::fmt;

use bitcoin::blockdata::opcodes::all::{
    OP_CHECKMULTISIG, OP_CHECKSIG, OP_CHECKSIGVERIFY, OP_CLTV, OP_CSV, OP_DROP, OP_DUP, OP_ELSE,
    OP_ENDIF, OP_EQUAL, OP_EQUALVERIFY, OP_HASH160, OP_IF, OP_IFDUP, OP_NOTIF, OP_PUSHNUM_1,
    OP_PUSHNUM_16, OP_PUSHNUM_2, OP_SIZE, OP_SWAP,
};
use bitcoin::hashes::sha256::Hash as Sha256;
use bitcoin::hashes::{Hash, HashEngine};
use bitcoin::secp256k1;
use bitcoin::secp256k1::{All, Message, PublicKey, Secp256k1, Signature};
use bitcoin::util::address::Payload;
use bitcoin::util::bip143;
use bitcoin::{OutPoint, Script, SigHashType, Transaction, TxIn, TxOut};
use lightning::chain::keysinterface::BaseSign;
use lightning::ln::chan_utils;
use lightning::ln::chan_utils::{
    get_revokeable_redeemscript, make_funding_redeemscript, HTLCOutputInCommitment, TxCreationKeys,
};
use lightning::ln::PaymentHash;

use crate::node::ChannelSetup;
use crate::policy::error::ValidationError;
use crate::policy::error::ValidationError::{Mismatch, ScriptFormat, TransactionFormat};
use crate::tx::script::{
    expect_data, expect_number, expect_op, expect_script_end, get_anchor_redeemscript,
    get_delayed_redeemscript, get_htlc_anchor_redeemscript,
};
use crate::util::crypto_utils::payload_for_p2wpkh;
use crate::util::debug_utils::DebugPayload;
use crate::util::enforcing_trait_impls::EnforcingSigner;

const MAX_DELAY: i64 = 1000;
pub const ANCHOR_SAT: u64 = 330;

// BEGIN NOT TESTED
pub fn get_commitment_transaction_number_obscure_factor(
    local_payment_basepoint: &PublicKey,
    counterparty_payment_basepoint: &PublicKey,
    outbound: bool,
) -> u64 {
    let mut sha = Sha256::engine();

    let their_payment_basepoint = counterparty_payment_basepoint.serialize();
    if outbound {
        sha.input(&local_payment_basepoint.serialize());
        sha.input(&their_payment_basepoint);
    } else {
        sha.input(&their_payment_basepoint);
        sha.input(&local_payment_basepoint.serialize());
    }
    let res = Sha256::from_engine(sha).into_inner();

    ((res[26] as u64) << 5 * 8)
        | ((res[27] as u64) << 4 * 8)
        | ((res[28] as u64) << 3 * 8)
        | ((res[29] as u64) << 2 * 8)
        | ((res[30] as u64) << 1 * 8)
        | ((res[31] as u64) << 0 * 8)
}
// END NOT TESTED

pub fn build_close_tx(
    to_holder_value_sat: u64,
    to_counterparty_value_sat: u64,
    local_shutdown_script: &Script,
    counterparty_shutdown_script: &Script,
    outpoint: OutPoint,
) -> Transaction {
    let txins = {
        let mut ins: Vec<TxIn> = Vec::new();
        ins.push(TxIn {
            previous_output: outpoint,
            script_sig: Script::new(),
            sequence: 0xffffffff,
            witness: Vec::new(),
        });
        ins
    };

    let mut txouts: Vec<(TxOut, ())> = Vec::new();

    if to_counterparty_value_sat > 0 {
        txouts.push((
            TxOut {
                script_pubkey: counterparty_shutdown_script.clone(),
                value: to_counterparty_value_sat,
            },
            (),
        ));
    }

    if to_holder_value_sat > 0 {
        txouts.push((
            TxOut {
                script_pubkey: local_shutdown_script.clone(),
                value: to_holder_value_sat,
            },
            (),
        ));
    }

    sort_outputs(&mut txouts, |_, _| cmp::Ordering::Equal); // Ordering doesnt matter if they used our pubkey...

    let mut outputs: Vec<TxOut> = Vec::new();
    for out in txouts.drain(..) {
        outputs.push(out.0);
    }

    Transaction {
        version: 2,
        lock_time: 0,
        input: txins,
        output: outputs,
    }
}

// BEGIN NOT TESTED
pub fn build_commitment_tx(
    keys: &TxCreationKeys,
    info: &CommitmentInfo2,
    obscured_commitment_transaction_number: u64,
    outpoint: OutPoint,
    option_anchor_outputs: bool,
    workaround_local_funding_pubkey: &PublicKey,
    workaround_remote_funding_pubkey: &PublicKey,
) -> (Transaction, Vec<Script>, Vec<HTLCOutputInCommitment>) {
    let txins = {
        let mut ins: Vec<TxIn> = Vec::new();
        ins.push(TxIn {
            previous_output: outpoint,
            script_sig: Script::new(),
            sequence: ((0x80 as u32) << 8 * 3)
                | ((obscured_commitment_transaction_number >> 3 * 8) as u32),
            witness: Vec::new(),
        });
        ins
    };

    let mut txouts: Vec<(TxOut, (Script, Option<HTLCOutputInCommitment>))> = Vec::new();

    if info.to_countersigner_value_sat > 0 {
        if !option_anchor_outputs {
            let script = payload_for_p2wpkh(&info.to_countersigner_pubkey).script_pubkey();
            txouts.push((
                TxOut {
                    script_pubkey: script.clone(),
                    value: info.to_countersigner_value_sat as u64,
                },
                (script, None),
            ))
        } else {
            let delayed_script = get_delayed_redeemscript(&info.to_countersigner_pubkey);
            txouts.push((
                TxOut {
                    script_pubkey: delayed_script.to_v0_p2wsh(),
                    value: info.to_countersigner_value_sat as u64,
                },
                (delayed_script, None),
            ));
            let anchor_script = get_anchor_redeemscript(workaround_remote_funding_pubkey);
            txouts.push((
                TxOut {
                    script_pubkey: anchor_script.to_v0_p2wsh(),
                    value: ANCHOR_SAT,
                },
                (anchor_script, None),
            ));
        }
    }

    if info.to_broadcaster_value_sat > 0 {
        let redeem_script = get_revokeable_redeemscript(
            &info.revocation_pubkey,
            info.to_self_delay,
            &info.to_broadcaster_delayed_pubkey,
        );
        txouts.push((
            TxOut {
                script_pubkey: redeem_script.to_v0_p2wsh(),
                value: info.to_broadcaster_value_sat as u64,
            },
            (redeem_script, None),
        ));
        if option_anchor_outputs {
            let anchor_script = get_anchor_redeemscript(workaround_local_funding_pubkey);
            txouts.push((
                TxOut {
                    script_pubkey: anchor_script.to_v0_p2wsh(),
                    value: ANCHOR_SAT,
                },
                (anchor_script, None),
            ));
        }
    }

    for out in &info.offered_htlcs {
        let htlc_in_tx = HTLCOutputInCommitment {
            offered: true,
            amount_msat: out.value_sat * 1000,
            cltv_expiry: out.cltv_expiry,
            payment_hash: out.payment_hash,
            transaction_output_index: None,
        };
        let script = if option_anchor_outputs {
            get_htlc_anchor_redeemscript(&htlc_in_tx, &keys)
        } else {
            chan_utils::get_htlc_redeemscript(&htlc_in_tx, &keys)
        };
        let txout = TxOut {
            script_pubkey: script.to_v0_p2wsh(),
            value: out.value_sat,
        };
        txouts.push((txout, (script, Some(htlc_in_tx))));
    }

    for out in &info.received_htlcs {
        let htlc_in_tx = HTLCOutputInCommitment {
            offered: false,
            amount_msat: out.value_sat * 1000,
            cltv_expiry: out.cltv_expiry,
            payment_hash: out.payment_hash,
            transaction_output_index: None,
        };
        let script = if option_anchor_outputs {
            get_htlc_anchor_redeemscript(&htlc_in_tx, &keys)
        } else {
            chan_utils::get_htlc_redeemscript(&htlc_in_tx, &keys)
        };
        let txout = TxOut {
            script_pubkey: script.to_v0_p2wsh(),
            value: out.value_sat,
        };
        txouts.push((txout, (script, Some(htlc_in_tx))));
    }
    sort_outputs(&mut txouts, |a, b| {
        if let &(_, Some(ref a_htlcout)) = a {
            if let &(_, Some(ref b_htlcout)) = b {
                a_htlcout.cltv_expiry.cmp(&b_htlcout.cltv_expiry)
            } else {
                cmp::Ordering::Equal
            }
        } else {
            cmp::Ordering::Equal
        }
    });
    let mut outputs = Vec::with_capacity(txouts.len());
    let mut scripts = Vec::with_capacity(txouts.len());
    let mut htlcs = Vec::new();
    for (idx, mut out) in txouts.drain(..).enumerate() {
        outputs.push(out.0);
        scripts.push((out.1).0.clone());
        if let Some(mut htlc) = (out.1).1.take() {
            htlc.transaction_output_index = Some(idx as u32);
            htlcs.push(htlc);
        }
    }

    (
        Transaction {
            version: 2,
            lock_time: ((0x20 as u32) << 8 * 3)
                | ((obscured_commitment_transaction_number & 0xffffffu64) as u32),
            input: txins,
            output: outputs,
        },
        scripts,
        htlcs,
    )
}
// END NOT TESTED

// Sign a Bitcoin commitment tx or a mutual-close tx
pub(crate) fn sign_commitment(
    secp_ctx: &Secp256k1<All>,
    keys: &EnforcingSigner,
    counterparty_funding_pubkey: &PublicKey,
    tx: &Transaction,
    channel_value_sat: u64,
) -> Result<Signature, secp256k1::Error> {
    let funding_key = keys.funding_key();
    let funding_pubkey = keys.pubkeys().funding_pubkey;
    let channel_funding_redeemscript =
        make_funding_redeemscript(&funding_pubkey, &counterparty_funding_pubkey);

    let commitment_sighash = Message::from_slice(
        &bip143::SigHashCache::new(tx).signature_hash(
            0,
            &channel_funding_redeemscript,
            channel_value_sat,
            SigHashType::All,
        )[..],
    )?;
    Ok(secp_ctx.sign(&commitment_sighash, funding_key))
}

pub fn sort_outputs<T, C: Fn(&T, &T) -> cmp::Ordering>(
    outputs: &mut Vec<(TxOut, T)>,
    tie_breaker: C,
) {
    outputs.sort_unstable_by(|a, b| {
        a.0.value.cmp(&b.0.value).then_with(|| {
            a.0.script_pubkey[..]
                .cmp(&b.0.script_pubkey[..])
                .then_with(|| tie_breaker(&a.1, &b.1))
        })
    });
}

/// Phase 1 HTLC info
#[derive(Clone)]
pub struct HTLCInfo {
    pub value_sat: u64,
    /// RIPEMD160 of 32 bytes hash
    pub payment_hash_hash: [u8; 20],
    /// This is zero (unknown) for offered HTLCs in phase 1
    pub cltv_expiry: u32,
}

// Implement manually so we can have hex encoded payment_hash_hash.
// BEGIN NOT TESTED
impl fmt::Debug for HTLCInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HTLCInfo")
            .field("value_sat", &self.value_sat)
            .field("payment_hash_hash", &hex::encode(&self.payment_hash_hash))
            .field("cltv_expiry", &self.cltv_expiry)
            .finish()
    }
}
// END NOT TESTED

/// Phase 2 HTLC info
#[derive(Debug, Clone)]
pub struct HTLCInfo2 {
    pub value_sat: u64,
    pub payment_hash: PaymentHash,
    /// This is zero for offered HTLCs in phase 1
    pub cltv_expiry: u32,
}

// BEGIN NOT TESTED
#[derive(Debug, Clone)]
pub struct CommitmentInfo2 {
    pub is_counterparty_broadcaster: bool,
    pub to_countersigner_pubkey: PublicKey,
    pub to_countersigner_value_sat: u64,
    /// Broadcaster revocation pubkey
    pub revocation_pubkey: PublicKey,
    pub to_broadcaster_delayed_pubkey: PublicKey,
    pub to_broadcaster_value_sat: u64,
    pub to_self_delay: u16,
    pub offered_htlcs: Vec<HTLCInfo2>,
    pub received_htlcs: Vec<HTLCInfo2>,
}
// END NOT TESTED

#[allow(dead_code)]
pub struct CommitmentInfo {
    pub is_counterparty_broadcaster: bool,
    pub to_countersigner_address: Option<Payload>,
    pub to_countersigner_pubkey: Option<PublicKey>,
    pub to_countersigner_value_sat: u64,
    pub to_countersigner_anchor_count: u16,
    /// Broadcaster revocation pubkey
    pub revocation_pubkey: Option<PublicKey>,
    pub to_broadcaster_delayed_pubkey: Option<PublicKey>,
    pub to_broadcaster_value_sat: u64,
    pub to_self_delay: u16,
    pub to_broadcaster_anchor_count: u16,
    pub offered_htlcs: Vec<HTLCInfo>,
    pub received_htlcs: Vec<HTLCInfo>,
}

// Define manually because Payload's fmt::Debug is lame.
// BEGIN NOT TESTED
impl fmt::Debug for CommitmentInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommitmentInfo")
            .field(
                "is_counterparty_broadcaster",
                &self.is_counterparty_broadcaster,
            )
            // Wrap the to_countersigner_address Payload w/ a nicer printing one.
            .field(
                "to_countersigner_address",
                &self
                    .to_countersigner_address
                    .as_ref()
                    .map(|p| DebugPayload(&p)),
            )
            .field("to_countersigner_pubkey", &self.to_countersigner_pubkey)
            .field(
                "to_countersigner_value_sat",
                &self.to_countersigner_value_sat,
            )
            .field(
                "to_countersigner_anchor_count",
                &self.to_countersigner_anchor_count,
            )
            .field("revocation_pubkey", &self.revocation_pubkey)
            .field(
                "to_broadcaster_delayed_pubkey",
                &self.to_broadcaster_delayed_pubkey,
            )
            .field("to_broadcaster_value_sat", &self.to_broadcaster_value_sat)
            .field("to_self_delay", &self.to_self_delay)
            .field(
                "to_broadcaster_anchor_count",
                &self.to_broadcaster_anchor_count,
            )
            .field("offered_htlcs", &self.offered_htlcs)
            .field("received_htlcs", &self.received_htlcs)
            .finish()
    }
}
// END NOT TESTED

impl CommitmentInfo {
    // FIXME - should the new_for_{holder,counterparty} wrappers move
    // to Validator::make_info_for_{holder,counterparty}?

    pub fn new_for_holder() -> Self {
        CommitmentInfo::new(false)
    }

    pub fn new_for_counterparty() -> Self {
        CommitmentInfo::new(true)
    }

    pub fn new(is_counterparty_broadcaster: bool) -> Self {
        CommitmentInfo {
            is_counterparty_broadcaster,
            to_countersigner_address: None,
            to_countersigner_pubkey: None,
            to_countersigner_value_sat: 0,
            to_countersigner_anchor_count: 0,
            revocation_pubkey: None,
            to_broadcaster_delayed_pubkey: None,
            to_broadcaster_value_sat: 0,
            to_self_delay: 0,
            to_broadcaster_anchor_count: 0,
            offered_htlcs: vec![],
            received_htlcs: vec![],
        }
    }

    pub fn has_to_broadcaster(&self) -> bool {
        self.to_broadcaster_delayed_pubkey.is_some()
    }

    pub fn has_to_countersigner(&self) -> bool {
        self.to_countersigner_address.is_some() || self.to_countersigner_pubkey.is_some()
    }

    pub fn to_broadcaster_anchor_value_sat(&self) -> u64 {
        if self.to_broadcaster_anchor_count == 1 {
            ANCHOR_SAT
        } else {
            0
        }
    }

    pub fn to_countersigner_anchor_value_sat(&self) -> u64 {
        if self.to_countersigner_anchor_count == 1 {
            ANCHOR_SAT
        } else {
            0
        }
    }

    fn parse_to_broadcaster_script(
        &self,
        script: &Script,
    ) -> Result<(Vec<u8>, i64, Vec<u8>), ValidationError> {
        let iter = &mut script.instructions();
        expect_op(iter, OP_IF)?;
        let revocation_pubkey = expect_data(iter)?;
        expect_op(iter, OP_ELSE)?;
        let delay = expect_number(iter)?;
        expect_op(iter, OP_CSV)?;
        expect_op(iter, OP_DROP)?;
        let delayed_pubkey = expect_data(iter)?;
        expect_op(iter, OP_ENDIF)?;
        expect_op(iter, OP_CHECKSIG)?;
        expect_script_end(iter)?;
        Ok((revocation_pubkey, delay, delayed_pubkey))
    }

    fn handle_to_broadcaster_output(
        &mut self,
        out: &TxOut,
        vals: (Vec<u8>, i64, Vec<u8>),
    ) -> Result<(), ValidationError> {
        let (revocation_pubkey, delay, delayed_pubkey) = vals;
        // policy-v1-commitment-singular-to-local
        if self.has_to_broadcaster() {
            return Err(TransactionFormat("already have to local".to_string()));
        }

        if delay < 0 {
            return Err(ScriptFormat("negative delay".to_string())); // NOT TESTED
        }
        if delay > MAX_DELAY {
            return Err(ScriptFormat("delay too large".to_string())); // NOT TESTED
        }

        // This is safe because we checked for negative
        self.to_self_delay = delay as u16;
        self.to_broadcaster_value_sat = out.value;
        self.to_broadcaster_delayed_pubkey = Some(
            PublicKey::from_slice(delayed_pubkey.as_slice())
                .map_err(|err| Mismatch(format!("delayed_pubkey malformed: {}", err)))?,
        ); // NOT TESTED
        self.revocation_pubkey = Some(
            PublicKey::from_slice(revocation_pubkey.as_slice())
                .map_err(|err| Mismatch(format!("revocation_pubkey malformed: {}", err)))?,
        ); // NOT TESTED

        Ok(())
    }

    // BEGIN NOT TESTED
    fn parse_to_countersigner_delayed_script(
        &self,
        script: &Script,
    ) -> Result<Vec<u8>, ValidationError> {
        let iter = &mut script.instructions();
        let pubkey_data = expect_data(iter)?;
        expect_op(iter, OP_CHECKSIGVERIFY)?;
        expect_op(iter, OP_PUSHNUM_1)?;
        expect_op(iter, OP_CSV)?;
        expect_script_end(iter)?;
        Ok(pubkey_data)
    }

    /// 1 block delayed because of anchor usage
    fn handle_to_countersigner_delayed_output(
        &mut self,
        out: &TxOut,
        to_countersigner_delayed_pubkey_data: Vec<u8>,
    ) -> Result<(), ValidationError> {
        // policy-v1-commitment-singular-to-remote
        if self.has_to_countersigner() {
            return Err(TransactionFormat("more than one to remote".to_string()));
        }
        self.to_countersigner_pubkey = Some(
            PublicKey::from_slice(to_countersigner_delayed_pubkey_data.as_slice()).map_err(
                |err| {
                    Mismatch(format!(
                        "to_countersigner delayed pubkey malformed: {}",
                        err
                    ))
                },
            )?,
        );
        self.to_countersigner_value_sat = out.value;
        Ok(())
    }
    // END NOT TESTED

    fn parse_received_htlc_script(
        &self,
        script: &Script,
        option_anchor_outputs: bool,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, i64), ValidationError> {
        let iter = &mut script.instructions();
        expect_op(iter, OP_DUP)?;
        expect_op(iter, OP_HASH160)?;
        let revocation_hash = expect_data(iter)?;
        expect_op(iter, OP_EQUAL)?;
        expect_op(iter, OP_IF)?;
        expect_op(iter, OP_CHECKSIG)?;
        expect_op(iter, OP_ELSE)?;
        let remote_htlc_pubkey = expect_data(iter)?;
        expect_op(iter, OP_SWAP)?;
        expect_op(iter, OP_SIZE)?;
        let thirty_two = expect_number(iter)?;
        if thirty_two != 32 {
            return Err(Mismatch(format!("expected 32, saw {}", thirty_two))); // NOT TESTED
        }
        expect_op(iter, OP_EQUAL)?;
        expect_op(iter, OP_IF)?;
        expect_op(iter, OP_HASH160)?;
        let payment_hash_vec = expect_data(iter)?;
        expect_op(iter, OP_EQUALVERIFY)?;
        expect_op(iter, OP_PUSHNUM_2)?;
        expect_op(iter, OP_SWAP)?;
        let local_htlc_pubkey = expect_data(iter)?;
        expect_op(iter, OP_PUSHNUM_2)?;
        expect_op(iter, OP_CHECKMULTISIG)?;
        expect_op(iter, OP_ELSE)?;
        expect_op(iter, OP_DROP)?;
        let cltv_expiry = expect_number(iter)?;
        expect_op(iter, OP_CLTV)?;
        expect_op(iter, OP_DROP)?;
        expect_op(iter, OP_CHECKSIG)?;
        expect_op(iter, OP_ENDIF)?;
        if option_anchor_outputs {
            // BEGIN NOT TESTED
            expect_op(iter, OP_PUSHNUM_1)?;
            expect_op(iter, OP_CSV)?;
            expect_op(iter, OP_DROP)?;
            // END NOT TESTED
        }
        expect_op(iter, OP_ENDIF)?;
        expect_script_end(iter)?;
        Ok((
            revocation_hash,
            remote_htlc_pubkey,
            payment_hash_vec,
            local_htlc_pubkey,
            cltv_expiry,
        ))
    }

    fn handle_received_htlc_output(
        &mut self,
        out: &TxOut,
        vals: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, i64),
    ) -> Result<(), ValidationError> {
        let (
            _revocation_hash,
            _remote_htlc_pubkey,
            payment_hash_vec,
            _local_htlc_pubkey,
            cltv_expiry,
        ) = vals;
        let payment_hash_hash = payment_hash_vec
            .as_slice()
            .try_into()
            .map_err(|_| Mismatch("payment hash RIPEMD160 must be length 20".to_string()))?;

        if cltv_expiry < 0 {
            return Err(ScriptFormat("negative CLTV".to_string())); // NOT TESTED
        }

        let cltv_expiry = cltv_expiry as u32;

        let htlc = HTLCInfo {
            value_sat: out.value,
            payment_hash_hash,
            cltv_expiry,
        };
        self.received_htlcs.push(htlc);

        Ok(())
    }

    fn parse_offered_htlc_script(
        &self,
        script: &Script,
        option_anchor_outputs: bool,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), ValidationError> {
        let iter = &mut script.instructions();
        expect_op(iter, OP_DUP)?;
        expect_op(iter, OP_HASH160)?;
        let revocation_hash = expect_data(iter)?;
        expect_op(iter, OP_EQUAL)?;
        expect_op(iter, OP_IF)?;
        expect_op(iter, OP_CHECKSIG)?;
        expect_op(iter, OP_ELSE)?;
        let remote_htlc_pubkey = expect_data(iter)?;
        expect_op(iter, OP_SWAP)?;
        expect_op(iter, OP_SIZE)?;
        let thirty_two = expect_number(iter)?;
        if thirty_two != 32 {
            return Err(Mismatch(format!("expected 32, saw {}", thirty_two))); // NOT TESTED
        }
        expect_op(iter, OP_EQUAL)?;
        expect_op(iter, OP_NOTIF)?;
        expect_op(iter, OP_DROP)?;
        expect_op(iter, OP_PUSHNUM_2)?;
        expect_op(iter, OP_SWAP)?;
        let local_htlc_pubkey = expect_data(iter)?;
        expect_op(iter, OP_PUSHNUM_2)?;
        expect_op(iter, OP_CHECKMULTISIG)?;
        expect_op(iter, OP_ELSE)?;
        expect_op(iter, OP_HASH160)?;
        let payment_hash_vec = expect_data(iter)?;
        expect_op(iter, OP_EQUALVERIFY)?;
        expect_op(iter, OP_CHECKSIG)?;
        expect_op(iter, OP_ENDIF)?;
        if option_anchor_outputs {
            // BEGIN NOT TESTED
            expect_op(iter, OP_PUSHNUM_1)?;
            expect_op(iter, OP_CSV)?;
            expect_op(iter, OP_DROP)?;
            // END NOT TESTED
        }
        expect_op(iter, OP_ENDIF)?;
        expect_script_end(iter)?;
        Ok((
            revocation_hash,
            remote_htlc_pubkey,
            local_htlc_pubkey,
            payment_hash_vec,
        ))
    }

    fn handle_offered_htlc_output(
        &mut self,
        out: &TxOut,
        vals: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
    ) -> Result<(), ValidationError> {
        let (_revocation_hash, _remote_htlc_pubkey, _local_htlc_pubkey, payment_hash_vec) = vals;

        let payment_hash_hash = payment_hash_vec
            .as_slice()
            .try_into()
            .map_err(|_| Mismatch("payment hash RIPEMD160 must be length 20".to_string()))?;

        let htlc = HTLCInfo {
            value_sat: out.value,
            payment_hash_hash,
            cltv_expiry: 0,
        };
        self.offered_htlcs.push(htlc);

        Ok(())
    }

    fn parse_anchor_script(&self, script: &Script) -> Result<Vec<u8>, ValidationError> {
        let iter = &mut script.instructions();
        let to_pubkey_data = expect_data(iter)?;
        expect_op(iter, OP_CHECKSIG)?;
        // BEGIN NOT TESTED
        expect_op(iter, OP_IFDUP)?;
        expect_op(iter, OP_NOTIF)?;
        expect_op(iter, OP_PUSHNUM_16)?;
        expect_op(iter, OP_CSV)?;
        expect_op(iter, OP_ENDIF)?;
        expect_script_end(iter)?;
        Ok(to_pubkey_data)
        // END NOT TESTED
    }

    fn handle_anchor_output(
        &mut self,
        keys: &EnforcingSigner,
        out: &TxOut,
        to_pubkey_data: Vec<u8>,
    ) -> Result<(), ValidationError> {
        let to_pubkey = PublicKey::from_slice(to_pubkey_data.as_slice())
            .map_err(|err| Mismatch(format!("anchor to_pubkey malformed: {}", err)))?;

        // These are dependent on which side owns this commitment.
        let (to_broadcaster_funding_pubkey, to_countersigner_funding_pubkey) =
            if self.is_counterparty_broadcaster {
                // BEGIN NOT TESTED
                (
                    keys.counterparty_pubkeys().funding_pubkey,
                    keys.pubkeys().funding_pubkey,
                )
            // END NOT TESTED
            } else {
                (
                    keys.pubkeys().funding_pubkey,
                    keys.counterparty_pubkeys().funding_pubkey,
                )
            };

        // policy-v1-commitment-anchor-amount
        if out.value != ANCHOR_SAT {
            return Err(Mismatch(format!("anchor wrong size: {}", out.value)));
        }

        if to_pubkey == to_broadcaster_funding_pubkey {
            // local anchor
            self.to_broadcaster_anchor_count += 1; // NOT TESTED
        } else if to_pubkey == to_countersigner_funding_pubkey {
            // remote anchor
            self.to_countersigner_anchor_count += 1; // NOT TESTED
        } else {
            // policy-v1-commitment-anchor-match-fundingkey
            return Err(Mismatch(format!(
                "anchor to_pubkey {} doesn't match local or remote",
                hex::encode(to_pubkey_data)
            )));
        }
        Ok(()) // NOT TESTED
    }

    pub fn handle_output(
        &mut self,
        keys: &EnforcingSigner,
        setup: &ChannelSetup,
        out: &TxOut,
        script_bytes: &[u8],
    ) -> Result<(), ValidationError> {
        // FIXME - This routine is only called on "remote" commitments. Remove this
        // assert when that is not the case ...
        assert!(self.is_counterparty_broadcaster);

        if out.script_pubkey.is_v0_p2wpkh() {
            // FIXME - Does this need it's own policy tag?
            if setup.option_anchor_outputs() {
                return Err(TransactionFormat(
                    "p2wpkh to_countersigner not valid with anchors".to_string(),
                ));
            }
            // policy-v1-commitment-singular-to-remote
            if self.has_to_countersigner() {
                return Err(TransactionFormat(
                    "more than one to_countersigner".to_string(),
                ));
            }
            self.to_countersigner_address = Payload::from_script(&out.script_pubkey);
            self.to_countersigner_value_sat = out.value;
        } else if out.script_pubkey.is_v0_p2wsh() {
            if script_bytes.is_empty() {
                return Err(TransactionFormat("missing witscript for p2wsh".to_string()));
            }
            let script = Script::from(script_bytes.to_vec());
            // FIXME - Does this need it's own policy tag?
            if out.script_pubkey != script.to_v0_p2wsh() {
                return Err(TransactionFormat(
                    "script pubkey doesn't match inner script".to_string(),
                ));
            }
            let vals = self.parse_to_broadcaster_script(&script);
            if vals.is_ok() {
                return self.handle_to_broadcaster_output(out, vals.unwrap());
            }
            let vals = self.parse_received_htlc_script(&script, setup.option_anchor_outputs());
            if vals.is_ok() {
                return self.handle_received_htlc_output(out, vals.unwrap());
            }
            let vals = self.parse_offered_htlc_script(&script, setup.option_anchor_outputs());
            if vals.is_ok() {
                return self.handle_offered_htlc_output(out, vals.unwrap());
            }
            let vals = self.parse_anchor_script(&script);
            if vals.is_ok() {
                return self.handle_anchor_output(keys, out, vals.unwrap());
            }
            if setup.option_anchor_outputs() {
                // BEGIN NOT TESTED
                let vals = self.parse_to_countersigner_delayed_script(&script);
                if vals.is_ok() {
                    return self.handle_to_countersigner_delayed_output(out, vals.unwrap());
                }
                // END NOT TESTED
            }
            // policy-v1-commitment-no-unrecognized-outputs
            return Err(TransactionFormat("unknown p2wsh script".to_string()));
        } else {
            // policy-v1-commitment-no-unrecognized-outputs
            return Err(TransactionFormat("unknown output type".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bitcoin::blockdata::script::Builder;
    use bitcoin::secp256k1::{Secp256k1, SecretKey};
    use bitcoin::{Address, Network};

    use crate::node::CommitmentType;
    use crate::util::test_utils::{
        make_reasonable_test_channel_setup, make_test_channel_keys, make_test_pubkey,
    };

    use super::*;

    #[test]
    fn parse_test_err() {
        let info = CommitmentInfo::new_for_holder();
        let script = Builder::new().into_script();
        let err = info.parse_to_broadcaster_script(&script);
        assert!(err.is_err());
    }

    #[test]
    fn parse_test() {
        let secp_ctx = Secp256k1::signing_only();
        let mut info = CommitmentInfo::new_for_holder();
        let out = TxOut {
            value: 123,
            script_pubkey: Default::default(),
        };
        let revocation_pubkey =
            PublicKey::from_secret_key(&secp_ctx, &SecretKey::from_slice(&[4u8; 32]).unwrap());
        let delayed_pubkey =
            PublicKey::from_secret_key(&secp_ctx, &SecretKey::from_slice(&[3u8; 32]).unwrap());
        let script = get_revokeable_redeemscript(&revocation_pubkey, 5, &delayed_pubkey);
        let vals = info.parse_to_broadcaster_script(&script).unwrap();
        let res = info.handle_to_broadcaster_output(&out, vals);
        assert!(res.is_ok());
        assert!(info.has_to_broadcaster());
        assert!(!info.has_to_countersigner());
        assert_eq!(info.revocation_pubkey.unwrap(), revocation_pubkey);
        assert_eq!(info.to_broadcaster_delayed_pubkey.unwrap(), delayed_pubkey);
        assert_eq!(info.to_self_delay, 5);
        assert_eq!(info.to_broadcaster_value_sat, 123);
        // Make sure you can't do it again (can't have two to_broadcaster outputs).
        let vals = info.parse_to_broadcaster_script(&script);
        let res = info.handle_to_broadcaster_output(&out, vals.unwrap());
        assert!(res.is_err());
        #[rustfmt::skip]
        assert_eq!( // NOT TESTED
            TransactionFormat("already have to local".to_string()),
                    res.expect_err("expecting err")
        );
    }

    #[test]
    fn handle_anchor_wrong_size_test() {
        let mut info = CommitmentInfo::new_for_holder();
        let keys = make_test_channel_keys();
        let out = TxOut {
            value: 329,
            script_pubkey: Default::default(),
        };
        let to_pubkey_data = keys.pubkeys().funding_pubkey.serialize().to_vec();
        let res = info.handle_anchor_output(&keys, &out, to_pubkey_data);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            Mismatch(format!("anchor wrong size: {}", out.value))
        );
    }

    #[test]
    fn handle_anchor_not_local_or_remote_test() {
        let mut info = CommitmentInfo::new_for_holder();
        let keys = make_test_channel_keys();
        let out = TxOut {
            value: 330,
            script_pubkey: Default::default(),
        };
        let to_pubkey_data = make_test_pubkey(42).serialize().to_vec(); // doesn't match
        let res = info.handle_anchor_output(&keys, &out, to_pubkey_data.clone());
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            Mismatch(format!(
                "anchor to_pubkey {} doesn\'t match local or remote",
                hex::encode(to_pubkey_data)
            ))
        );
    }

    #[test]
    fn handle_output_unknown_output_type_test() {
        let mut info = CommitmentInfo::new_for_counterparty();
        let keys = make_test_channel_keys();
        let setup = make_reasonable_test_channel_setup();
        let out = TxOut {
            value: 42,
            script_pubkey: Default::default(),
        };
        let script_bytes = [3u8; 30];
        let res = info.handle_output(&keys, &setup, &out, &script_bytes);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            TransactionFormat("unknown output type".to_string())
        );
    }

    #[test]
    fn handle_output_unknown_p2wsh_script_test() {
        let mut info = CommitmentInfo::new_for_counterparty();
        let keys = make_test_channel_keys();
        let setup = make_reasonable_test_channel_setup();
        let script = Builder::new()
            .push_slice(&[0u8; 42]) // invalid
            .into_script();
        let out = TxOut {
            value: 42,
            script_pubkey: Address::p2wsh(&script, Network::Testnet).script_pubkey(),
        };
        let res = info.handle_output(&keys, &setup, &out, script.as_bytes());
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            TransactionFormat("unknown p2wsh script".to_string())
        );
    }

    #[test]
    fn handle_output_p2wpkh_to_countersigner_with_anchors_test() {
        let mut info = CommitmentInfo::new_for_counterparty();
        let keys = make_test_channel_keys();
        let mut setup = make_reasonable_test_channel_setup();
        setup.commitment_type = CommitmentType::Anchors;
        let pubkey = bitcoin::PublicKey::from_slice(&make_test_pubkey(43).serialize()[..]).unwrap();
        let out = TxOut {
            value: 42,
            script_pubkey: Address::p2wpkh(&pubkey, Network::Testnet)
                .unwrap()
                .script_pubkey(),
        };
        let res = info.handle_output(&keys, &setup, &out, &[0u8; 0]);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            TransactionFormat("p2wpkh to_countersigner not valid with anchors".to_string())
        );
    }

    #[test]
    fn handle_output_more_than_one_to_countersigner_test() {
        let mut info = CommitmentInfo::new_for_counterparty();
        let keys = make_test_channel_keys();
        let setup = make_reasonable_test_channel_setup();
        let pubkey = bitcoin::PublicKey::from_slice(&make_test_pubkey(43).serialize()[..]).unwrap();
        let address = Address::p2wpkh(&pubkey, Network::Testnet).unwrap();
        let out = TxOut {
            value: 42,
            script_pubkey: address.script_pubkey(),
        };

        // Make the info look like a to_remote has already been seen.
        info.to_countersigner_address = Some(address.payload);

        let res = info.handle_output(&keys, &setup, &out, &[0u8; 0]);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            TransactionFormat("more than one to_countersigner".to_string())
        );
    }

    #[test]
    fn handle_output_missing_witscript_test() {
        let mut info = CommitmentInfo::new_for_counterparty();
        let keys = make_test_channel_keys();
        let setup = make_reasonable_test_channel_setup();
        let script = Builder::new().into_script();
        let out = TxOut {
            value: 42,
            script_pubkey: Address::p2wsh(&script, Network::Testnet).script_pubkey(),
        };
        let res = info.handle_output(&keys, &setup, &out, script.as_bytes());
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            TransactionFormat("missing witscript for p2wsh".to_string())
        );
    }

    #[test]
    fn handle_output_script_pubkey_doesnt_match_test() {
        let mut info = CommitmentInfo::new_for_counterparty();
        let keys = make_test_channel_keys();
        let setup = make_reasonable_test_channel_setup();
        let script0 = Builder::new().into_script();
        let script1 = Builder::new().push_slice(&[0u8; 42]).into_script();
        let out = TxOut {
            value: 42,
            script_pubkey: Address::p2wsh(&script0, Network::Testnet).script_pubkey(),
        };
        let res = info.handle_output(&keys, &setup, &out, script1.as_bytes());
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            TransactionFormat("script pubkey doesn\'t match inner script".to_string())
        );
    }
}