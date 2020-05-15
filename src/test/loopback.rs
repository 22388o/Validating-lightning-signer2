use std::sync::Arc;

use bitcoin::{Script, Transaction};
use lightning::chain::keysinterface::{ChannelKeys, KeysInterface};
use lightning::ln::chan_utils::{ChannelPublicKeys, HTLCOutputInCommitment, TxCreationKeys};
use lightning::ln::msgs::UnsignedChannelAnnouncement;
use secp256k1::{PublicKey, Secp256k1, SecretKey, Signature};

use crate::node::node::{ChannelId, ChannelSlot};
use crate::server::my_signer::MySigner;
use crate::util::test_utils::make_test_channel_setup;

/// Adapt MySigner to KeysInterface
pub struct LoopbackSignerKeysInterface {
    pub node_id: PublicKey,
    pub signer: Arc<MySigner>,
}

#[derive(Clone)]
pub struct LoopbackChannelSigner {
    pub node_id: PublicKey,
    pub channel_id: ChannelId,
    pub signer: Arc<MySigner>,
}

impl LoopbackChannelSigner {
    fn new(
        node_id: &PublicKey,
        channel_id: &ChannelId,
        signer: Arc<MySigner>,
    ) -> LoopbackChannelSigner {
        log_info!(signer, "new channel {:?} {:?}", node_id, channel_id);
        LoopbackChannelSigner {
            node_id: *node_id,
            channel_id: *channel_id,
            signer: signer.clone(),
        }
    }
}

impl ChannelKeys for LoopbackChannelSigner {
    // TODO leaking secret key
    fn funding_key(&self) -> &SecretKey {
        let key = self.signer
            .with_channel_slot(&self.node_id, &self.channel_id, |slot| match slot {
                None => Err(self
                    .signer
                    .internal_error(format!("no such channel: {}", self.channel_id))),
                Some(ChannelSlot::Stub(stub)) => Ok(stub.keys.funding_key().clone()),
                Some(ChannelSlot::Ready(chan)) => Ok(chan.keys.funding_key().clone()),
            })
            .expect("funding_key");
        &key
    }

    // TODO leaking secret key
    fn revocation_base_key(&self) -> &SecretKey {
        self.keys.revocation_base_key()
    }

    // TODO leaking secret key
    fn payment_base_key(&self) -> &SecretKey {
        self.keys.payment_base_key()
    }

    // TODO leaking secret key
    fn delayed_payment_base_key(&self) -> &SecretKey {
        self.keys.delayed_payment_base_key()
    }

    // TODO leaking secret key
    fn htlc_base_key(&self) -> &SecretKey {
        self.keys.htlc_base_key()
    }

    // TODO leaking secret key
    fn commitment_seed(&self) -> &[u8; 32] {
        self.keys.commitment_seed()
    }

    // BEGIN NOT TESTED
    fn pubkeys(&self) -> &ChannelPublicKeys {
        self.keys.pubkeys()
    }
    // END NOT TESTED

    // BEGIN NOT TESTED
    fn remote_pubkeys(&self) -> &Option<ChannelPublicKeys> {
        self.keys.remote_pubkeys()
    }
    // END NOT TESTED

    // FIXME - Couldn't this return a declared error signature?
    fn sign_remote_commitment<T: secp256k1::Signing + secp256k1::Verification>(
        &self,
        feerate_per_kw: u64,
        commitment_tx: &Transaction,
        keys: &TxCreationKeys,
        htlcs: &[&HTLCOutputInCommitment],
        to_self_delay: u16,
        _secp_ctx: &Secp256k1<T>,
    ) -> Result<(Signature, Vec<Signature>), ()> {
        let signer = &self.signer;
        log_info!(
            signer,
            "sign_remote_commitment {:?} {:?}",
            self.node_id,
            self.channel_id
        );
        self.signer
            .with_channel_slot(&self.node_id, &self.channel_id, |slot| match slot {
                None => Err(()),
                Some(ChannelSlot::Stub(_)) => Err(()),
                Some(ChannelSlot::Ready(chan)) => chan
                    .sign_remote_commitment(
                        feerate_per_kw,
                        commitment_tx,
                        &keys.per_commitment_point,
                        htlcs,
                        to_self_delay,
                    )
                    .map_err(|_| ()),
            })
    }

    // FIXME - Couldn't this return a declared error signature?
    // BEGIN NOT TESTED
    fn sign_closing_transaction<T: secp256k1::Signing>(
        &self,
        _closing_tx: &Transaction,
        _secp_ctx: &Secp256k1<T>,
    ) -> Result<Signature, ()> {
        unimplemented!()
    }
    // END NOT TESTED

    fn sign_channel_announcement<T: secp256k1::Signing>(
        &self,
        msg: &UnsignedChannelAnnouncement,
        _secp_ctx: &Secp256k1<T>,
    ) -> Result<Signature, ()> {
        let signer = &self.signer;
        log_info!(
            signer,
            "sign_remote_commitment {:?} {:?}",
            self.node_id,
            self.channel_id
        );
        self.signer
            .with_channel_slot(&self.node_id, &self.channel_id, |slot| match slot {
                None => Err(()),
                Some(ChannelSlot::Stub(_)) => Err(()),
                Some(ChannelSlot::Ready(chan)) => {
                    chan.sign_channel_announcement(msg).map_err(|_| ())
                }
            })
    }

    fn set_remote_channel_pubkeys(&mut self, channel_points: &ChannelPublicKeys) {
        let signer = &self.signer;
        log_info!(
            signer,
            "set_remote_channel_keys {:?} {:?}",
            self.node_id,
            self.channel_id
        );
        let _: Result<(), ()> =
            self.signer
                .with_channel_slot(&self.node_id, &self.channel_id, |slot| match slot {
                    None => panic!("channel doesn't exist"),
                    Some(ChannelSlot::Stub(_)) => panic!("channel isn't ready"),
                    Some(ChannelSlot::Ready(chan)) => {
                        chan.keys.inner.set_remote_channel_pubkeys(channel_points);
                        Ok(())
                    }
                });
    }
}

impl KeysInterface for LoopbackSignerKeysInterface {
    type ChanKeySigner = LoopbackChannelSigner;

    // TODO secret key leaking
    fn get_node_secret(&self) -> SecretKey {
        self.signer
            .with_node(&self.node_id, |node_opt| {
                node_opt.map_or(Err(()), |n| Ok(n.get_node_secret()))
            })
            .unwrap()
    }

    fn get_destination_script(&self) -> Script {
        self.signer
            .with_node(&self.node_id, |node_opt| {
                node_opt.map_or(Err(()), |n| Ok(n.get_destination_script()))
            })
            .unwrap()
    }

    fn get_shutdown_pubkey(&self) -> PublicKey {
        self.signer
            .with_node(&self.node_id, |node_opt| {
                node_opt.map_or(Err(()), |n| Ok(n.get_shutdown_pubkey()))
            })
            .unwrap()
    }

    fn get_channel_keys(
        &self,
        channel_id: [u8; 32],
        _inbound: bool,
        _channel_value_sat: u64,
    ) -> Self::ChanKeySigner {
        let channel_id = self
            .signer
            .new_channel(&self.node_id, None, Some(ChannelId(channel_id)))
            .unwrap();
        self.signer
            .ready_channel(&self.node_id, channel_id, make_test_channel_setup())
            .unwrap();
        self.signer
            .with_ready_channel(&self.node_id, &channel_id, |chan| {
                Ok(LoopbackChannelSigner::new(
                    &self.node_id,
                    &channel_id,
                    Arc::clone(&self.signer),
                ))
            })
            .unwrap()
    }

    // TODO secret key leaking
    fn get_onion_rand(&self) -> (SecretKey, [u8; 32]) {
        self.signer
            .with_node(&self.node_id, |node_opt| {
                node_opt.map_or(Err(()), |n| Ok(n.get_onion_rand()))
            })
            .unwrap()
    }

    fn get_channel_id(&self) -> [u8; 32] {
        self.signer
            .with_node(&self.node_id, |node_opt| {
                node_opt.map_or(Err(()), |n| Ok(n.get_channel_id()))
            })
            .unwrap()
    }
}
