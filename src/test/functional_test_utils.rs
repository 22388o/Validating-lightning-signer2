//! A bunch of useful utilities for building networks of nodes and exchanging messages between
//! nodes for functional tests.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use bitcoin::blockdata::block::BlockHeader;
use bitcoin::blockdata::transaction::{Transaction, TxOut};
use bitcoin::Network;
use bitcoin::util::hash::BitcoinHash;
use bitcoin_hashes::Hash;
use bitcoin_hashes::sha256::Hash as Sha256;
use bitcoin_hashes::sha256d::Hash as Sha256d;
use chain::chaininterface;
use chain::keysinterface::KeysInterface;
use chain::transaction::OutPoint;
use lightning::chain;
use lightning::ln;
use lightning::util;
use lightning::util::config::UserConfig;
use ln::channelmanager::{ChannelManager, PaymentHash, PaymentPreimage};
use ln::features::InitFeatures;
use ln::msgs;
use ln::msgs::{ChannelMessageHandler, RoutingMessageHandler};
use ln::router::{Route, Router};
use rand::{Rng, thread_rng};
use secp256k1::key::PublicKey;
use secp256k1::Secp256k1;
use util::errors::APIError;
use util::events::{Event, EventsProvider, MessageSendEvent, MessageSendEventsProvider};
use util::logger::Logger;

use crate::util::enforcing_trait_impls::EnforcingChannelKeys;
use crate::util::test_utils;
use crate::util::test_utils::TestChannelMonitor;

pub const CHAN_CONFIRM_DEPTH: u32 = 100;

pub fn confirm_transaction<'a, 'b: 'a>(notifier: &'a chaininterface::BlockNotifierRef<'b>, chain: &chaininterface::ChainWatchInterfaceUtil, tx: &Transaction, chan_id: u32) {
    assert!(chain.does_match_tx(tx));
    let mut header = BlockHeader { version: 0x20000000, prev_blockhash: Default::default(), merkle_root: Default::default(), time: 42, bits: 42, nonce: 42 };
    notifier.block_connected_checked(&header, 1, &[tx; 1], &[chan_id; 1]);
    for i in 2..CHAN_CONFIRM_DEPTH {
        header = BlockHeader { version: 0x20000000, prev_blockhash: header.bitcoin_hash(), merkle_root: Default::default(), time: 42, bits: 42, nonce: 42 };
        notifier.block_connected_checked(&header, i, &vec![], &[0; 0]);
    }
}

pub fn connect_blocks<'a, 'b>(notifier: &'a chaininterface::BlockNotifierRef<'b>, depth: u32, height: u32, parent: bool, prev_blockhash: Sha256d) -> Sha256d {
    let mut header = BlockHeader { version: 0x2000000, prev_blockhash: if parent { prev_blockhash } else { Default::default() }, merkle_root: Default::default(), time: 42, bits: 42, nonce: 42 };
    notifier.block_connected_checked(&header, height + 1, &Vec::new(), &Vec::new());
    for i in 2..depth + 1 {
        header = BlockHeader { version: 0x20000000, prev_blockhash: header.bitcoin_hash(), merkle_root: Default::default(), time: 42, bits: 42, nonce: 42 };
        notifier.block_connected_checked(&header, height + i, &Vec::new(), &Vec::new());
    }
    header.bitcoin_hash()
}

pub struct NodeCfg {
    pub chain_monitor: Arc<chaininterface::ChainWatchInterfaceUtil>,
    pub tx_broadcaster: Arc<test_utils::TestBroadcaster>,
    pub fee_estimator: Arc<test_utils::TestFeeEstimator>,
    pub chan_monitor: test_utils::TestChannelMonitor,
    pub keys_manager: Arc<test_utils::TestKeysInterface>,
    pub logger: Arc<test_utils::TestLogger>,
    pub node_seed: [u8; 32],
}

pub struct Node<'a, 'b: 'a> {
    pub block_notifier: chaininterface::BlockNotifierRef<'b>,
    pub chain_monitor: Arc<chaininterface::ChainWatchInterfaceUtil>,
    pub tx_broadcaster: Arc<test_utils::TestBroadcaster>,
    pub chan_monitor: &'b test_utils::TestChannelMonitor,
    pub keys_manager: Arc<test_utils::TestKeysInterface>,
    pub node: &'a ChannelManager<EnforcingChannelKeys, &'b TestChannelMonitor>,
    pub router: Router,
    pub node_seed: [u8; 32],
    pub network_payment_count: Rc<RefCell<u8>>,
    pub network_chan_count: Rc<RefCell<u32>>,
    pub logger: Arc<test_utils::TestLogger>,
}

impl<'a, 'b> Drop for Node<'a, 'b> {
    fn drop(&mut self) {
        if !::std::thread::panicking() {
            // Check that we processed all pending events
            assert!(self.node.get_and_clear_pending_msg_events().is_empty());
            assert!(self.node.get_and_clear_pending_events().is_empty());
            assert!(self.chan_monitor.added_monitors.lock().unwrap().is_empty());
        }
    }
}

pub fn create_chan_between_nodes<'a, 'b, 'c>(node_a: &'a Node<'b, 'c>, node_b: &'a Node<'b, 'c>, a_flags: InitFeatures, b_flags: InitFeatures) -> (msgs::ChannelAnnouncement, msgs::ChannelUpdate, msgs::ChannelUpdate, [u8; 32], Transaction) {
    create_chan_between_nodes_with_value(node_a, node_b, 100000, 10001, a_flags, b_flags)
}

pub fn create_chan_between_nodes_with_value<'a, 'b, 'c>(node_a: &'a Node<'b, 'c>, node_b: &'a Node<'b, 'c>, channel_value: u64, push_msat: u64, a_flags: InitFeatures, b_flags: InitFeatures) -> (msgs::ChannelAnnouncement, msgs::ChannelUpdate, msgs::ChannelUpdate, [u8; 32], Transaction) {
    let (funding_locked, channel_id, tx) = create_chan_between_nodes_with_value_a(node_a, node_b, channel_value, push_msat, a_flags, b_flags);
    let (announcement, as_update, bs_update) = create_chan_between_nodes_with_value_b(node_a, node_b, &funding_locked);
    (announcement, as_update, bs_update, channel_id, tx)
}

macro_rules! get_revoke_commit_msgs {
	($node: expr, $node_id: expr) => {
		{
			let events = $node.node.get_and_clear_pending_msg_events();
			assert_eq!(events.len(), 2);
			(match events[0] {
				MessageSendEvent::SendRevokeAndACK { ref node_id, ref msg } => {
					assert_eq!(*node_id, $node_id);
					(*msg).clone()
				},
				_ => panic!("Unexpected event"),
			}, match events[1] {
				MessageSendEvent::UpdateHTLCs { ref node_id, ref updates } => {
					assert_eq!(*node_id, $node_id);
					assert!(updates.update_add_htlcs.is_empty());
					assert!(updates.update_fulfill_htlcs.is_empty());
					assert!(updates.update_fail_htlcs.is_empty());
					assert!(updates.update_fail_malformed_htlcs.is_empty());
					assert!(updates.update_fee.is_none());
					updates.commitment_signed.clone()
				},
				_ => panic!("Unexpected event"),
			})
		}
	}
}

macro_rules! get_event_msg {
	($node: expr, $event_type: path, $node_id: expr) => {
		{
			let events = $node.node.get_and_clear_pending_msg_events();
			assert_eq!(events.len(), 1);
			match events[0] {
				$event_type { ref node_id, ref msg } => {
					assert_eq!(*node_id, $node_id);
					(*msg).clone()
				},
				_ => panic!("Unexpected event"),
			}
		}
	}
}

pub fn create_funding_transaction(node: &Node, expected_chan_value: u64, expected_user_chan_id: u64) -> ([u8; 32], Transaction, OutPoint) {
	let chan_id = *node.network_chan_count.borrow();

	let events = node.node.get_and_clear_pending_events();
	assert_eq!(events.len(), 1);
	match events[0] {
		Event::FundingGenerationReady { ref temporary_channel_id, ref channel_value_satoshis, ref output_script, user_channel_id } => {
			assert_eq!(*channel_value_satoshis, expected_chan_value);
			assert_eq!(user_channel_id, expected_user_chan_id);

			let tx = Transaction { version: chan_id as u32, lock_time: 0, input: Vec::new(), output: vec![TxOut {
				value: *channel_value_satoshis, script_pubkey: output_script.clone(),
			}]};
			let funding_outpoint = OutPoint::new(tx.txid(), 0);
			(*temporary_channel_id, tx, funding_outpoint)
		},
		_ => panic!("Unexpected event"),
	}
}

pub fn create_chan_between_nodes_with_value_init(node_a: &Node, node_b: &Node, channel_value: u64, push_msat: u64, a_flags: InitFeatures, b_flags: InitFeatures) -> Transaction {
	node_a.node.create_channel(node_b.node.get_our_node_id(), channel_value, push_msat, 42).unwrap();
	node_b.node.handle_open_channel(&node_a.node.get_our_node_id(), a_flags, &get_event_msg!(node_a, MessageSendEvent::SendOpenChannel, node_b.node.get_our_node_id()));
	node_a.node.handle_accept_channel(&node_b.node.get_our_node_id(), b_flags, &get_event_msg!(node_b, MessageSendEvent::SendAcceptChannel, node_a.node.get_our_node_id()));

	let (temporary_channel_id, tx, funding_output) = create_funding_transaction(node_a, channel_value, 42);

	{
		node_a.node.funding_transaction_generated(&temporary_channel_id, funding_output);
		let mut added_monitors = node_a.chan_monitor.added_monitors.lock().unwrap();
		assert_eq!(added_monitors.len(), 1);
		assert_eq!(added_monitors[0].0, funding_output);
		added_monitors.clear();
	}

	node_b.node.handle_funding_created(&node_a.node.get_our_node_id(), &get_event_msg!(node_a, MessageSendEvent::SendFundingCreated, node_b.node.get_our_node_id()));
	{
		let mut added_monitors = node_b.chan_monitor.added_monitors.lock().unwrap();
		assert_eq!(added_monitors.len(), 1);
		assert_eq!(added_monitors[0].0, funding_output);
		added_monitors.clear();
	}

		node_a.node.handle_funding_signed(&node_b.node.get_our_node_id(), &get_event_msg!(node_b, MessageSendEvent::SendFundingSigned, node_a.node.get_our_node_id()));
		{
		let mut added_monitors = node_a.chan_monitor.added_monitors.lock().unwrap();
		assert_eq!(added_monitors.len(), 1);
		assert_eq!(added_monitors[0].0, funding_output);
		added_monitors.clear();
	}

	let events_4 = node_a.node.get_and_clear_pending_events();
	assert_eq!(events_4.len(), 1);
	match events_4[0] {
		Event::FundingBroadcastSafe { ref funding_txo, user_channel_id } => {
			assert_eq!(user_channel_id, 42);
			assert_eq!(*funding_txo, funding_output);
		},
		_ => panic!("Unexpected event"),
	};

	tx
}

pub fn create_chan_between_nodes_with_value_confirm_first(node_recv: &Node, node_conf: &Node, tx: &Transaction) {
	confirm_transaction(&node_conf.block_notifier, &node_conf.chain_monitor, &tx, tx.version);
	node_recv.node.handle_funding_locked(&node_conf.node.get_our_node_id(), &get_event_msg!(node_conf, MessageSendEvent::SendFundingLocked, node_recv.node.get_our_node_id()));
}

pub fn create_chan_between_nodes_with_value_confirm_second(node_recv: &Node, node_conf: &Node) -> ((msgs::FundingLocked, msgs::AnnouncementSignatures), [u8; 32]) {
	let channel_id;
	let events_6 = node_conf.node.get_and_clear_pending_msg_events();
	assert_eq!(events_6.len(), 2);
	((match events_6[0] {
		MessageSendEvent::SendFundingLocked { ref node_id, ref msg } => {
			channel_id = msg.channel_id.clone();
			assert_eq!(*node_id, node_recv.node.get_our_node_id());
			msg.clone()
		},
		_ => panic!("Unexpected event"),
	}, match events_6[1] {
		MessageSendEvent::SendAnnouncementSignatures { ref node_id, ref msg } => {
			assert_eq!(*node_id, node_recv.node.get_our_node_id());
			msg.clone()
		},
		_ => panic!("Unexpected event"),
	}), channel_id)
}

pub fn create_chan_between_nodes_with_value_confirm(node_a: &Node, node_b: &Node, tx: &Transaction) -> ((msgs::FundingLocked, msgs::AnnouncementSignatures), [u8; 32]) {
	create_chan_between_nodes_with_value_confirm_first(node_a, node_b, tx);
	confirm_transaction(&node_a.block_notifier, &node_a.chain_monitor, &tx, tx.version);
	create_chan_between_nodes_with_value_confirm_second(node_b, node_a)
}

pub fn create_chan_between_nodes_with_value_a(node_a: &Node, node_b: &Node, channel_value: u64, push_msat: u64, a_flags: InitFeatures, b_flags: InitFeatures) -> ((msgs::FundingLocked, msgs::AnnouncementSignatures), [u8; 32], Transaction) {
	let tx = create_chan_between_nodes_with_value_init(node_a, node_b, channel_value, push_msat, a_flags, b_flags);
	let (msgs, chan_id) = create_chan_between_nodes_with_value_confirm(node_a, node_b, &tx);
	(msgs, chan_id, tx)
}

pub fn create_chan_between_nodes_with_value_b(node_a: &Node, node_b: &Node, as_funding_msgs: &(msgs::FundingLocked, msgs::AnnouncementSignatures)) -> (msgs::ChannelAnnouncement, msgs::ChannelUpdate, msgs::ChannelUpdate) {
	node_b.node.handle_funding_locked(&node_a.node.get_our_node_id(), &as_funding_msgs.0);
	let bs_announcement_sigs = get_event_msg!(node_b, MessageSendEvent::SendAnnouncementSignatures, node_a.node.get_our_node_id());
	node_b.node.handle_announcement_signatures(&node_a.node.get_our_node_id(), &as_funding_msgs.1);

	let events_7 = node_b.node.get_and_clear_pending_msg_events();
	assert_eq!(events_7.len(), 1);
	let (announcement, bs_update) = match events_7[0] {
		MessageSendEvent::BroadcastChannelAnnouncement { ref msg, ref update_msg } => {
			(msg, update_msg)
		},
		_ => panic!("Unexpected event"),
	};

	node_a.node.handle_announcement_signatures(&node_b.node.get_our_node_id(), &bs_announcement_sigs);
	let events_8 = node_a.node.get_and_clear_pending_msg_events();
	assert_eq!(events_8.len(), 1);
	let as_update = match events_8[0] {
		MessageSendEvent::BroadcastChannelAnnouncement { ref msg, ref update_msg } => {
			assert!(*announcement == *msg);
			update_msg
		},
		_ => panic!("Unexpected event"),
	};

	*node_a.network_chan_count.borrow_mut() += 1;

	((*announcement).clone(), (*as_update).clone(), (*bs_update).clone())
}

pub fn create_announced_chan_between_nodes(nodes: &Vec<Node>, a: usize, b: usize, a_flags: InitFeatures, b_flags: InitFeatures) -> (msgs::ChannelUpdate, msgs::ChannelUpdate, [u8; 32], Transaction) {
	create_announced_chan_between_nodes_with_value(nodes, a, b, 100000, 10001, a_flags, b_flags)
}

pub fn create_announced_chan_between_nodes_with_value(nodes: &Vec<Node>, a: usize, b: usize, channel_value: u64, push_msat: u64, a_flags: InitFeatures, b_flags: InitFeatures) -> (msgs::ChannelUpdate, msgs::ChannelUpdate, [u8; 32], Transaction) {
	let chan_announcement = create_chan_between_nodes_with_value(&nodes[a], &nodes[b], channel_value, push_msat, a_flags, b_flags);
	for node in nodes {
		assert!(node.router.handle_channel_announcement(&chan_announcement.0).unwrap());
		node.router.handle_channel_update(&chan_announcement.1).unwrap();
		node.router.handle_channel_update(&chan_announcement.2).unwrap();
	}
	(chan_announcement.1, chan_announcement.2, chan_announcement.3, chan_announcement.4)
}

macro_rules! check_spends {
	($tx: expr, $spends_tx: expr) => {
		{
			$tx.verify(|out_point| {
				if out_point.txid == $spends_tx.txid() {
					$spends_tx.output.get(out_point.vout as usize).cloned()
				} else {
					None
				}
			}).unwrap();
		}
	}
}

macro_rules! get_closing_signed_broadcast {
	($node: expr, $dest_pubkey: expr) => {
		{
			let events = $node.get_and_clear_pending_msg_events();
			assert!(events.len() == 1 || events.len() == 2);
			(match events[events.len() - 1] {
				MessageSendEvent::BroadcastChannelUpdate { ref msg } => {
					msg.clone()
				},
				_ => panic!("Unexpected event"),
			}, if events.len() == 2 {
				match events[0] {
					MessageSendEvent::SendClosingSigned { ref node_id, ref msg } => {
						assert_eq!(*node_id, $dest_pubkey);
						Some(msg.clone())
					},
					_ => panic!("Unexpected event"),
				}
			} else { None })
		}
	}
}

pub fn close_channel<'a, 'b>(outbound_node: &Node<'a, 'b>, inbound_node: &Node<'a, 'b>, channel_id: &[u8; 32], funding_tx: Transaction, close_inbound_first: bool) -> (msgs::ChannelUpdate, msgs::ChannelUpdate, Transaction) {
    let (node_a, broadcaster_a, struct_a) = if close_inbound_first { (&inbound_node.node, &inbound_node.tx_broadcaster, inbound_node) } else { (&outbound_node.node, &outbound_node.tx_broadcaster, outbound_node) };
    let (node_b, broadcaster_b) = if close_inbound_first { (&outbound_node.node, &outbound_node.tx_broadcaster) } else { (&inbound_node.node, &inbound_node.tx_broadcaster) };
    let (tx_a, tx_b);

    node_a.close_channel(channel_id).unwrap();
    node_b.handle_shutdown(&node_a.get_our_node_id(), &get_event_msg!(struct_a, MessageSendEvent::SendShutdown, node_b.get_our_node_id()));

    let events_1 = node_b.get_and_clear_pending_msg_events();
    assert!(events_1.len() >= 1);
    let shutdown_b = match events_1[0] {
		MessageSendEvent::SendShutdown { ref node_id, ref msg } => {
			assert_eq!(node_id, &node_a.get_our_node_id());
			msg.clone()
		},
		_ => panic!("Unexpected event"),
	};

	let closing_signed_b = if !close_inbound_first {
		assert_eq!(events_1.len(), 1);
		None
	} else {
		Some(match events_1[1] {
			MessageSendEvent::SendClosingSigned { ref node_id, ref msg } => {
				assert_eq!(node_id, &node_a.get_our_node_id());
				msg.clone()
			},
			_ => panic!("Unexpected event"),
		})
	};

	node_a.handle_shutdown(&node_b.get_our_node_id(), &shutdown_b);
	let (as_update, bs_update) = if close_inbound_first {
		assert!(node_a.get_and_clear_pending_msg_events().is_empty());
		node_a.handle_closing_signed(&node_b.get_our_node_id(), &closing_signed_b.unwrap());
		assert_eq!(broadcaster_a.txn_broadcasted.lock().unwrap().len(), 1);
		tx_a = broadcaster_a.txn_broadcasted.lock().unwrap().remove(0);
		let (as_update, closing_signed_a) = get_closing_signed_broadcast!(node_a, node_b.get_our_node_id());

		node_b.handle_closing_signed(&node_a.get_our_node_id(), &closing_signed_a.unwrap());
		let (bs_update, none_b) = get_closing_signed_broadcast!(node_b, node_a.get_our_node_id());
		assert!(none_b.is_none());
		assert_eq!(broadcaster_b.txn_broadcasted.lock().unwrap().len(), 1);
		tx_b = broadcaster_b.txn_broadcasted.lock().unwrap().remove(0);
		(as_update, bs_update)
	} else {
		let closing_signed_a = get_event_msg!(struct_a, MessageSendEvent::SendClosingSigned, node_b.get_our_node_id());

		node_b.handle_closing_signed(&node_a.get_our_node_id(), &closing_signed_a);
		assert_eq!(broadcaster_b.txn_broadcasted.lock().unwrap().len(), 1);
		tx_b = broadcaster_b.txn_broadcasted.lock().unwrap().remove(0);
		let (bs_update, closing_signed_b) = get_closing_signed_broadcast!(node_b, node_a.get_our_node_id());

		node_a.handle_closing_signed(&node_b.get_our_node_id(), &closing_signed_b.unwrap());
		let (as_update, none_a) = get_closing_signed_broadcast!(node_a, node_b.get_our_node_id());
		assert!(none_a.is_none());
		assert_eq!(broadcaster_a.txn_broadcasted.lock().unwrap().len(), 1);
		tx_a = broadcaster_a.txn_broadcasted.lock().unwrap().remove(0);
		(as_update, bs_update)
	};
	assert_eq!(tx_a, tx_b);
	check_spends!(tx_a, funding_tx);

	(as_update, bs_update, tx_a)
}

pub struct SendEvent {
	pub node_id: PublicKey,
	pub msgs: Vec<msgs::UpdateAddHTLC>,
	pub commitment_msg: msgs::CommitmentSigned,
}
impl SendEvent {
	pub fn from_commitment_update(node_id: PublicKey, updates: msgs::CommitmentUpdate) -> SendEvent {
		assert!(updates.update_fulfill_htlcs.is_empty());
		assert!(updates.update_fail_htlcs.is_empty());
		assert!(updates.update_fail_malformed_htlcs.is_empty());
		assert!(updates.update_fee.is_none());
		SendEvent { node_id: node_id, msgs: updates.update_add_htlcs, commitment_msg: updates.commitment_signed }
	}

	pub fn from_event(event: MessageSendEvent) -> SendEvent {
		match event {
			MessageSendEvent::UpdateHTLCs { node_id, updates } => SendEvent::from_commitment_update(node_id, updates),
			_ => panic!("Unexpected event type!"),
		}
	}

	pub fn from_node(node: &Node) -> SendEvent {
		let mut events = node.node.get_and_clear_pending_msg_events();
		assert_eq!(events.len(), 1);
		SendEvent::from_event(events.pop().unwrap())
	}
}

macro_rules! check_added_monitors {
	($node: expr, $count: expr) => {
		{
			let mut added_monitors = $node.chan_monitor.added_monitors.lock().unwrap();
			assert_eq!(added_monitors.len(), $count);
			added_monitors.clear();
		}
	}
}

macro_rules! commitment_signed_dance {
	($node_a: expr, $node_b: expr, $commitment_signed: expr, $fail_backwards: expr, true /* skip last step */) => {
		{
			check_added_monitors!($node_a, 0);
			assert!($node_a.node.get_and_clear_pending_msg_events().is_empty());
			$node_a.node.handle_commitment_signed(&$node_b.node.get_our_node_id(), &$commitment_signed);
			check_added_monitors!($node_a, 1);
			commitment_signed_dance!($node_a, $node_b, (), $fail_backwards, true, false);
		}
	};
	($node_a: expr, $node_b: expr, (), $fail_backwards: expr, true /* skip last step */, true /* return extra message */, true /* return last RAA */) => {
		{
			let (as_revoke_and_ack, as_commitment_signed) = get_revoke_commit_msgs!($node_a, $node_b.node.get_our_node_id());
			check_added_monitors!($node_b, 0);
			assert!($node_b.node.get_and_clear_pending_msg_events().is_empty());
			$node_b.node.handle_revoke_and_ack(&$node_a.node.get_our_node_id(), &as_revoke_and_ack);
			assert!($node_b.node.get_and_clear_pending_msg_events().is_empty());
			check_added_monitors!($node_b, 1);
			$node_b.node.handle_commitment_signed(&$node_a.node.get_our_node_id(), &as_commitment_signed);
			let (bs_revoke_and_ack, extra_msg_option) = {
				let events = $node_b.node.get_and_clear_pending_msg_events();
				assert!(events.len() <= 2);
				(match events[0] {
					MessageSendEvent::SendRevokeAndACK { ref node_id, ref msg } => {
						assert_eq!(*node_id, $node_a.node.get_our_node_id());
						(*msg).clone()
					},
					_ => panic!("Unexpected event"),
				}, events.get(1).map(|e| e.clone()))
			};
			check_added_monitors!($node_b, 1);
			if $fail_backwards {
				assert!($node_a.node.get_and_clear_pending_events().is_empty());
				assert!($node_a.node.get_and_clear_pending_msg_events().is_empty());
			}
			(extra_msg_option, bs_revoke_and_ack)
		}
	};
	($node_a: expr, $node_b: expr, $commitment_signed: expr, $fail_backwards: expr, true /* skip last step */, false /* return extra message */, true /* return last RAA */) => {
		{
			check_added_monitors!($node_a, 0);
			assert!($node_a.node.get_and_clear_pending_msg_events().is_empty());
			$node_a.node.handle_commitment_signed(&$node_b.node.get_our_node_id(), &$commitment_signed);
			check_added_monitors!($node_a, 1);
			let (extra_msg_option, bs_revoke_and_ack) = commitment_signed_dance!($node_a, $node_b, (), $fail_backwards, true, true, true);
			assert!(extra_msg_option.is_none());
			bs_revoke_and_ack
		}
	};
	($node_a: expr, $node_b: expr, (), $fail_backwards: expr, true /* skip last step */, true /* return extra message */) => {
		{
			let (extra_msg_option, bs_revoke_and_ack) = commitment_signed_dance!($node_a, $node_b, (), $fail_backwards, true, true, true);
			$node_a.node.handle_revoke_and_ack(&$node_b.node.get_our_node_id(), &bs_revoke_and_ack);
			check_added_monitors!($node_a, 1);
			extra_msg_option
		}
	};
	($node_a: expr, $node_b: expr, (), $fail_backwards: expr, true /* skip last step */, false /* no extra message */) => {
		{
			assert!(commitment_signed_dance!($node_a, $node_b, (), $fail_backwards, true, true).is_none());
		}
	};
	($node_a: expr, $node_b: expr, $commitment_signed: expr, $fail_backwards: expr) => {
		{
			commitment_signed_dance!($node_a, $node_b, $commitment_signed, $fail_backwards, true);
			if $fail_backwards {
				expect_pending_htlcs_forwardable!($node_a);
				check_added_monitors!($node_a, 1);
			} else {
				assert!($node_a.node.get_and_clear_pending_msg_events().is_empty());
			}
		}
	}
}

macro_rules! get_payment_preimage_hash {
	($node: expr) => {
		{
			let payment_preimage = PaymentPreimage([*$node.network_payment_count.borrow(); 32]);
			*$node.network_payment_count.borrow_mut() += 1;
			let payment_hash = PaymentHash(Sha256::hash(&payment_preimage.0[..]).into_inner());
			(payment_preimage, payment_hash)
		}
	}
}

macro_rules! expect_pending_htlcs_forwardable {
	($node: expr) => {{
		let events = $node.node.get_and_clear_pending_events();
		assert_eq!(events.len(), 1);
		match events[0] {
			Event::PendingHTLCsForwardable { .. } => { },
			_ => panic!("Unexpected event"),
		};
		$node.node.process_pending_htlc_forwards();
	}}
}

macro_rules! expect_payment_sent {
	($node: expr, $expected_payment_preimage: expr) => {
		let events = $node.node.get_and_clear_pending_events();
		assert_eq!(events.len(), 1);
		match events[0] {
			Event::PaymentSent { ref payment_preimage } => {
				assert_eq!($expected_payment_preimage, *payment_preimage);
			},
			_ => panic!("Unexpected event"),
		}
	}
}

pub fn send_along_route_with_hash<'a, 'b>(origin_node: &Node<'a, 'b>, route: Route, expected_route: &[&Node<'a, 'b>], recv_value: u64, our_payment_hash: PaymentHash) {
    let mut payment_event = {
        origin_node.node.send_payment(route, our_payment_hash).unwrap();
        check_added_monitors!(origin_node, 1);

        let mut events = origin_node.node.get_and_clear_pending_msg_events();
        assert_eq!(events.len(), 1);
        SendEvent::from_event(events.remove(0))
    };
    let mut prev_node = origin_node;

	for (idx, &node) in expected_route.iter().enumerate() {
		assert_eq!(node.node.get_our_node_id(), payment_event.node_id);

		node.node.handle_update_add_htlc(&prev_node.node.get_our_node_id(), &payment_event.msgs[0]);
		check_added_monitors!(node, 0);
		commitment_signed_dance!(node, prev_node, payment_event.commitment_msg, false);

		expect_pending_htlcs_forwardable!(node);

		if idx == expected_route.len() - 1 {
			let events_2 = node.node.get_and_clear_pending_events();
			assert_eq!(events_2.len(), 1);
			match events_2[0] {
				Event::PaymentReceived { ref payment_hash, amt } => {
					assert_eq!(our_payment_hash, *payment_hash);
					assert_eq!(amt, recv_value);
				},
				_ => panic!("Unexpected event"),
			}
		} else {
			let mut events_2 = node.node.get_and_clear_pending_msg_events();
			assert_eq!(events_2.len(), 1);
			check_added_monitors!(node, 1);
			payment_event = SendEvent::from_event(events_2.remove(0));
			assert_eq!(payment_event.msgs.len(), 1);
		}

		prev_node = node;
	}
}

pub fn send_along_route<'a, 'b>(origin_node: &Node<'a, 'b>, route: Route, expected_route: &[&Node<'a, 'b>], recv_value: u64) -> (PaymentPreimage, PaymentHash) {
    let (our_payment_preimage, our_payment_hash) = get_payment_preimage_hash!(origin_node);
    send_along_route_with_hash(origin_node, route, expected_route, recv_value, our_payment_hash);
    (our_payment_preimage, our_payment_hash)
}

pub fn claim_payment_along_route<'a, 'b>(origin_node: &Node<'a, 'b>, expected_route: &[&Node<'a, 'b>], skip_last: bool, our_payment_preimage: PaymentPreimage, expected_amount: u64) {
    assert!(expected_route.last().unwrap().node.claim_funds(our_payment_preimage, expected_amount));
    check_added_monitors!(expected_route.last().unwrap(), 1);

    let mut next_msgs: Option<(msgs::UpdateFulfillHTLC, msgs::CommitmentSigned)> = None;
    let mut expected_next_node = expected_route.last().unwrap().node.get_our_node_id();
    macro_rules! get_next_msgs {
		($node: expr) => {
			{
				let events = $node.node.get_and_clear_pending_msg_events();
				assert_eq!(events.len(), 1);
				match events[0] {
					MessageSendEvent::UpdateHTLCs { ref node_id, updates: msgs::CommitmentUpdate { ref update_add_htlcs, ref update_fulfill_htlcs, ref update_fail_htlcs, ref update_fail_malformed_htlcs, ref update_fee, ref commitment_signed } } => {
						assert!(update_add_htlcs.is_empty());
						assert_eq!(update_fulfill_htlcs.len(), 1);
						assert!(update_fail_htlcs.is_empty());
						assert!(update_fail_malformed_htlcs.is_empty());
						assert!(update_fee.is_none());
						expected_next_node = node_id.clone();
						Some((update_fulfill_htlcs[0].clone(), commitment_signed.clone()))
					},
					_ => panic!("Unexpected event"),
				}
			}
		}
	}

	macro_rules! last_update_fulfill_dance {
		($node: expr, $prev_node: expr) => {
			{
				$node.node.handle_update_fulfill_htlc(&$prev_node.node.get_our_node_id(), &next_msgs.as_ref().unwrap().0);
				check_added_monitors!($node, 0);
				assert!($node.node.get_and_clear_pending_msg_events().is_empty());
				commitment_signed_dance!($node, $prev_node, next_msgs.as_ref().unwrap().1, false);
			}
		}
	}
	macro_rules! mid_update_fulfill_dance {
		($node: expr, $prev_node: expr, $new_msgs: expr) => {
			{
				$node.node.handle_update_fulfill_htlc(&$prev_node.node.get_our_node_id(), &next_msgs.as_ref().unwrap().0);
				check_added_monitors!($node, 1);
				let new_next_msgs = if $new_msgs {
					get_next_msgs!($node)
				} else {
					assert!($node.node.get_and_clear_pending_msg_events().is_empty());
					None
				};
				commitment_signed_dance!($node, $prev_node, next_msgs.as_ref().unwrap().1, false);
				next_msgs = new_next_msgs;
			}
		}
	}

	let mut prev_node = expected_route.last().unwrap();
	for (idx, node) in expected_route.iter().rev().enumerate() {
		assert_eq!(expected_next_node, node.node.get_our_node_id());
		let update_next_msgs = !skip_last || idx != expected_route.len() - 1;
		if next_msgs.is_some() {
			mid_update_fulfill_dance!(node, prev_node, update_next_msgs);
		} else if update_next_msgs {
			next_msgs = get_next_msgs!(node);
		} else {
			assert!(node.node.get_and_clear_pending_msg_events().is_empty());
		}
		if !skip_last && idx == expected_route.len() - 1 {
			assert_eq!(expected_next_node, origin_node.node.get_our_node_id());
		}

		prev_node = node;
	}

	if !skip_last {
		last_update_fulfill_dance!(origin_node, expected_route.first().unwrap());
		expect_payment_sent!(origin_node, our_payment_preimage);
	}
}

pub fn claim_payment<'a, 'b>(origin_node: &Node<'a, 'b>, expected_route: &[&Node<'a, 'b>], our_payment_preimage: PaymentPreimage, expected_amount: u64) {
    claim_payment_along_route(origin_node, expected_route, false, our_payment_preimage, expected_amount);
}

pub const TEST_FINAL_CLTV: u32 = 32;

pub fn route_payment<'a, 'b>(origin_node: &Node<'a, 'b>, expected_route: &[&Node<'a, 'b>], recv_value: u64) -> (PaymentPreimage, PaymentHash) {
    let route = origin_node.router.get_route(&expected_route.last().unwrap().node.get_our_node_id(), None, &Vec::new(), recv_value, TEST_FINAL_CLTV).unwrap();
    assert_eq!(route.hops.len(), expected_route.len());
    for (node, hop) in expected_route.iter().zip(route.hops.iter()) {
        assert_eq!(hop.pubkey, node.node.get_our_node_id());
    }

    send_along_route(origin_node, route, expected_route, recv_value)
}

pub fn route_over_limit<'a, 'b>(origin_node: &Node<'a, 'b>, expected_route: &[&Node<'a, 'b>], recv_value: u64) {
    let route = origin_node.router.get_route(&expected_route.last().unwrap().node.get_our_node_id(), None, &Vec::new(), recv_value, TEST_FINAL_CLTV).unwrap();
    assert_eq!(route.hops.len(), expected_route.len());
    for (node, hop) in expected_route.iter().zip(route.hops.iter()) {
        assert_eq!(hop.pubkey, node.node.get_our_node_id());
    }

    let (_, our_payment_hash) = get_payment_preimage_hash!(origin_node);

    let err = origin_node.node.send_payment(route, our_payment_hash).err().unwrap();
    match err {
		APIError::ChannelUnavailable{err} => assert_eq!(err, "Cannot send value that would put us over the max HTLC value in flight our peer will accept"),
		_ => panic!("Unknown error variants"),
	};
}

pub fn send_payment<'a, 'b>(origin: &Node<'a, 'b>, expected_route: &[&Node<'a, 'b>], recv_value: u64, expected_value: u64) {
    let our_payment_preimage = route_payment(&origin, expected_route, recv_value).0;
    claim_payment(&origin, expected_route, our_payment_preimage, expected_value);
}

pub fn create_node_cfgs(node_count: usize) -> Vec<NodeCfg> {
    let mut nodes = Vec::new();
    let mut rng = thread_rng();

    for i in 0..node_count {
        let logger = Arc::new(test_utils::TestLogger::with_id(format!("node {}", i)));
        let fee_estimator = Arc::new(test_utils::TestFeeEstimator { sat_per_kw: 253 });
        let chain_monitor = Arc::new(chaininterface::ChainWatchInterfaceUtil::new(Network::Testnet, logger.clone() as Arc<Logger>));
        let tx_broadcaster = Arc::new(test_utils::TestBroadcaster { txn_broadcasted: Mutex::new(Vec::new()) });
        let mut seed = [0; 32];
        rng.fill_bytes(&mut seed);
        let keys_manager = Arc::new(test_utils::TestKeysInterface::new(&seed, Network::Testnet, logger.clone() as Arc<Logger>));
        let chan_monitor = test_utils::TestChannelMonitor::new(chain_monitor.clone(), tx_broadcaster.clone(), logger.clone(), fee_estimator.clone());
        nodes.push(NodeCfg { chain_monitor, logger, tx_broadcaster, fee_estimator, chan_monitor, keys_manager, node_seed: seed });
    }

    nodes
}

pub fn create_node_chanmgrs<'a, 'b>(node_count: usize, cfgs: &'a Vec<NodeCfg>, node_config: &[Option<UserConfig>]) -> Vec<ChannelManager<EnforcingChannelKeys, &'a TestChannelMonitor>> {
    let mut chanmgrs = Vec::new();
    for i in 0..node_count {
        let mut default_config = UserConfig::default();
        default_config.channel_options.announced_channel = true;
        default_config.peer_channel_config_limits.force_announced_channel_preference = false;
        let node = ChannelManager::new(Network::Testnet, cfgs[i].fee_estimator.clone(), &cfgs[i].chan_monitor, cfgs[i].tx_broadcaster.clone(), cfgs[i].logger.clone(), cfgs[i].keys_manager.clone(), if node_config[i].is_some() { node_config[i].clone().unwrap() } else { default_config }, 0).unwrap();
        chanmgrs.push(node);
    }

    chanmgrs
}

pub fn create_network<'a, 'b>(node_count: usize, cfgs: &'a Vec<NodeCfg>, chan_mgrs: &'b Vec<ChannelManager<EnforcingChannelKeys, &'a TestChannelMonitor>>) -> Vec<Node<'a, 'b>> {
    let secp_ctx = Secp256k1::new();
    let mut nodes = Vec::new();
    let chan_count = Rc::new(RefCell::new(0));
    let payment_count = Rc::new(RefCell::new(0));

    for i in 0..node_count {
        let block_notifier = chaininterface::BlockNotifier::new(cfgs[i].chain_monitor.clone());
        block_notifier.register_listener(&cfgs[i].chan_monitor.simple_monitor as &chaininterface::ChainListener);
        block_notifier.register_listener(&chan_mgrs[i] as &chaininterface::ChainListener);
        let router = Router::new(PublicKey::from_secret_key(&secp_ctx, &cfgs[i].keys_manager.get_node_secret()), cfgs[i].chain_monitor.clone(), cfgs[i].logger.clone() as Arc<Logger>);
        nodes.push(Node {
            chain_monitor: cfgs[i].chain_monitor.clone(),
            block_notifier,
            tx_broadcaster: cfgs[i].tx_broadcaster.clone(),
            chan_monitor: &cfgs[i].chan_monitor,
            keys_manager: cfgs[i].keys_manager.clone(),
            node: &chan_mgrs[i],
            router,
            node_seed: cfgs[i].node_seed,
            network_chan_count: chan_count.clone(),
            network_payment_count: payment_count.clone(),
            logger: cfgs[i].logger.clone(),
        })
    }

	nodes
}
