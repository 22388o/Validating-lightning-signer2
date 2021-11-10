#![crate_name = "lightning_signer_server"]
#![forbid(unsafe_code)]
#![allow(bare_trait_objects)]
#![allow(ellipsis_inclusive_range_patterns)]

extern crate bitcoin;
extern crate hex;
#[cfg(feature = "grpc")]
extern crate tonic;

use lightning_signer::lightning;

pub mod fslogger;
pub mod persist;
pub mod util;
#[macro_use]
#[cfg(feature = "grpc")]
pub mod client;
#[cfg(feature = "grpc")]
pub mod server;

#[cfg(feature = "chain_test")]
pub mod bitcoind_client;
#[cfg(feature = "chain_test")]
mod convert;
