use std::convert::TryInto;

use lightning::chain::keysinterface::ChannelKeys;
use secp256k1::{PublicKey, Secp256k1};
use tonic::{Request, Response, Status, transport::Server};

use signer::*;
use signer::signer_server::{Signer, SignerServer};

use crate::server::my_keys_manager::MyKeysManager;
use crate::server::my_signer::{ChannelId, MySigner};

use super::signer;

impl MySigner {
    fn node_id(node_id: &Vec<u8>) -> Result<PublicKey, Status> {
        node_id.as_slice().try_into().map_err(|_| Status::invalid_argument("node ID"))
            .map(|node_id| PublicKey::from_slice(node_id).unwrap())
    }

    fn channel_id(channel_nonce: &Vec<u8>) -> Result<ChannelId, Status> {
        if channel_nonce.is_empty() {
            Err(Status::invalid_argument("channel ID"))
        } else {
            Ok(ChannelId(channel_nonce.as_slice().try_into().expect("channel_id length != 32")))
        }
    }
}

#[tonic::async_trait]
impl Signer for MySigner {
    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingReply>, Status> {
        log_info!(self, "Got a request: {:?}", request);
        let msg = request.into_inner();

        let reply = signer::PingReply {
            message: format!("Hello {}!", msg.message).into(), // We must use .into_inner() as the fields of gRPC requests and responses are private
        };

        Ok(Response::new(reply))
    }

    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitReply>, Status> {
        log_info!(self, "Got a request: {:?}", request);
        let msg = request.into_inner();
        let hsm_secret = msg.hsm_secret.as_slice().try_into().expect("secret length != 32");

        let node_id = self.new_node_from_seed(hsm_secret);

        let reply = signer::InitReply {
            self_node_id: node_id.serialize().to_vec(),
        };
        Ok(Response::new(reply))
    }

    async fn ecdh(&self, _request: Request<EcdhRequest>) -> Result<Response<EcdhReply>, Status> {
        panic!("not implemented")
    }

    async fn new_channel(&self, request: Request<NewChannelRequest>) -> Result<Response<NewChannelReply>, Status> {
        log_info!(self, "Got a request: {:?}", request);
        let msg: NewChannelRequest = request.into_inner();
        let node_id = MySigner::node_id(&msg.self_node_id)?;
        let channel_id = MySigner::channel_id(&msg.channel_nonce).ok();

        let channel_id_result = self.new_channel(&node_id, msg.channel_value, channel_id).unwrap();
        let reply = NewChannelReply {
            channel_nonce: channel_id_result.0.to_vec(),
        };
        Ok(Response::new(reply))
    }

    async fn get_per_commitment_point(&self, request: Request<GetPerCommitmentPointRequest>) -> Result<Response<GetPerCommitmentPointReply>, Status> {
        log_info!(self, "Got a request: {:?}", request);
        let msg = request.into_inner();
        let node_id = MySigner::node_id(&msg.self_node_id)?;
        let channel_id = MySigner::channel_id(&msg.channel_nonce).expect("must provide channel ID");
        let secp_ctx = Secp256k1::signing_only();

        let point = self.with_channel(&node_id, &channel_id, |opt_chan| {
            match opt_chan {
                None => Err(Status::invalid_argument("channel not found")),
                Some(chan) =>
                    Ok(MyKeysManager::per_commitment_point(&secp_ctx, chan.keys.commitment_seed(), msg.n)),
            }
        })?;
        let reply = GetPerCommitmentPointReply {
            per_commitment_point: point.serialize().to_vec(),
            old_secret: vec![], // TODO
        };
        Ok(Response::new(reply))
    }

    async fn sign_funding_tx(&self, _request: Request<SignFundingTxRequest>) -> Result<Response<SignFundingTxReply>, Status> {
        panic!("not implemented")
    }

    async fn sign_remote_commitment_tx(&self, _request: Request<SignRemoteCommitmentTxRequest>) -> Result<Response<SignRemoteCommitmentTxReply>, Status> {
        panic!("not implemented")
    }

    async fn sign_remote_htlc_tx(&self, _request: Request<SignRemoteHtlcTxRequest>) -> Result<Response<SignRemoteHtlcTxReply>, Status> {
        panic!("not implemented")
    }

    async fn sign_mutual_close_tx(&self, _request: Request<SignMutualCloseTxRequest>) -> Result<Response<SignMutualCloseTxReply>, Status> {
        panic!("not implemented")
    }
}

#[tokio::main]
pub async fn start() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let signer = MySigner::new();

    Server::builder()
        .add_service(SignerServer::new(signer))
        .serve(addr)
        .await?;

    Ok(())
}
