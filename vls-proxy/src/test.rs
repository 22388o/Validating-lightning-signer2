use std::os::unix::io::RawFd;
use std::sync::Arc;

use lightning_signer::persist::DummyPersister;
use lightning_signer::persist::Persist;
use log::info;
use nix::sys::socket::{socketpair, AddressFamily, SockFlag, SockType};
use nix::unistd::{close, fork, ForkResult};
use secp256k1::rand::rngs::OsRng;
use secp256k1::Secp256k1;

use vls_protocol_signer::handler::RootHandler;
use vls_protocol_signer::vls_protocol::model::PubKey;
use vls_protocol_signer::vls_protocol::msgs;

use crate::client::{Client, UnixClient};
use crate::connection::UnixConnection;
use crate::signer_loop;

fn run_parent(fd: RawFd) {
    let mut client = UnixClient::new(UnixConnection::new(fd));
    info!("parent: start");
    client.write(msgs::Memleak {}).unwrap();
    info!("parent: {:?}", client.read());
    let secp = Secp256k1::new();
    let mut rng = OsRng::new().unwrap();
    let (_, key) = secp.generate_keypair(&mut rng);

    client
        .write(msgs::ClientHsmFd { peer_id: PubKey(key.serialize()), dbid: 0, capabilities: 0 })
        .unwrap();
    info!("parent: {:?}", client.read());
    let fd = client.recv_fd().expect("fd");
    info!("parent: received fd {}", fd);
    let mut client1 = UnixClient::new(UnixConnection::new(fd));
    client1.write(msgs::Memleak {}).unwrap();
    info!("parent: client1 {:?}", client1.read());
}

pub(crate) fn run_test() {
    info!("starting test");
    let (fd3, fd4) =
        socketpair(AddressFamily::Unix, SockType::Stream, None, SockFlag::empty()).unwrap();
    assert_eq!(fd3, 3);
    assert_eq!(fd4, 4);
    match unsafe { fork() } {
        Ok(ForkResult::Parent { child, .. }) => {
            info!("child pid {}", child);
            close(fd3).unwrap();
            run_parent(fd4)
        }
        Ok(ForkResult::Child) => {
            close(fd4).unwrap();
            let conn = UnixConnection::new(fd3);
            let client = UnixClient::new(conn);
            let persister: Arc<dyn Persist> = Arc::new(DummyPersister {});
            let seed = Some([0; 32]);
            let handler = RootHandler::new(client.id(), seed, persister, vec![]);
            signer_loop(client, handler)
        }
        Err(_) => {}
    }
}
