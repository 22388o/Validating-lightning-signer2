use super::{Database, Error};
use crate::Value;
use async_trait::async_trait;
use cdrs_tokio::authenticators::{NoneAuthenticatorProvider, StaticPasswordAuthenticatorProvider};
use cdrs_tokio::cluster::connection_pool::ConnectionPoolConfig;
use cdrs_tokio::cluster::session::Session as CdrsSession;
use cdrs_tokio::cluster::session::{RustlsSessionBuilder, SessionBuilder, TcpSessionBuilder};
use cdrs_tokio::cluster::topology::NodeDistance;
use cdrs_tokio::cluster::{
    ConnectionManager, NodeInfo, NodeRustlsConfigBuilder, NodeTcpConfigBuilder,
    RustlsConnectionManager, TcpConnectionManager,
};
use cdrs_tokio::consistency::Consistency;
pub use cdrs_tokio::error::Error as CdrsError;
use cdrs_tokio::frame::message_batch::BodyReqBatch;
use cdrs_tokio::frame::{Envelope, TryFromRow, Version};
use cdrs_tokio::load_balancing::node_distance_evaluator::NodeDistanceEvaluator;
use cdrs_tokio::load_balancing::{LoadBalancingStrategy, RoundRobinLoadBalancingStrategy};
use cdrs_tokio::query::{BatchQueryBuilder, PreparedQuery, QueryValues};
use cdrs_tokio::query_values;
use cdrs_tokio::retry::NeverReconnectionPolicy;
use cdrs_tokio::transport::{CdrsTransport, TransportRustls, TransportTcp};
use cdrs_tokio::types::blob::Blob;
use cdrs_tokio::types::IntoRustByIndex;
use cdrs_tokio::{IntoCdrsValue, TryFromRow};
use std::convert::TryInto;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

type TcpSession = CdrsSession<
    TransportTcp,
    TcpConnectionManager,
    RoundRobinLoadBalancingStrategy<TransportTcp, TcpConnectionManager>,
>;

pub struct CassandraTcpDatabase {
    session: TcpSession,
}

type TlsSession = CdrsSession<
    TransportRustls,
    RustlsConnectionManager,
    RoundRobinLoadBalancingStrategy<TransportRustls, RustlsConnectionManager>,
>;

pub struct CassandraTlsDatabase {
    session: TlsSession,
}

#[async_trait]
pub trait CassandraDatabase: Send + Sync {
    async fn prepare(&self, query: &str) -> Result<PreparedQuery, CdrsError>;
    async fn batch(&self, batch: BodyReqBatch) -> Result<Envelope, CdrsError>;
    async fn query_with_values(
        &self,
        query: &str,
        values: QueryValues,
    ) -> Result<Envelope, CdrsError>;
}

fn setup_tokio_log() {
    let subscriber =
        tracing_subscriber::FmtSubscriber::builder().with_max_level(tracing::Level::TRACE).finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}

pub async fn new() -> Result<Arc<dyn Database>, Error> {
    if CassandraTlsDatabase::has_config() {
        Ok(Arc::new(CassandraTlsDatabase::new().await?))
    } else {
        Ok(Arc::new(CassandraTcpDatabase::new().await?))
    }
}

pub async fn new_and_clear() -> Result<Arc<dyn Database>, Error> {
    setup_tokio_log();
    if CassandraTlsDatabase::has_config() {
        let db = CassandraTlsDatabase::new().await?;
        db.clear().await?;
        Ok(Arc::new(db))
    } else {
        let db = CassandraTcpDatabase::new().await?;
        db.clear().await?;
        Ok(Arc::new(db))
    }
}

#[async_trait]
impl CassandraDatabase for CassandraTcpDatabase {
    async fn prepare(&self, query: &str) -> Result<PreparedQuery, CdrsError> {
        self.session.prepare(query).await
    }

    async fn batch(&self, batch: BodyReqBatch) -> Result<Envelope, CdrsError> {
        self.session.batch(batch).await
    }

    async fn query_with_values(
        &self,
        query: &str,
        values: QueryValues,
    ) -> Result<Envelope, CdrsError> {
        self.session.query_with_values(query, values).await
    }
}

#[async_trait]
impl CassandraDatabase for CassandraTlsDatabase {
    async fn prepare(&self, query: &str) -> Result<PreparedQuery, CdrsError> {
        self.session.prepare(query).await
    }

    async fn batch(&self, batch: BodyReqBatch) -> Result<Envelope, CdrsError> {
        self.session.batch(batch).await
    }

    async fn query_with_values(
        &self,
        query: &str,
        values: QueryValues,
    ) -> Result<Envelope, CdrsError> {
        self.session.query_with_values(query, values).await
    }
}

async fn migrate_database<
    T: CdrsTransport + 'static,
    CM: ConnectionManager<T> + 'static,
    LB: LoadBalancingStrategy<T, CM> + Send + Sync + 'static,
>(
    session: &CdrsSession<T, CM, LB>,
) -> Result<(), Error> {
    session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 1}").await?;
    session
        .query(
            "CREATE TABLE IF NOT EXISTS ks.data (client blob, key varchar, version bigint, value blob, primary key (client, key)) WITH clustering order by (key asc)",
        ).await?;
    Ok(())
}

struct NoneNodeDistanceEvaluator;

impl NodeDistanceEvaluator for NoneNodeDistanceEvaluator {
    fn compute_distance(&self, _node: &NodeInfo) -> Option<NodeDistance> {
        None
    }
}

impl CassandraTcpDatabase {
    pub async fn new() -> Result<Self, Error> {
        let uri = std::env::var("CASSANDRA").unwrap_or("172.17.0.2:9042".to_string());

        println!("Connecting to {} ...", uri);

        let session = Self::make_session(uri).await?;
        migrate_database(&session).await?;
        Ok(Self { session })
    }

    async fn make_session(uri: String) -> Result<TcpSession, Error> {
        let cluster_config = NodeTcpConfigBuilder::new()
            .with_contact_point(uri.into())
            .with_authenticator_provider(Arc::new(NoneAuthenticatorProvider))
            .with_version(Version::V4)
            .build()
            .await?;
        let lb = RoundRobinLoadBalancingStrategy::new();
        let session_builder = TcpSessionBuilder::new(lb, cluster_config)
            // do not connect to other nodes for now (useful for naively deploying a cluster on gcloud)
            .with_node_distance_evaluator(Box::new(NoneNodeDistanceEvaluator))
            .with_connection_pool_config(ConnectionPoolConfig::new(16, 16, None));
        Ok(session_builder.build().unwrap())
    }

    pub async fn clear(&self) -> Result<(), Error> {
        self.session.query("TRUNCATE ks.data").await?;
        Ok(())
    }
}

// struct NoCertificateVerification {}
//
// impl rustls::client::ServerCertVerifier for NoCertificateVerification {
//     fn verify_server_cert(&self, _end_entity: &rustls::Certificate, _intermediates: &[rustls::Certificate], _server_name: &rustls::ServerName, _scts: &mut dyn Iterator<Item=&[u8]>, _ocsp_response: &[u8], _now: SystemTime) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
//         Ok(rustls::client::ServerCertVerified::assertion())
//     }
// }
//
impl CassandraTlsDatabase {
    pub async fn new() -> Result<Self, Error> {
        let session = Self::make_session().await?;
        migrate_database(&session).await?;
        Ok(Self { session })
    }

    pub fn has_config() -> bool {
        std::env::var("CASSANDRA_CREDS").is_ok()
    }

    async fn make_session() -> Result<TlsSession, Error> {
        let dir = std::env::var("CASSANDRA_CREDS").unwrap_or("creds".to_string());
        let path = Path::new(&dir);
        let ca_crt = Self::read_cert(path, "ca.crt");
        let client_crt = Self::read_cert(path, "cert");
        let client_key = Self::read_key(path, "key");
        let uri = String::from_utf8(fs::read(path.to_path_buf().join("uri")).unwrap())
            .unwrap()
            .trim()
            .to_string();
        let user = String::from_utf8(fs::read(path.to_path_buf().join("user")).unwrap())
            .unwrap()
            .trim()
            .to_string();
        let pass = String::from_utf8(fs::read(path.to_path_buf().join("pass")).unwrap())
            .unwrap()
            .trim()
            .to_string();
        println!("Connecting to {} ...", uri);
        println!("Connecting to {:?} ...", std::net::ToSocketAddrs::to_socket_addrs(&uri).unwrap());

        let mut store = rustls::RootCertStore::empty();
        store.add(&ca_crt).unwrap();
        let tls_config = rustls::ClientConfig::builder()
            .with_safe_defaults()
            // .with_custom_certificate_verifier(Arc::new(NoCertificateVerification {}))
            .with_root_certificates(store)
            .with_single_cert(vec![client_crt], client_key)
            .expect("tls config");
        let (host, _port) = uri.split_once(':').unwrap();
        let auther = StaticPasswordAuthenticatorProvider::new(user, pass);
        let cluster_config =
            NodeRustlsConfigBuilder::new(host.try_into().unwrap(), Arc::new(tls_config))
                .with_contact_point(uri.into())
                .with_authenticator_provider(Arc::new(auther))
                .build()
                .await?;
        let lb = RoundRobinLoadBalancingStrategy::new();
        let session_builder = RustlsSessionBuilder::new(lb, cluster_config)
            // helps with debugging
            .with_reconnection_policy(Arc::new(NeverReconnectionPolicy::default()))
            .with_connection_pool_config(ConnectionPoolConfig::new(16, 16, None));
        Ok(session_builder.build().unwrap())
    }

    fn read_key(path: &Path, name: &str) -> rustls::PrivateKey {
        let key = match rustls_pemfile::read_one(&mut Self::open_file(path, name)).unwrap().unwrap()
        {
            rustls_pemfile::Item::RSAKey(key) => key,
            rustls_pemfile::Item::ECKey(key) => key,
            rustls_pemfile::Item::PKCS8Key(key) => key,
            _ => panic!("expected private key"),
        };
        rustls::PrivateKey(key)
    }

    fn read_cert(dir_path: &Path, path: &str) -> rustls::Certificate {
        let cert = rustls_pemfile::certs(&mut Self::open_file(dir_path, path))
            .unwrap()
            .get(0)
            .unwrap()
            .clone();
        rustls::Certificate(cert)
    }

    fn open_file(dir: &Path, name: &str) -> BufReader<File> {
        let path = dir.to_path_buf().join(name);
        BufReader::new(File::open(&path).expect(path.to_str().unwrap()))
    }

    pub async fn clear(&self) -> Result<(), Error> {
        self.session.query("TRUNCATE ks.data").await?;
        Ok(())
    }
}

#[derive(Clone, Debug, IntoCdrsValue, TryFromRow, PartialEq)]
struct InsertRow {
    client: Blob,
    key: String,
    version: i64,
    value: Blob,
    prev_version: Option<i64>,
}

impl InsertRow {
    fn into_query_values(self) -> QueryValues {
        query_values!(self.version, self.value, self.client, self.key, self.prev_version)
    }
}

#[derive(Clone, Debug, IntoCdrsValue, TryFromRow, PartialEq)]
struct QueryRow {
    client: Blob,
    key: String,
}

impl QueryRow {
    fn into_query_values(self) -> QueryValues {
        query_values!(self.client, self.key)
    }
}

#[derive(Clone, Debug, IntoCdrsValue, TryFromRow, PartialEq)]
struct ResultRow {
    key: String,
    version: i64,
    value: Blob,
}

#[derive(Clone, Debug, IntoCdrsValue, TryFromRow, PartialEq)]
struct ConflictRow {
    key: String,
    version: i64,
}

#[async_trait]
impl Database for dyn CassandraDatabase {
    async fn put(&self, client_id: &[u8], kvs: &Vec<(String, Value)>) -> Result<(), Error> {
        let mut builder: BatchQueryBuilder = BatchQueryBuilder::new();
        let prepared = self
            .prepare("UPDATE ks.data SET version = ?, value = ? WHERE client = ? AND key = ? IF version = ?")
            .await?;
        for (key, value) in kvs {
            let prev_version =
                if value.version == 0 { None } else { Some(value.version as i64 - 1) };
            let row = InsertRow {
                client: Blob::from(client_id),
                key: key.to_string(),
                // we lose one bit here because cassandra only does signed integers
                version: value.version as i64,
                value: Blob::from(value.value.as_slice()),
                prev_version,
            };
            builder = builder.add_query_prepared(&prepared, row.into_query_values());
        }
        builder = builder
            .with_consistency(Consistency::Quorum)
            .with_serial_consistency(Consistency::LocalSerial);
        let batch = builder.build()?;
        let result = self.batch(batch).await?.response_body()?.into_rows().unwrap();

        let mut some_succeeded = false;
        let mut conflicts = Vec::new();
        for (row, (key, _value)) in result.into_iter().zip(kvs) {
            let applied: bool = row.get_by_index(0).unwrap().unwrap();
            if applied {
                some_succeeded = true;
            } else {
                // TODO supply current DB values
                conflicts.push((key.clone(), None));
            }
        }
        if !conflicts.is_empty() && some_succeeded {
            panic!("partial batch success should not happen");
        }
        if conflicts.is_empty() {
            Ok(())
        } else {
            Err(Error::Conflict(conflicts))
        }
    }

    async fn get_with_prefix(
        &self,
        client_id: &[u8],
        key_prefix: String,
    ) -> Result<Vec<(String, Value)>, Error> {
        let query_args =
            QueryRow { client: Blob::from(client_id), key: key_prefix.clone() }.into_query_values();
        let rows = self.query_with_values(
                "SELECT key, version, value FROM ks.data WHERE client = ? AND key >= ? ORDER by key ASC",
                query_args,
            )
            .await?.response_body()?.into_rows().expect("expected result");
        let mut results = Vec::new();
        for r in rows.into_iter() {
            let res = ResultRow::try_from_row(r)?;
            if !res.key.starts_with(&key_prefix) {
                break;
            }
            results.push((
                res.key,
                Value { version: res.version as u64, value: res.value.into_vec() },
            ));
        }
        Ok(results)
    }
}

#[async_trait]
impl Database for CassandraTcpDatabase {
    async fn put(&self, client_id: &[u8], kvs: &Vec<(String, Value)>) -> Result<(), Error> {
        (self as &dyn CassandraDatabase).put(client_id, kvs).await
    }

    async fn get_with_prefix(
        &self,
        client_id: &[u8],
        key_prefix: String,
    ) -> Result<Vec<(String, Value)>, Error> {
        (self as &dyn CassandraDatabase).get_with_prefix(client_id, key_prefix).await
    }
}

#[async_trait]
impl Database for CassandraTlsDatabase {
    async fn put(&self, client_id: &[u8], kvs: &Vec<(String, Value)>) -> Result<(), Error> {
        (self as &dyn CassandraDatabase).put(client_id, kvs).await
    }

    async fn get_with_prefix(
        &self,
        client_id: &[u8],
        key_prefix: String,
    ) -> Result<Vec<(String, Value)>, Error> {
        (self as &dyn CassandraDatabase).get_with_prefix(client_id, key_prefix).await
    }
}
